"""Inject FAILURE NOTES into cache/plan_cache.json from a cold-run JSONL.

For every failed task in the cold-run JSONL (f1_score < 1.0), we:
  1. Use cache_keyword_used (or fall back to the task category mapping) to
     pick the matching cache keyword.
  2. Pull a short task excerpt + a natural-language failure description.
  3. Append it to every cache entry of that keyword as a failure note.

Usage:
  python scripts/inject_pitfalls.py <cold_run_jsonl> [--cache CACHE_FILE] \
                                                       [--limit N]

Defaults:
  --cache  cache/plan_cache.json
  --limit  5 failure notes per keyword (most recent failures kept)
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def norm(u: str) -> str:
    return (u or "").rstrip("/").lower()


def normset(it):
    return {norm(u) for u in (it or []) if u}


def derive_failure_description(task: Dict[str, Any]) -> str:
    """Natural-language description of why this task failed.

    Written so the cache-hit LLM can act on it directly without needing to
    interpret an opaque error-class label.
    """
    err_type = (task.get("error_type") or "").strip()
    if err_type == "GraphRecursionError":
        return ("Previous attempt hit the recursion limit before returning a "
                "final answer. Keep tool calls minimal and answer as soon as "
                "the required URLs are clear.")

    parsed = normset(task.get("parsed_urls"))
    expected = normset(task.get("correct_answers"))
    fns = expected - parsed
    fps = parsed - expected

    cache_sum = task.get("cache_summary") or {}
    cs_class = (cache_sum.get("error_class") or "").strip()
    if cs_class == "CacheTemplateOvergeneralizationError":
        return ("Previous attempt reused a plan that was too broad and "
                "returned products that did not match the exact requested "
                "specification. Verify each candidate against the task's "
                "specific constraints before including its URL.")

    if fps and not fns:
        return ("Previous attempt returned extra products that did not match "
                "the task requirements. Be precise: only include URLs whose "
                "product clearly matches every constraint in the task.")
    if fns and not fps:
        return ("Previous attempt missed some of the expected products. "
                "Search across all four shops and check candidates "
                "thoroughly before answering.")
    if fns and fps:
        return ("Previous attempt both missed expected products and added "
                "wrong ones. Re-check both recall (search all four shops) "
                "and precision (match the exact constraints) before "
                "answering.")
    return "Previous attempt failed; cause not deducible from URL diffs."


def short_task_excerpt(task: Dict[str, Any], max_len: int = 160) -> str:
    """Strip XML wrappers / boilerplate and trim."""
    raw = task.get("user_task") or ""
    raw = raw.replace("\\n", " ").replace("\n", " ")
    raw = re.sub(r"<task>|</task>", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    if len(raw) > max_len:
        raw = raw[: max_len - 1].rstrip() + "..."
    return raw


_CATEGORY_FALLBACK = {
    "Specific_Product": "specific_product_search",
    "Best_Fit_Vague": "vague_product_search",
    "Best_Fit_Specific": "vague_product_search",
    "Substitute": "substitute_product_search",
    "Cheapest_Product": "cheapest_product_search",
    "Cheapest_Best_Fit_Specific": "cheapest_specific_search",
    "Cheapest_Best_Fit_Vague": "cheapest_vague_search",
    "Compatible_Product": "compatible_product_search",
}


def keyword_for(task: Dict[str, Any]) -> str:
    kw = (task.get("cache_keyword_used") or "").strip()
    if kw:
        return kw
    cat = (task.get("task_category") or "").strip()
    return _CATEGORY_FALLBACK.get(cat, "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path,
                        help="Path to the cold-run benchmark_results_*.jsonl")
    parser.add_argument("--cache", type=Path,
                        default=Path("cache/plan_cache.json"),
                        help="Plan cache JSON to mutate in place")
    parser.add_argument("--limit", type=int, default=5,
                        help="Maximum pitfalls retained per keyword")
    args = parser.parse_args()

    with args.jsonl.open(encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    failed = [
        t for t in tasks
        if (t.get("metrics", {}) or {}).get("f1_score", 1.0) < 1.0
    ]
    print(f"[pitfalls] {len(failed)}/{len(tasks)} failed tasks in {args.jsonl.name}")

    by_keyword: Dict[str, List[Dict[str, str]]] = {}
    for t in failed:
        kw = keyword_for(t)
        if not kw:
            print(f"  - skip task {t.get('task_id')}: no keyword")
            continue
        excerpt = short_task_excerpt(t)
        if not excerpt:
            continue
        entry = {
            "task_excerpt": excerpt,
            "description": derive_failure_description(t),
            "note": f"task_id={t.get('task_id', '')}",
        }
        by_keyword.setdefault(kw, []).append(entry)

    if not args.cache.exists():
        raise SystemExit(f"cache file not found: {args.cache}")

    with args.cache.open(encoding="utf-8") as f:
        cache_raw = json.load(f)

    if "entries" not in cache_raw or not isinstance(cache_raw["entries"], dict):
        raise SystemExit(f"unexpected cache schema in {args.cache}")

    total_injected = 0
    for kw, entries in cache_raw["entries"].items():
        if not isinstance(entries, list):
            continue
        pits = by_keyword.get(kw, [])
        if not pits:
            continue
        # Cap to most-recent N failures so the prompt stays small.
        pits = pits[-args.limit:]
        for e in entries:
            if not isinstance(e, dict):
                continue
            existing = e.get("pitfalls", []) or []
            # de-dupe by task_excerpt
            seen = {p.get("task_excerpt") for p in existing if isinstance(p, dict)}
            for p in pits:
                if p["task_excerpt"] in seen:
                    continue
                existing.append(p)
                seen.add(p["task_excerpt"])
                total_injected += 1
            e["pitfalls"] = existing[-args.limit:]
        print(f"  + keyword='{kw}': attached {len(pits)} pitfall(s) "
              f"to {len(entries)} template(s)")

    # Refresh meta entry_count to be safe.
    cache_raw.setdefault("_meta", {})["entry_count"] = sum(
        len(v) for v in cache_raw["entries"].values()
        if isinstance(v, list)
    )

    with args.cache.open("w", encoding="utf-8") as f:
        json.dump(cache_raw, f, indent=2, ensure_ascii=False)

    keyword_summary = {kw: len(v) for kw, v in by_keyword.items()}
    print(f"[pitfalls] injected {total_injected} pitfall(s) into {args.cache}")
    print(f"[pitfalls] per-keyword failure counts: {keyword_summary}")


if __name__ == "__main__":
    main()
