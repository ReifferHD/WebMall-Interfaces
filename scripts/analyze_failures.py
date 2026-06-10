"""Per-failure attribution for any RAG run (baseline, filtering, caching, masking).
Each FN/FP is assigned to exactly one bucket; each task gets a primary cause.
"""
import json
import sys
import re
from collections import Counter
from pathlib import Path


def norm(u: str) -> str:
    return (u or "").rstrip("/").lower()


def normset(it):
    return {norm(u) for u in (it or []) if u}


def tokenize_query(text: str):
    return {tok for tok in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(tok) > 1}


def query_similarity(a: str, b: str) -> float:
    ta = tokenize_query(a)
    tb = tokenize_query(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def analyze(jsonl_path: Path, run_label: str):
    with jsonl_path.open(encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f]

    failed = [t for t in tasks if t["metrics"]["f1_score"] < 1.0]
    print("=" * 80)
    print(f"RUN: {run_label}")
    print(f"File: {jsonl_path.name}")
    print(f"Total: {len(tasks)} | Failed (F1<1.0): {len(failed)}")
    print("=" * 80)

    fn_bucket_counts = Counter()
    fp_bucket_counts = Counter()
    task_buckets = Counter()

    for t in failed:
        tid = t["task_id"]
        f1 = t["metrics"]["f1_score"]
        err = t.get("error_type")
        parsed = normset(t.get("parsed_urls"))
        expected = normset(t.get("correct_answers"))
        fns = expected - parsed
        fps = parsed - expected
        db_found = normset(t.get("db_urls_found"))
        db_missing = normset(t.get("db_urls_missing"))

        fdec = t.get("filter_decisions", []) or []
        pre_all = set()
        post_all = set()
        removed_all = set()
        for d in fdec:
            pre_all |= normset(d.get("pre_filter_urls"))
            post_all |= normset(d.get("post_filter_urls"))
            removed_all |= normset(d.get("removed_urls"))

        # Search history: what URLs the agent saw across all searches
        # (post-filter for filtering runs, raw for baseline/caching/masking)
        search_urls_seen = set()
        for sr in (t.get("tool_history") or []):
            if sr.get("tool_type") == "search":
                out = sr.get("tool_output", {})
                search_urls_seen |= normset(out.get("result_urls"))

        # Masking events
        mev = t.get("masking_events", []) or []
        masked_urls = set()
        masked_search_events = []
        masked_detail_events = []
        for m in mev:
            masked_urls |= normset(m.get("masked_urls"))
            if m.get("tool_type") == "search":
                masked_search_events.append(m)
            elif m.get("tool_type") == "details":
                masked_detail_events.append(m)

        cache_hit = t.get("cache_hit", False)
        cache_summary = t.get("cache_summary", {}) or {}
        cache_error_class = cache_summary.get("error_class")
        if not cache_error_class and cache_hit:
            if fps and not fns:
                cache_error_class = "CacheTemplateOvergeneralizationError"
            elif fns:
                cache_error_class = "CacheTemplateMisapplicationError"

        th = t.get("tool_history", []) or []
        n_search = sum(1 for x in th if x.get("tool_type") == "search")
        search_history = [x for x in th if x.get("tool_type") == "search"]
        detail_history = [x for x in th if x.get("tool_type") == "details"]

        masked_research_loop = False
        for event in masked_search_events:
            event_query = event.get("search_query", "")
            event_urls = normset(event.get("masked_urls"))
            event_index = int(event.get("search_call_index") or 0)
            for later in search_history:
                later_query = (later.get("tool_args", {}) or {}).get("query", "")
                later_index = int((later.get("tool_args", {}) or {}).get("call_index", 0) or 0)
                if later_index and event_index and later_index <= event_index:
                    continue
                later_urls = normset(((later.get("tool_output") or {}).get("result_urls")) or [])
                same_query = query_similarity(event_query, later_query) >= 0.6
                repeated_results = bool(event_urls and later_urls and (event_urls & later_urls))
                if same_query or repeated_results:
                    masked_research_loop = True
                    break
            if masked_research_loop:
                break

        masked_detail_not_reused = False
        for event in masked_detail_events:
            event_urls = normset(event.get("masked_urls"))
            event_index = int(event.get("details_call_index") or 0)
            if not event_urls:
                continue
            for later in detail_history:
                later_args = later.get("tool_args", {}) or {}
                later_urls = normset(later_args.get("urls", []))
                if not later_urls:
                    continue
                repeated_detail = bool(event_urls & later_urls)
                later_index = int(later_args.get("call_index", 0) or 0)
                if repeated_detail and (not event_index or not later_index or later_index > event_index):
                    masked_detail_not_reused = True
                    break
            if masked_detail_not_reused:
                break

        masking_summary = t.get("masking_summary", {}) or {}
        summary_masked_lost_count = masking_summary.get(
            "masked_unrecovered_expected_urls_count",
            masking_summary.get("masked_missing_expected_urls_count", 0),
        )
        summary_masked_lost = int(summary_masked_lost_count or 0) > 0
        summary_research_loop = bool(masking_summary.get("re_search_after_mask"))
        summary_detail_refetch = bool(masking_summary.get("detail_refetch_after_mask"))
        summary_recursion_after_masking = bool(masking_summary.get("graph_recursion_after_masking"))
        summary_evidence_loop = bool(masking_summary.get("masked_evidence_before_loop"))
        masking_error_class = masking_summary.get("error_class")
        masking_error_basis = masking_summary.get("error_class_basis")
        if masking_error_basis == "expected_url_masked_and_not_recovered":
            masking_error_class = "MaskedObservation-LostUrlError"
        elif masking_error_basis == "masked_evidence_before_recursion_limit":
            masking_error_class = None
        elif masking_error_class == "MaskedEvidence-LoopError":
            masking_error_class = None

        # FN attribution (priority order)
        task_fn_bkts = []
        for url in fns:
            if url in db_missing:
                b = "FN: not in DB (retrieval gap)"
            elif url in removed_all:
                b = "FN: removed by filter"
            elif url in masked_urls and url not in search_urls_seen:
                # was visible at some point but only via a now-masked observation
                b = "FN: only visible via masked observation"
            elif url in masked_urls:
                b = "FN: visible (also in masked obs), agent ignored"
            elif url in post_all or url in search_urls_seen:
                b = "FN: agent saw URL, did not pick"
            elif url in db_found:
                b = "FN: in DB, never surfaced by agent's queries"
            else:
                b = "FN: unknown"
            fn_bucket_counts[b] += 1
            task_fn_bkts.append(b)

        # FP attribution
        task_fp_bkts = []
        for url in fps:
            if url in post_all or url in search_urls_seen:
                b = "FP: surfaced by search, picked"
            else:
                b = "FP: never surfaced (hallucinated / details-only)"
            fp_bucket_counts[b] += 1
            task_fp_bkts.append(b)

        # Primary task-level cause (priority order: method-specific signals first)
        if masking_error_class:
            primary = masking_error_class
        elif summary_masked_lost:
            primary = "MaskedObservation-LostUrlError"
        elif err == "GraphRecursionError":
            primary = "Iteration limit"
        elif n_search == 0:
            primary = "Agent did not search"
        elif task_fn_bkts and "FN: removed by filter" in task_fn_bkts:
            primary = "Filter removed correct URL"
        elif task_fn_bkts and "FN: only visible via masked observation" in task_fn_bkts:
            primary = "MaskedObservation-LostUrlError"
        elif cache_error_class:
            primary = cache_error_class
        elif "FN: agent saw URL, did not pick" in task_fn_bkts:
            primary = "Agent ignored URL it saw"
        elif "FN: in DB, never surfaced by agent's queries" in task_fn_bkts:
            primary = "Agent's queries missed correct URL"
        elif "FN: not in DB (retrieval gap)" in task_fn_bkts:
            primary = "Retrieval gap (URL not in DB)"
        elif task_fp_bkts and not task_fn_bkts:
            primary = "FP only (precision issue)"
        else:
            primary = "Other / mixed"
        task_buckets[primary] += 1

        print(f"{tid[:55]:55} F1={f1:.2f} cache_hit={cache_hit} mev={len(mev)} fdec={len(fdec)}")
        print(f"   FN={len(fns)} {dict(Counter(task_fn_bkts))}")
        print(f"   FP={len(fps)} {dict(Counter(task_fp_bkts))}")
        if cache_hit or cache_summary.get("causal_flags"):
            print(
                "   cache-signals: "
                f"class={cache_error_class} "
                f"keyword={cache_summary.get('cache_keyword_used', t.get('cache_keyword_used'))} "
                f"score={cache_summary.get('cache_score')} "
                f"expected_visible={cache_summary.get('expected_visible_urls_count')} "
                f"expected_missing_from_tools={cache_summary.get('expected_missing_from_tools_count')} "
                f"expected_seen_not_returned={cache_summary.get('expected_seen_not_returned_count')} "
                f"fp_surfaced={cache_summary.get('fp_surfaced_by_tools_count')} "
                f"flags={cache_summary.get('causal_flags', [])}"
            )
        if masked_research_loop or masked_detail_not_reused or masking_summary.get("causal_flags"):
            print(
                "   masking-signals: "
                f"class={masking_error_class} "
                f"basis={masking_error_basis} "
                f"re_search_loop={summary_research_loop or masked_research_loop} "
                f"detail_refetch={summary_detail_refetch or masked_detail_not_reused} "
                f"masked_lost={summary_masked_lost} "
                f"unrecovered={masking_summary.get('masked_unrecovered_expected_urls_count', 0)} "
                f"recovered_later={masking_summary.get('masked_recovered_later_expected_urls_count', 0)} "
                f"evidence_loop={summary_evidence_loop} "
                f"recursion_after_masking={summary_recursion_after_masking} "
                f"flags={masking_summary.get('causal_flags', [])}"
            )
        print(f"   --> primary: {primary}")
        print()

    print(f"\n{run_label}: FN BUCKETS")
    for b, n in fn_bucket_counts.most_common():
        print(f"  {n:3}  {b}")
    print(f"\n{run_label}: FP BUCKETS")
    for b, n in fp_bucket_counts.most_common():
        print(f"  {n:3}  {b}")
    print(f"\n{run_label}: PRIMARY TASK-LEVEL")
    for b, n in task_buckets.most_common():
        print(f"  {n:3}  {b}")

    return {"fn": fn_bucket_counts, "fp": fp_bucket_counts, "task": task_buckets, "n_failed": len(failed), "n_total": len(tasks)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # default: the GPT-5-mini runs reported in the paper (see RESULTS.md)
        runs = [
            ("BASELINE", Path("results/rag/gpt-5-mini-medium/benchmark_results_20260424_222156.jsonl")),
            ("FILTERING", Path("results/rag-filtering/gpt-5-mini-medium/benchmark_results_20260426_193413.jsonl")),
            ("CACHING-COLD", Path("results/rag-caching/gpt-5-mini-medium/benchmark_results_20260531_203835.jsonl")),
            ("CACHING-WARM-A", Path("results/rag-caching/gpt-5-mini-medium/benchmark_results_20260608_223054.jsonl")),
            ("CACHING-WARM-B", Path("results/rag-caching/gpt-5-mini-medium/benchmark_results_20260608_230115.jsonl")),
            ("MASKING-PLACEHOLDER", Path("results/rag-masking/gpt-5-mini-medium/benchmark_results_20260429_171008.jsonl")),
            ("MASKING-EMPTY", Path("results/rag-masking/gpt-5-mini-medium/benchmark_results_20260531_214959.jsonl")),
        ]
    else:
        runs = [(sys.argv[1].upper(), Path(sys.argv[2]))]

    summaries = {}
    for label, path in runs:
        if not path.exists():
            print(f"SKIP {label}: {path} not found")
            continue
        summaries[label] = analyze(path, label)
        print()

    if len(summaries) > 1:
        print("=" * 80)
        print("CROSS-RUN COMPARISON (primary task-level)")
        print("=" * 80)
        all_keys = set()
        for s in summaries.values():
            all_keys |= set(s["task"].keys())
        labels = list(summaries.keys())
        header = f"{'Class':50}" + "".join(f"{l:>14}" for l in labels)
        print(header)
        print("-" * len(header))
        for k in sorted(all_keys):
            row = f"{k:50}" + "".join(f"{summaries[l]['task'].get(k, 0):>14}" for l in labels)
            print(row)
        print("-" * len(header))
        totals = f"{'TOTAL FAILED':50}" + "".join(f"{summaries[l]['n_failed']:>14}" for l in labels)
        print(totals)
