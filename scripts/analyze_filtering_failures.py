"""Per-failure attribution for the filtering run.
Each FN gets assigned to exactly one bucket; FPs likewise.
"""
import json
from collections import Counter
from pathlib import Path

JSONL = Path("results/rag-filtering/gpt-5-mini-medium/benchmark_results_20260426_193413.jsonl")


def norm(u: str) -> str:
    return (u or "").rstrip("/").lower()


def normset(it):
    return {norm(u) for u in (it or []) if u}


with JSONL.open(encoding="utf-8") as f:
    tasks = [json.loads(line) for line in f]

failed = [t for t in tasks if t["metrics"]["f1_score"] < 1.0]
print(f"Total: {len(tasks)} | Failed (F1<1.0): {len(failed)}")
print("=" * 80)

fn_bucket_counts = Counter()
fp_bucket_counts = Counter()
task_buckets = Counter()  # primary attribution per task

for t in failed:
    tid = t["task_id"]
    cat = t.get("task_category", "?")
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

    th = t.get("tool_history", []) or []
    n_search = sum(1 for x in th if x.get("tool_type") == "search")
    n_details = sum(1 for x in th if x.get("tool_type") == "details")

    # --- FN attribution (each FN -> exactly one bucket, priority order) ---
    task_fn_bkts = []
    for url in fns:
        if url in db_missing:
            b = "FN: not in DB (retrieval gap)"
        elif url in removed_all:
            b = "FN: removed by filter"
        elif url in post_all:
            b = "FN: passed filter, agent ignored"
        elif url in pre_all:  # was in pre-filter, neither removed nor in post (shouldn't happen)
            b = "FN: pre-filter only"
        elif url in db_found:
            b = "FN: in DB, never searched"
        else:
            b = "FN: unknown"
        fn_bucket_counts[b] += 1
        task_fn_bkts.append(b)

    # --- FP attribution ---
    task_fp_bkts = []
    for url in fps:
        if url in post_all:
            b = "FP: passed filter -> picked"
        elif url in pre_all:
            b = "FP: in pre-filter only -> picked"
        else:
            b = "FP: never in any search"  # likely hallucinated or from get_product_details only
        fp_bucket_counts[b] += 1
        task_fp_bkts.append(b)

    # --- Task-level primary cause ---
    if err == "GraphRecursionError":
        primary = "Iteration limit"
    elif n_search == 0:
        primary = "Agent did not search"
    elif task_fn_bkts and task_fn_bkts[0].startswith("FN: removed by filter"):
        primary = "Filter removed correct URL"
    elif any(b == "FN: passed filter, agent ignored" for b in task_fn_bkts):
        primary = "Agent ignored URL it saw"
    elif any(b == "FN: in DB, never searched" for b in task_fn_bkts):
        primary = "Agent's queries missed correct URL"
    elif any(b == "FN: not in DB (retrieval gap)" for b in task_fn_bkts):
        primary = "Retrieval gap (URL not in DB)"
    elif task_fp_bkts and not task_fn_bkts:
        primary = "FP only (precision issue)"
    else:
        primary = "Other / mixed"
    task_buckets[primary] += 1

    print(f"{tid[:50]:50}  F1={f1:.2f}  searches={n_search}  details={n_details}  err={err}")
    print(f"   FN={len(fns)} {dict(Counter(task_fn_bkts))}")
    print(f"   FP={len(fps)} {dict(Counter(task_fp_bkts))}")
    print(f"   --> primary: {primary}")
    print()

print("=" * 80)
print("\nFN BUCKETS (across all failed tasks):")
for b, n in fn_bucket_counts.most_common():
    print(f"  {n:3}  {b}")
print("\nFP BUCKETS:")
for b, n in fp_bucket_counts.most_common():
    print(f"  {n:3}  {b}")
print("\nPRIMARY TASK-LEVEL BUCKET:")
for b, n in task_buckets.most_common():
    print(f"  {n:3}  {b}")
