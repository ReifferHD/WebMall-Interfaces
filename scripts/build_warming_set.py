"""Build task_sets/task_sets_warming.json: all tasks in task_sets.json that
are NOT in task_sets_subset.json, excluding cart/checkout/end-to-end sets.

The warming set is used to populate the plan cache with templates derived from
tasks that are disjoint from the evaluation subset, so the warm-run evaluation
is not measured on tasks already seen by the template generator.
"""
import json
import pathlib

root = pathlib.Path(__file__).parent.parent
main_path = root / "task_sets" / "task_sets.json"
subset_path = root / "task_sets" / "task_sets_subset.json"
warming_path = root / "task_sets" / "task_sets_warming.json"

EXCLUDED_TASK_SETS = {
    "Web Mall Agent Benchmark: Add Product to Cart",
    "Web Mall Agent Benchmark: Checkout and Order",
    "Web Mall Agent Benchmark: Find and Order (End-to-End Webshopping process)",
}

with open(main_path, "r", encoding="utf-8") as f:
    main = json.load(f)
with open(subset_path, "r", encoding="utf-8") as f:
    subset = json.load(f)

subset_ids = set()
for ts in subset:
    for t in ts["tasks"]:
        subset_ids.add(t["id"])

warming = []
for ts in main:
    if ts["name"] in EXCLUDED_TASK_SETS:
        continue
    new_tasks = [t for t in ts["tasks"] if t["id"] not in subset_ids]
    if not new_tasks:
        continue
    new_ts = {k: v for k, v in ts.items() if k != "tasks"}
    new_ts["tasks"] = new_tasks
    warming.append(new_ts)

with open(warming_path, "w", encoding="utf-8") as f:
    json.dump(warming, f, indent=2, ensure_ascii=False)

total = 0
print("Warming task distribution:")
for ts in warming:
    cat_counts = {}
    for task in ts["tasks"]:
        c = task.get("category", "?")
        cat_counts[c] = cat_counts.get(c, 0) + 1
    print(f"  [{len(ts['tasks']):2}] {ts['name']}")
    for c, n in cat_counts.items():
        print(f"        {c}: {n}")
    total += len(ts["tasks"])
print(f"  TOTAL: {total} warming tasks (disjoint from {len(subset_ids)}-task subset)")
