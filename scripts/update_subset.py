"""Update task_sets_subset.json:
- remove 2 from Vague bucket
- remove 3 from Cheapest bucket
- add 5 from Specific_Product (Single Product Search) task set
"""
import json
import pathlib

root = pathlib.Path(__file__).parent
main_path = root / "task_sets" / "task_sets.json"
subset_path = root / "task_sets" / "task_sets_subset.json"

with open(main_path, "r", encoding="utf-8") as f:
    main = json.load(f)
with open(subset_path, "r", encoding="utf-8") as f:
    subset = json.load(f)

# Remove last task from these subset task sets
remove_counts = {
    # Cheapest bucket: remove 3 (distributed across 3 subcategories)
    "Web Mall Agent Benchmark: Cheapest Offer Search": 1,                                           # Cheapest_Product 10 -> 9
    "Web Mall Agent Benchmark: Cheapest Offer for Best Fit (Specific Requirements)": 1,             # 10 -> 9
    "Web Mall Agent Benchmark: Cheapest Offer for Best Fit (Vague Requirements)": 1,                # 6 -> 5
    # Vague bucket: remove 2 (distributed across 2 subcategories)
    "Web Mall Agent Benchmark: Best Fit Product Selection (Vague Requirements)": 1,                 # Best_Fit_Vague 8 -> 7
    "Web Mall Agent Benchmark: Find alternative Products for a given Product (substitute products)": 1,  # Substitute 6 -> 5
}

for ts in subset:
    if ts["name"] in remove_counts:
        n = remove_counts[ts["name"]]
        ts["tasks"] = ts["tasks"][:-n]

# Find Single Product Search task set in main
specific_ts_src = next(
    (ts for ts in main if ts["name"] == "Web Mall Agent Benchmark: Single Product Search"),
    None,
)
if specific_ts_src is None:
    raise SystemExit("Could not find Single Product Search task set in main")

# Copy metadata, take first 5 tasks
new_specific_ts = {k: v for k, v in specific_ts_src.items() if k != "tasks"}
new_specific_ts["tasks"] = specific_ts_src["tasks"][:5]

# Insert at the start (before Cheapest) so it's visually grouped at top
subset.insert(0, new_specific_ts)

with open(subset_path, "w", encoding="utf-8") as f:
    json.dump(subset, f, indent=2, ensure_ascii=False)

# Report new distribution
print("New subset distribution:")
total = 0
for ts in subset:
    cat_counts = {}
    for task in ts["tasks"]:
        c = task["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
    print(f"  [{len(ts['tasks']):2}] {ts['name']}")
    for c, n in cat_counts.items():
        print(f"        {c}: {n}")
    total += len(ts["tasks"])
print(f"  TOTAL: {total} tasks")
