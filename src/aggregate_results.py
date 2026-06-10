"""Aggregate RAG baseline benchmark results into Steiner-style tables.

Reproduces (for the RAG interface only, on the 45-task subset) the table
formats from Steiner et al.:
  - Table 3: Average performance per LLM (CR, F1, Token, Cost, Runtime)
  - Table 4: Average F1 by task set per LLM
  - Tables 5-7: Per task set, per LLM (CR, F1, Token, Cost, Runtime)
"""
import csv
import glob
import os
from collections import defaultdict
from statistics import mean

BASE_RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
# MODIFIED: which interface to aggregate ("rag" baseline or "rag-filtering")
INTERFACE = os.getenv("AGG_INTERFACE", "rag")
RESULTS_DIR = os.path.join(BASE_RESULTS, INTERFACE)
OUT_DIR = os.path.join(RESULTS_DIR, "summary")

MODELS = {
    "gpt-5-mini-medium": "GPT-5-mini",
    "claude-sonnet-4-20250514": "Sonnet 4",
}

# USD per 1M tokens (input / output) — Steiner uses same prices
PRICING = {
    "gpt-5-mini-medium": (0.25, 2.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
}

# Helper-LLM pricing per 1M tokens (input / output)
FILTER_PRICING = (0.05, 0.40)  # gpt-5-nano (filtering optimization)
CACHE_PRICING = (0.15, 0.60)   # gpt-4o-mini (plan caching optimization)

# Map fine-grained subset categories to Steiner's 4 task sets
TASK_SET_MAP = {
    # Specific Product Search
    "Webmall_Single_Product_Search": "Specific Product Search",
    "Webmall_Best_Fit_Specific": "Specific Product Search",

    # Vague Product Search
    "Webmall_Best_Fit_Vague": "Vague Product Search",
    "Webmall_Find_Compatible_Products": "Vague Product Search",
    "Webmall_Substitutes": "Vague Product Search",

    # Cheapest Product Search
    "Webmall_Cheapest_Product_Search": "Cheapest Product Search",
    "Webmall_Cheapest_Best_Fit_Specific": "Cheapest Product Search",
    "Webmall_Cheapest_Best_Fit_Vague": "Cheapest Product Search",
}

TASK_SET_ORDER = [
    "Specific Product Search",
    "Vague Product Search",
    "Cheapest Product Search",
]


def latest_metrics_csv(model_dir: str):
    pattern = os.path.join(RESULTS_DIR, model_dir, "benchmark_metrics_*.csv")
    files = [f for f in glob.glob(pattern) if "_stream" not in f]
    return max(files, key=os.path.getmtime) if files else None


def load_rows(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["task_set"] = TASK_SET_MAP.get(r["category"], "Other")
    return rows


def cost_per_task(r: dict, model_dir: str) -> float:
    pin, pout = PRICING[model_dir]
    cost = (int(r["prompt_tokens"]) / 1_000_000 * pin
            + int(r["completion_tokens"]) / 1_000_000 * pout)
    fpin, fpout = FILTER_PRICING
    cost += (_ftok(r, "filter_prompt_tokens") / 1_000_000 * fpin
             + _ftok(r, "filter_completion_tokens") / 1_000_000 * fpout)
    cpin, cpout = CACHE_PRICING
    cost += (_ftok(r, "cache_prompt_tokens") / 1_000_000 * cpin
             + _ftok(r, "cache_completion_tokens") / 1_000_000 * cpout)
    return cost


def _ftok(r, key):
    return int(r.get(key, 0) or 0)


def agg(rows, model_dir):
    n = len(rows)
    cr = mean(float(r["task_completion_rate"]) for r in rows)
    f1 = mean(float(r["f1_score"]) for r in rows)
    avg_tokens = mean(
        int(r["prompt_tokens"]) + int(r["completion_tokens"])
        + _ftok(r, "filter_prompt_tokens") + _ftok(r, "filter_completion_tokens")
        + _ftok(r, "cache_prompt_tokens") + _ftok(r, "cache_completion_tokens")
        for r in rows
    )
    avg_cost = mean(cost_per_task(r, model_dir) for r in rows)
    avg_runtime = mean(float(r["execution_duration"]) for r in rows)
    return {
        "n": n,
        "CR": round(cr, 2),
        "F1": round(f1, 2),
        "Token": f"{round(avg_tokens):,}",
        "Cost": f"${avg_cost:.2f}",
        "Runtime": f"{round(avg_runtime)} s",
    }


def write_table(name: str, rows: list[dict]):
    if not rows:
        return
    csv_path = os.path.join(OUT_DIR, f"{name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md_path = os.path.join(OUT_DIR, f"{name}.md")
    headers = list(rows[0].keys())
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[h]) for h in headers) + " |\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data = {}
    for m in MODELS:
        path = latest_metrics_csv(m)
        if not path:
            print(f"[skip] {m}")
            continue
        data[m] = load_rows(path)
        print(f"[ok]   {m}: {len(data[m])} tasks")

    # ---------- Table 3: Average performance per LLM (RAG only) ----------
    t3 = []
    for m, rows in data.items():
        a = agg(rows, m)
        t3.append({
            "Model": MODELS[m],
            "CR": a["CR"], "F1": a["F1"],
            "Token": a["Token"], "Cost": a["Cost"], "Runtime": a["Runtime"],
        })
    write_table("steiner_table3_avg_per_llm", t3)

    # ---------- Table 4: Average F1 by task set per LLM ----------
    t4 = []
    for ts in TASK_SET_ORDER:
        row = {"Task Set": ts}
        for m, rows in data.items():
            sub = [r for r in rows if r["task_set"] == ts]
            row[MODELS[m]] = round(mean(float(r["f1_score"]) for r in sub), 2) if sub else "-"
        t4.append(row)
    # Add overall row
    overall = {"Task Set": "Overall (subset)"}
    for m, rows in data.items():
        overall[MODELS[m]] = round(mean(float(r["f1_score"]) for r in rows), 2)
    t4.append(overall)
    write_table("steiner_table4_f1_by_taskset", t4)

    # ---------- Tables 5-7: Per task set, per LLM ----------
    for i, ts in enumerate(TASK_SET_ORDER, start=5):
        rows_out = []
        for m, rows in data.items():
            sub = [r for r in rows if r["task_set"] == ts]
            if not sub:
                continue
            a = agg(sub, m)
            rows_out.append({
                "Model": MODELS[m],
                "n": a["n"],
                "CR": a["CR"], "F1": a["F1"],
                "Token": a["Token"], "Cost": a["Cost"], "Runtime": a["Runtime"],
            })
        slug = ts.lower().replace(" ", "_")
        write_table(f"steiner_table{i}_{slug}", rows_out)

    # Print all
    print("\n=== Table 3: Average performance per LLM (RAG, 45-task subset) ===")
    for r in t3: print(r)
    print("\n=== Table 4: F1 by task set ===")
    for r in t4: print(r)
    for i, ts in enumerate(TASK_SET_ORDER, start=5):
        print(f"\n=== Table {i}: {ts} ===")
        slug = ts.lower().replace(" ", "_")
        path = os.path.join(OUT_DIR, f"steiner_table{i}_{slug}.csv")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                print(f.read().strip())

    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
