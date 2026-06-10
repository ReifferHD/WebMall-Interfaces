"""Print aggregate metrics for the runs reported in the seminar paper,
formatted to drop straight into the LaTeX results tables.

Note: avgTok counts main-agent tokens only (from the JSONL). The token
columns in the paper additionally include helper-LLM tokens (filter and
cache models); those sums come from the benchmark_metrics_*.csv files.
Cost here DOES include the helper models."""
import json
import glob
import os
import datetime
from statistics import mean

# GPT-5-mini list prices (per 1k tokens)
PRICES = {
    "gpt-5-mini":   {"in": 0.25 / 1e6, "out": 2.00 / 1e6},
    "gpt-5-nano":   {"in": 0.05 / 1e6, "out": 0.40 / 1e6},
    "gpt-4o-mini":  {"in": 0.15 / 1e6, "out": 0.60 / 1e6},
    "gemini-2.5-flash": {"in": 0.30 / 1e6, "out": 2.50 / 1e6},
    "embedding": 0.02 / 1e6,
}

# The runs reported in the seminar paper (see RESULTS.md for the mapping).
RUNS = [
    # --- Table 2: GPT-5-mini ---
    ("GPT5  Baseline run 1", "gpt-5-mini-medium", "20260424_170816", "gpt-5-mini"),
    ("GPT5  Baseline run 2", "gpt-5-mini-medium", "20260424_201845", "gpt-5-mini"),
    ("GPT5  Baseline run 3", "gpt-5-mini-medium", "20260424_210557", "gpt-5-mini"),
    ("GPT5  Baseline run 4", "gpt-5-mini-medium", "20260424_214631", "gpt-5-mini"),
    ("GPT5  Baseline run 5", "gpt-5-mini-medium", "20260424_222156", "gpt-5-mini"),
    ("GPT5  Filtering", "gpt-5-mini-medium", "20260426_193413", "gpt-5-mini"),
    ("GPT5  Caching warm A (shape gate)", "gpt-5-mini-medium", "20260608_223054", "gpt-5-mini"),
    ("GPT5  Caching warm B (failure notes)", "gpt-5-mini-medium", "20260608_230115", "gpt-5-mini"),
    ("GPT5  Masking placeholder", "gpt-5-mini-medium", "20260429_171008", "gpt-5-mini"),
    ("GPT5  Masking empty", "gpt-5-mini-medium", "20260531_214959", "gpt-5-mini"),
    # --- Table 3: Gemini 2.5 Flash ---
    ("FLASH Baseline run 1", "gemini-2.5-flash", "20260430_140927", "gemini-2.5-flash"),
    ("FLASH Baseline run 2", "gemini-2.5-flash", "20260501_180002", "gemini-2.5-flash"),
    ("FLASH Baseline run 3", "gemini-2.5-flash", "20260501_182732", "gemini-2.5-flash"),
    ("FLASH Baseline run 4", "gemini-2.5-flash", "20260501_184602", "gemini-2.5-flash"),
    ("FLASH Baseline run 5", "gemini-2.5-flash", "20260501_190206", "gemini-2.5-flash"),
    ("FLASH Filtering", "gemini-2.5-flash", "20260502_153335", "gemini-2.5-flash"),
    ("FLASH Caching warm A (shape gate)", "gemini-2.5-flash", "20260608_233508", "gemini-2.5-flash"),
    ("FLASH Caching warm B (failure notes)", "gemini-2.5-flash", "20260608_235251", "gemini-2.5-flash"),
    ("FLASH Masking placeholder", "gemini-2.5-flash", "20260502_163217", "gemini-2.5-flash"),
    ("FLASH Masking empty", "gemini-2.5-flash", "20260531_225545", "gemini-2.5-flash"),
    # --- Sec. 4.2 / 7: caching infrastructure runs ---
    ("GPT5  Caching cold (warming pool)", "gpt-5-mini-medium", "20260531_203835", "gpt-5-mini"),
    ("FLASH Caching cold (warming pool)", "gemini-2.5-flash", "20260531_221623", "gemini-2.5-flash"),
    ("GPT5  Caching strict gate A", "gpt-5-mini-medium", "20260608_204558", "gpt-5-mini"),
    ("GPT5  Caching strict gate B", "gpt-5-mini-medium", "20260608_211427", "gpt-5-mini"),
    ("FLASH Caching strict gate", "gemini-2.5-flash", "20260608_215845", "gemini-2.5-flash"),
]


def find_jsonl(model_dir: str, ts: str) -> str:
    for sub in ("rag-caching", "rag-masking", "rag-filtering", "rag"):
        p = f"results/{sub}/{model_dir}/benchmark_results_{ts}.jsonl"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no jsonl for {model_dir} @ {ts}")


def cost_for(prompt_tok, completion_tok, model_key):
    p = PRICES[model_key]
    return prompt_tok * p["in"] + completion_tok * p["out"]


def cost_helpers(t):
    """Cache + filter helper LLM cost share."""
    c = 0.0
    c += cost_for(t.get("filter_prompt_tokens", 0) or 0,
                  t.get("filter_completion_tokens", 0) or 0, "gpt-5-nano")
    c += cost_for(t.get("cache_prompt_tokens", 0) or 0,
                  t.get("cache_completion_tokens", 0) or 0, "gpt-4o-mini")
    return c


def per_cat_breakdown(tasks):
    out = {}
    for t in tasks:
        cat = t.get("task_category", "?")
        out.setdefault(cat, []).append(t)
    return out


def summarize(label, path, main_model):
    with open(path, encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    n = len(tasks)
    f1s = [(t.get("metrics", {}) or {}).get("f1_score", 0.0) for t in tasks]
    crs = [(t.get("metrics", {}) or {}).get("task_completion_rate", 0) for t in tasks]
    prompt = [t.get("prompt_tokens", 0) or 0 for t in tasks]
    completion = [t.get("completion_tokens", 0) or 0 for t in tasks]
    runtimes = [t.get("execution_time_seconds", 0) or 0 for t in tasks]
    avg_tokens = mean([p + c for p, c in zip(prompt, completion)])
    # Total run cost = main model + helpers
    total_cost = 0.0
    for t in tasks:
        ag_model = t.get("agent_model_used") or main_model
        # if cache_hit, the agent is the small cache-hit model; the
        # prompt/completion tokens are from THAT model
        if t.get("cache_hit"):
            ag_key = "gpt-4o-mini"
        else:
            ag_key = main_model
        total_cost += cost_for(t.get("prompt_tokens", 0) or 0,
                               t.get("completion_tokens", 0) or 0, ag_key)
        total_cost += cost_helpers(t)
    hit_rate = sum(1 for t in tasks if t.get("cache_hit")) / max(n, 1)

    print(f"\n=== {label} ({os.path.basename(path)}) ===")
    print(f"  N={n}   F1={mean(f1s):.3f}   CR={mean(crs):.3f}   "
          f"avgTok={avg_tokens:.0f}   cost=${total_cost:.3f}   "
          f"runtime={mean(runtimes):.1f}s   cacheHit={hit_rate:.0%}")
    # Per-category
    for cat, ts in per_cat_breakdown(tasks).items():
        f1c = mean([(x.get("metrics", {}) or {}).get("f1_score", 0.0) for x in ts])
        tokc = mean([(x.get("prompt_tokens", 0) or 0) + (x.get("completion_tokens", 0) or 0) for x in ts])
        print(f"    [{cat:24}] n={len(ts):2d}  F1={f1c:.3f}  avgTok={tokc:.0f}")


print("=" * 80)
print("v2 BATCH RESULTS")
print("=" * 80)

for label, model_dir, ts, main_model in RUNS:
    try:
        path = find_jsonl(model_dir, ts)
    except FileNotFoundError as e:
        print(f"\n!! {label}: {e}")
        continue
    summarize(label, path, main_model)
