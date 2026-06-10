# Results Manifest

Maps every run reported in the seminar paper *"Reducing the Resource Consumption
of LLM Agents"* (Luis Reifferscheid, 2026) to its result files in this repository.
All aggregates below were re-verified against the metrics CSVs on 2026-06-10.

Each run consists of `benchmark_metrics_<ts>.csv` (per-task metrics),
`benchmark_results_<ts>.json` / `.jsonl` (full answers + tool-call logs), and for
caching runs additionally `benchmark_metrics_<ts>_stream.csv`.
Run logs live in `results/rag/run_logs/` (note: `*.log` is gitignored).

## Table 2 — GPT-5-mini (45-task subset)

| Paper row | Result timestamp | Folder | Run log prefix |
|---|---|---|---|
| Baseline (5-run avg, F1 0.747) | `20260424_170816`, `20260424_201845`, `20260424_210557`, `20260424_214631`, `20260424_222156` | `results/rag/gpt-5-mini-medium/` | — |
| Filtering (F1 0.711) | `20260426_193413` | `results/rag-filtering/gpt-5-mini-medium/` | `filtering_20260426_193401` |
| Caching warm A (F1 0.596, 25/45 hits) | `20260608_223054` | `results/rag-caching/gpt-5-mini-medium/` | `v9_gpt5mini_caching_fairgate_20260608_223046` |
| Caching warm B (F1 0.612, 24/45 hits) | `20260608_230115` | `results/rag-caching/gpt-5-mini-medium/` | `v10_gpt5mini_caching_fairgateB_20260608_230101` |
| Masking placeholder (F1 0.742) | `20260429_171008` | `results/rag-masking/gpt-5-mini-medium/` | `masking_hard_rerun_bg_20260429_170948` |
| Masking empty (F1 0.731) | `20260531_214959` | `results/rag-masking/gpt-5-mini-medium/` | `gpt5mini_v2_masking_empty_20260531_203828` |

## Table 3 — Gemini 2.5 Flash (45-task subset)

| Paper row | Result timestamp | Folder | Run log prefix |
|---|---|---|---|
| Baseline (5-run avg, F1 0.701) | `20260430_140927`, `20260501_180002`, `20260501_182732`, `20260501_184602`, `20260501_190206` | `results/rag/gemini-2.5-flash/` | `gemini_flash_all_20260430_140915`, `gemini_flash_baselines_20260501_175938` |
| Filtering (F1 0.702) | `20260502_153335` | `results/rag-filtering/gemini-2.5-flash/` | `gemini_flash_filtering_methods_20260502_153251` |
| Caching warm A (F1 0.537, 28/45 hits) | `20260608_233508` | `results/rag-caching/gemini-2.5-flash/` | `v11_flash_caching_fairgate_A_20260608_233451` |
| Caching warm B (F1 0.499, 28/45 hits) | `20260608_235251` | `results/rag-caching/gemini-2.5-flash/` | `v11_flash_caching_fairgate_B_20260608_233451` |
| Masking placeholder (F1 0.692) | `20260502_163217` | `results/rag-masking/gemini-2.5-flash/` | `gemini_flash_masking_methods_hard_20260502_153251` |
| Masking empty (F1 0.704) | `20260531_225545` | `results/rag-masking/gemini-2.5-flash/` | `gemini_flash_v2_masking_empty_20260531_221616` |

Note: the paper's token columns are the metrics-CSV sums including helper-LLM
tokens (filter and cache models). As of 2026-06-10 every cell of the paper's
Tables 2-4 reproduces exactly from the metrics CSVs listed here (aggregate
values via `scripts/summarize_v2_results.py` plus the CSV token columns).
The error-analysis counts in Table 6 reproduce from the JSONL files via
`scripts/analyze_failures.py` (cache error classes) and the Table 5 signal
definitions applied to `filter_decisions` / `masking_events` (filter and
masking classes).

## Caching infrastructure (Sections 4.2 and 7)

| Artifact | File(s) |
|---|---|
| Cold run on warming pool, GPT-5-mini (23 tasks) | `results/rag-caching/gpt-5-mini-medium/` `20260531_203835` |
| Cold run on warming pool, Gemini Flash (23 tasks) | `results/rag-caching/gemini-2.5-flash/` `20260531_221623` |
| Cold-cache snapshot GPT-5-mini (used by shape-gate runs) | `cache/plan_cache_after_gpt5mini_v2_cold_20260531_203828.json` |
| Cold-cache snapshot Gemini Flash | `cache/plan_cache_after_gemini_flash_v2_cold_20260531_221616.json` |
| Strict six-field gate, "never fires" (0/45), GPT-5-mini A/B | `results/rag-caching/gpt-5-mini-medium/` `20260608_204558`, `20260608_211427` |
| Strict six-field gate, Gemini Flash | `results/rag-caching/gemini-2.5-flash/` `20260608_215845` |
| 45-task evaluation subset | `task_sets/task_sets_subset.json` |
| 23-task warming pool | `task_sets/task_sets_warming.json` |

## Scripts

| Script | Purpose / paper section |
|---|---|
| `scripts/run_gpt5_mini_methods_v2.ps1` | v2 batch: caching cold run + masking empty (GPT-5-mini) |
| `scripts/run_gemini_flash_methods_v2.ps1` | v2 batch: caching cold run + masking empty (Gemini) |
| `scripts/run_caching_shape_gate_gpt5mini_A.ps1` | Caching warm A, shape-only gate (formerly `run_v9_caching_fairgate.ps1`) |
| `scripts/run_caching_shape_gate_gpt5mini_B.ps1` | Caching warm B with failure notes (formerly `run_v10_caching_fairgate_B.ps1`) |
| `scripts/run_caching_shape_gate_gemini_AB.ps1` | Caching warm A+B on Gemini (formerly `run_v11_caching_fairgate_gemini.ps1`) |
| `scripts/run_caching_strict_gate.ps1` | Strict six-field gate runs, Sec. 7 (formerly `run_v8_caching_frozen.ps1`) |
| `scripts/run_gemini_flash_baselines.ps1` | Gemini baseline runs 2–5 |
| `scripts/run_all_gemini_flash.ps1` / `run_gemini_flash_methods.ps1` | Gemini baseline 1 / filtering + masking placeholder |
| `scripts/build_warming_set.py` | Builds `task_sets_warming.json` |
| `scripts/inject_pitfalls.py` | Attaches failure notes to cache entries (Sec. 4.2, variant B) |
| `scripts/update_subset.py` | Builds the 45-task `task_sets_subset.json` (Sec. 3.2) |
| `scripts/summarize_v2_results.py` | Aggregates result CSVs into the paper tables |
| `scripts/analyze_failures.py`, `scripts/analyze_filtering_failures.py` | Error analysis (Sec. 6) |
| `scripts/run_caching_gpt5_mini.py`, `scripts/run_rag_gemini_flash.py` | Python benchmark entry points used by the .ps1 drivers |
