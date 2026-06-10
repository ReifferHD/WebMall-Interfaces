$ErrorActionPreference = "Stop"

# v10: GPT-5-mini caching run, FAIR gate (shape_only) + FROZEN, variant B
# (failure notes injected). Counterpart to v9 (variant A). Because the fair
# gate fires on ~25/45 tasks, the FAILURE NOTES block is now actually shown to
# the cache-hit model on many tasks -- so the A-vs-B contrast (RQ4) becomes
# measurable here, unlike under the strict gate (0 hits, B == A trivially).

$RepoRoot = "C:\SeminarCode\WebMall-Interfaces"
Set-Location $RepoRoot

$RunTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = Join-Path $RepoRoot "results\rag\run_logs"
$CacheDir = Join-Path $RepoRoot "cache"
$CacheFile = Join-Path $CacheDir "plan_cache.json"
$BenchScript = Join-Path $RepoRoot "src\benchmark_rag.py"
$SummaryLog = Join-Path $LogDir "v10_fairgateB_$RunTimestamp.summary.log"
$ColdGPT = Join-Path $CacheDir "plan_cache_after_gpt5mini_v2_cold_20260531_203828.json"
$ColdJsonl = "$RepoRoot\results\rag-caching\gpt-5-mini-medium\benchmark_results_20260531_203835.jsonl"

if (-not (Test-Path $ColdGPT))   { throw "missing v2 cold snapshot: $ColdGPT" }
if (-not (Test-Path $ColdJsonl)) { throw "missing cold jsonl: $ColdJsonl" }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$CacheBackup = Join-Path $CacheDir "plan_cache_backup_before_v10_$RunTimestamp.json"
if (Test-Path $CacheFile) { Copy-Item -LiteralPath $CacheFile -Destination $CacheBackup -Force }

function Write-Summary { param([string]$m) "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" | Tee-Object -FilePath $SummaryLog -Append }

# Restore disjoint warming cache, then inject failure notes (variant B).
Copy-Item -LiteralPath $ColdGPT -Destination $CacheFile -Force
Write-Summary "restored v2 cold snapshot; injecting natural-language pitfalls (fair gate B)"
& python scripts\inject_pitfalls.py $ColdJsonl --cache $CacheFile --limit 5
Copy-Item -LiteralPath $CacheFile -Destination (Join-Path $CacheDir "plan_cache_after_v10_gpt5mini_fairgateB_pitfalls_$RunTimestamp.json") -Force

Remove-Item env:BENCHMARK_JSON_PATH -ErrorAction SilentlyContinue
Remove-Item env:MASKING_MODE -ErrorAction SilentlyContinue
$env:PYTHONUNBUFFERED = "1"
$env:MAIN_MODEL = "gpt-5-mini"
$env:MAIN_REASONING_EFFORT = "medium"
$env:FILTER_MODEL = "gpt-5-nano"
$env:CACHE_MODEL = "gpt-4o-mini"
$env:CACHE_HIT_MODEL = "gpt-4o-mini"
$env:CACHE_MATCH_POLICY = "deterministic_gate"
$env:CACHE_GATE_LEVEL = "shape_only"   # fair gate
$env:CACHE_FREEZE = "1"                 # read-only: only disjoint warming templates
$env:MASKING_WINDOW = "2"
$env:OPTIMIZATION_METHOD = "caching"

$OutLog = Join-Path $LogDir "v10_gpt5mini_caching_fairgateB_$RunTimestamp.out.log"
$ErrLog = Join-Path $LogDir "v10_gpt5mini_caching_fairgateB_$RunTimestamp.err.log"
Write-Summary "START v10 fairgate-B gate=$env:CACHE_GATE_LEVEL freeze=$env:CACHE_FREEZE model=$env:MAIN_MODEL"
$prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
try {
    & python $BenchScript > $OutLog 2> $ErrLog
    $code = $LASTEXITCODE
} finally { $ErrorActionPreference = $prev }
Write-Summary "END v10 fairgate-B exit=$code"
if ($code -ne 0) { throw "v10 fairgate-B failed: $ErrLog" }
Copy-Item -LiteralPath $CacheFile -Destination (Join-Path $CacheDir "plan_cache_after_v10_gpt5mini_fairgateB_$RunTimestamp.json") -Force
Write-Summary "v10 fairgate-B run completed."
