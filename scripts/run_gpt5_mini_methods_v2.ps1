$ErrorActionPreference = "Stop"

# Driver for the v2 method comparison on GPT-5-mini.
# Only runs the methods that are NEW or changed vs. the existing results:
#   caching cold  : warming-set cold run (fills the cache)
#   caching A     : warm run on subset, NO pitfalls
#   caching B     : warm run on subset, WITH pitfalls
#   masking empty : no placeholder, content = ""
#
# Baseline, filtering, and masking-hard re-use the existing results
# (pipeline unchanged for those methods).

$RepoRoot = "C:\SeminarCode\WebMall-Interfaces"
Set-Location $RepoRoot

$RunTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = Join-Path $RepoRoot "results\rag\run_logs"
$CacheDir = Join-Path $RepoRoot "cache"
$CacheFile = Join-Path $CacheDir "plan_cache.json"
$CacheBackup = Join-Path $CacheDir "plan_cache_backup_before_gpt5mini_v2_$RunTimestamp.json"
$SummaryLog = Join-Path $LogDir "gpt5mini_v2_$RunTimestamp.summary.log"
$BenchScript = Join-Path $RepoRoot "src\benchmark_rag.py"
$WarmingTaskFile = "task_sets/task_sets_warming.json"
$ResultsDir = Join-Path $RepoRoot "results\rag-caching\gpt-5-mini-medium"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
if (Test-Path -LiteralPath $CacheFile) {
    Copy-Item -LiteralPath $CacheFile -Destination $CacheBackup -Force
}

function Write-Summary {
    param([string]$Message)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    $line | Tee-Object -FilePath $SummaryLog -Append
}

function Reset-Env {
    Remove-Item env:BENCHMARK_JSON_PATH -ErrorAction SilentlyContinue
    Remove-Item env:OPTIMIZATION_METHOD -ErrorAction SilentlyContinue
    Remove-Item env:MASKING_MODE -ErrorAction SilentlyContinue
    $env:PYTHONUNBUFFERED = "1"
    $env:MAIN_MODEL = "gpt-5-mini"
    $env:MAIN_REASONING_EFFORT = "medium"
    $env:FILTER_MODEL = "gpt-5-nano"
    $env:CACHE_MODEL = "gpt-4o-mini"
    $env:CACHE_HIT_MODEL = "gpt-4o-mini"
    $env:CACHE_MATCH_POLICY = "deterministic_gate"
    $env:MASKING_WINDOW = "2"
}

function Invoke-Run {
    param([string]$Name)
    $OutLog = Join-Path $LogDir "$Name`_$RunTimestamp.out.log"
    $ErrLog = Join-Path $LogDir "$Name`_$RunTimestamp.err.log"

    Write-Summary "START $Name optim=$env:OPTIMIZATION_METHOD masking_mode=$env:MASKING_MODE tasks=$env:BENCHMARK_JSON_PATH"
    Write-Summary "OUT $OutLog"

    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python $BenchScript > $OutLog 2> $ErrLog
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $prev
    }
    Write-Summary "END $Name exit_code=$code"
    if ($code -ne 0) { throw "$Name failed with exit code $code. See $ErrLog" }
}

Write-Summary "GPT-5-mini v2 batch started (caching A/B + masking empty)"

# --- CACHING COLD on disjoint warming set ---
Reset-Env
'{"_meta": {"schema_version": 2, "entry_count": 0}, "entries": {}}' |
    Set-Content -LiteralPath $CacheFile -Encoding UTF8
Write-Summary "Cache cleared for warming cold run."
$env:OPTIMIZATION_METHOD = "caching"
$env:BENCHMARK_JSON_PATH = $WarmingTaskFile
Invoke-Run -Name "gpt5mini_v2_caching_cold_warming"

$ColdCacheSnapshot = Join-Path $CacheDir "plan_cache_after_gpt5mini_v2_cold_$RunTimestamp.json"
Copy-Item -LiteralPath $CacheFile -Destination $ColdCacheSnapshot -Force
Write-Summary "Cold-cache snapshot saved: $ColdCacheSnapshot"

# Locate the JSONL produced by the warming cold run (newest under results/rag-caching).
$ColdJsonl = Get-ChildItem -Path $ResultsDir -Filter "benchmark_results_*.jsonl" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($null -eq $ColdJsonl) { throw "Could not find any cold-run JSONL under $ResultsDir" }
Write-Summary "Cold-run JSONL detected: $($ColdJsonl.FullName)"

# --- CACHING A: warm run on subset, NO pitfalls ---
Reset-Env
$env:OPTIMIZATION_METHOD = "caching"
Invoke-Run -Name "gpt5mini_v2_caching_warm_A"
Copy-Item -LiteralPath $CacheFile -Destination (Join-Path $CacheDir "plan_cache_after_gpt5mini_v2_warmA_$RunTimestamp.json") -Force

# --- Inject pitfalls and re-run = CACHING B ---
# Restore the cold snapshot so usage_count and template state start clean for B.
Copy-Item -LiteralPath $ColdCacheSnapshot -Destination $CacheFile -Force
Write-Summary "Restored cold cache for B."
& python scripts\inject_pitfalls.py $ColdJsonl.FullName --cache $CacheFile --limit 5
Copy-Item -LiteralPath $CacheFile -Destination (Join-Path $CacheDir "plan_cache_after_gpt5mini_v2_pitfalls_$RunTimestamp.json") -Force

Reset-Env
$env:OPTIMIZATION_METHOD = "caching"
Invoke-Run -Name "gpt5mini_v2_caching_warm_B"

# --- MASKING empty (no placeholder) ---
Reset-Env
$env:OPTIMIZATION_METHOD = "masking"
$env:MASKING_MODE = "empty"
Invoke-Run -Name "gpt5mini_v2_masking_empty"

Write-Summary "GPT-5-mini v2 batch completed."
