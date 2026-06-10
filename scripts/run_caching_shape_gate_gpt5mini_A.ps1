$ErrorActionPreference = "Stop"

# v9: One GPT-5-mini caching run with the "fair" shape_only gate
# (kw + url + cheap only -- drops the family/number content tags that the
# plan template already abstracts via <PRODUCT>/<SPEC> placeholders). Cache is
# FROZEN (read-only) and warmed only from the disjoint v2 cold snapshot, so
# this isolates one question: under a strategy-level gate, do disjoint warming
# templates fire on the eval set, and at what F1? Contrast vs. the strict-gate
# frozen run (0 hits) and the in-sample shape_only runs (many hits, F1 collapse).
# Variant A only (templates, no failure notes).

$RepoRoot = "C:\SeminarCode\WebMall-Interfaces"
Set-Location $RepoRoot

$RunTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = Join-Path $RepoRoot "results\rag\run_logs"
$CacheDir = Join-Path $RepoRoot "cache"
$CacheFile = Join-Path $CacheDir "plan_cache.json"
$BenchScript = Join-Path $RepoRoot "src\benchmark_rag.py"
$SummaryLog = Join-Path $LogDir "v9_fairgate_$RunTimestamp.summary.log"
$ColdGPT = Join-Path $CacheDir "plan_cache_after_gpt5mini_v2_cold_20260531_203828.json"

if (-not (Test-Path $ColdGPT)) { throw "missing v2 cold snapshot: $ColdGPT" }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$CacheBackup = Join-Path $CacheDir "plan_cache_backup_before_v9_$RunTimestamp.json"
if (Test-Path $CacheFile) { Copy-Item -LiteralPath $CacheFile -Destination $CacheBackup -Force }

function Write-Summary { param([string]$m) "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" | Tee-Object -FilePath $SummaryLog -Append }

# Restore the disjoint warming cache; query it with the fair gate, read-only.
Copy-Item -LiteralPath $ColdGPT -Destination $CacheFile -Force
Write-Summary "restored v2 cold snapshot (fair gate=shape_only, frozen, no pitfalls)"

Remove-Item env:BENCHMARK_JSON_PATH -ErrorAction SilentlyContinue
Remove-Item env:MASKING_MODE -ErrorAction SilentlyContinue
$env:PYTHONUNBUFFERED = "1"
$env:MAIN_MODEL = "gpt-5-mini"
$env:MAIN_REASONING_EFFORT = "medium"
$env:FILTER_MODEL = "gpt-5-nano"
$env:CACHE_MODEL = "gpt-4o-mini"
$env:CACHE_HIT_MODEL = "gpt-4o-mini"
$env:CACHE_MATCH_POLICY = "deterministic_gate"
$env:CACHE_GATE_LEVEL = "shape_only"   # the "fair" strategy-level gate
$env:CACHE_FREEZE = "1"                 # read-only: only disjoint warming templates
$env:MASKING_WINDOW = "2"
$env:OPTIMIZATION_METHOD = "caching"

$OutLog = Join-Path $LogDir "v9_gpt5mini_caching_fairgate_$RunTimestamp.out.log"
$ErrLog = Join-Path $LogDir "v9_gpt5mini_caching_fairgate_$RunTimestamp.err.log"
Write-Summary "START v9 fairgate gate=$env:CACHE_GATE_LEVEL freeze=$env:CACHE_FREEZE model=$env:MAIN_MODEL"
$prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
try {
    & python $BenchScript > $OutLog 2> $ErrLog
    $code = $LASTEXITCODE
} finally { $ErrorActionPreference = $prev }
Write-Summary "END v9 fairgate exit=$code"
if ($code -ne 0) { throw "v9 fairgate failed: $ErrLog" }
Copy-Item -LiteralPath $CacheFile -Destination (Join-Path $CacheDir "plan_cache_after_v9_gpt5mini_fairgate_$RunTimestamp.json") -Force
Write-Summary "v9 fairgate run completed."
