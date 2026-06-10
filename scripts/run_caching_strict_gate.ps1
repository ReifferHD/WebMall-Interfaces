$ErrorActionPreference = "Stop"

# v8: Re-run Caching A and B for both models with a FROZEN (read-only) cache
# during evaluation (CACHE_FREEZE=1). This removes intra-run learning: the
# evaluation run can only HIT templates that came from the disjoint warming
# pool, it never distills new templates from earlier evaluation tasks. This
# is the clean disjoint-warming measurement for RQ3/RQ4.
#
# Cold (warming) snapshots are reused from the v2 batch -- only the evaluation
# runs are repeated. Strict 6-field gate, natural-language failure notes for B.

$RepoRoot = "C:\SeminarCode\WebMall-Interfaces"
Set-Location $RepoRoot

$RunTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = Join-Path $RepoRoot "results\rag\run_logs"
$CacheDir = Join-Path $RepoRoot "cache"
$CacheFile = Join-Path $CacheDir "plan_cache.json"
$BenchScript = Join-Path $RepoRoot "src\benchmark_rag.py"
$SummaryLog = Join-Path $LogDir "v8_frozen_$RunTimestamp.summary.log"

$ColdGPT   = Join-Path $CacheDir "plan_cache_after_gpt5mini_v2_cold_20260531_203828.json"
$ColdFlash = Join-Path $CacheDir "plan_cache_after_gemini_flash_v2_cold_20260531_221616.json"
$ColdJsonlGPT   = "$RepoRoot\results\rag-caching\gpt-5-mini-medium\benchmark_results_20260531_203835.jsonl"
$ColdJsonlFlash = "$RepoRoot\results\rag-caching\gemini-2.5-flash\benchmark_results_20260531_221623.jsonl"

foreach ($p in @($ColdGPT, $ColdFlash, $ColdJsonlGPT, $ColdJsonlFlash)) {
    if (-not (Test-Path $p)) { throw "missing required input: $p" }
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$CacheBackup = Join-Path $CacheDir "plan_cache_backup_before_v8_$RunTimestamp.json"
if (Test-Path $CacheFile) { Copy-Item -LiteralPath $CacheFile -Destination $CacheBackup -Force }

function Write-Summary {
    param([string]$Message)
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" | Tee-Object -FilePath $SummaryLog -Append
}

function Reset-Env {
    param([string]$MainModel, [string]$Reasoning)
    Remove-Item env:BENCHMARK_JSON_PATH -ErrorAction SilentlyContinue
    Remove-Item env:MASKING_MODE -ErrorAction SilentlyContinue
    Remove-Item env:MAIN_REASONING_EFFORT -ErrorAction SilentlyContinue
    $env:PYTHONUNBUFFERED = "1"
    $env:MAIN_MODEL = $MainModel
    if ($Reasoning) { $env:MAIN_REASONING_EFFORT = $Reasoning }
    $env:FILTER_MODEL = "gpt-5-nano"
    $env:CACHE_MODEL = "gpt-4o-mini"
    $env:CACHE_HIT_MODEL = "gpt-4o-mini"
    $env:CACHE_MATCH_POLICY = "deterministic_gate"
    $env:CACHE_GATE_LEVEL = "strict"
    $env:CACHE_FREEZE = "1"     # read-only cache during evaluation
    $env:MASKING_WINDOW = "2"
    $env:OPTIMIZATION_METHOD = "caching"
}

function Invoke-Run {
    param([string]$Name)
    $OutLog = Join-Path $LogDir "$Name`_$RunTimestamp.out.log"
    $ErrLog = Join-Path $LogDir "$Name`_$RunTimestamp.err.log"
    Write-Summary "START $Name gate=$env:CACHE_GATE_LEVEL freeze=$env:CACHE_FREEZE model=$env:MAIN_MODEL"
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try {
        & python $BenchScript > $OutLog 2> $ErrLog
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $prev }
    Write-Summary "END $Name exit=$code"
    if ($code -ne 0) { throw "$Name failed: $ErrLog" }
}

function Run-Model {
    param([string]$Tag, [string]$Model, [string]$Reasoning, [string]$ColdSnapshot, [string]$ColdJsonl)

    # --- Variant A: warming templates only, frozen, no pitfalls ---
    Copy-Item -LiteralPath $ColdSnapshot -Destination $CacheFile -Force
    Write-Summary "[$Tag] A: restored cold snapshot (frozen, no pitfalls)"
    Reset-Env -MainModel $Model -Reasoning $Reasoning
    Invoke-Run -Name "v8_${Tag}_caching_warm_A_frozen"
    Copy-Item -LiteralPath $CacheFile -Destination (Join-Path $CacheDir "plan_cache_after_v8_${Tag}_warmA_$RunTimestamp.json") -Force

    # --- Variant B: warming templates + failure notes, frozen ---
    Copy-Item -LiteralPath $ColdSnapshot -Destination $CacheFile -Force
    Write-Summary "[$Tag] B: restored cold snapshot, injecting natural-language pitfalls (frozen)"
    & python scripts\inject_pitfalls.py $ColdJsonl --cache $CacheFile --limit 5
    Copy-Item -LiteralPath $CacheFile -Destination (Join-Path $CacheDir "plan_cache_after_v8_${Tag}_pitfalls_$RunTimestamp.json") -Force
    Reset-Env -MainModel $Model -Reasoning $Reasoning
    Invoke-Run -Name "v8_${Tag}_caching_warm_B_frozen"
    Copy-Item -LiteralPath $CacheFile -Destination (Join-Path $CacheDir "plan_cache_after_v8_${Tag}_warmB_$RunTimestamp.json") -Force
}

Write-Summary "v8 frozen-cache caching re-run started (A+B, both models)"
Run-Model -Tag "gpt5mini" -Model "gpt-5-mini" -Reasoning "medium" -ColdSnapshot $ColdGPT -ColdJsonl $ColdJsonlGPT
Run-Model -Tag "flash"    -Model "gemini-2.5-flash" -Reasoning "" -ColdSnapshot $ColdFlash -ColdJsonl $ColdJsonlFlash
Write-Summary "v8 frozen-cache caching re-run completed."
