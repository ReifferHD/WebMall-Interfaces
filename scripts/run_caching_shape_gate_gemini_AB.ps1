$ErrorActionPreference = "Stop"

# v11: Gemini-2.5-Flash caching runs, FAIR gate (shape_only) + FROZEN,
# variant A (templates only) then B (failure notes). Counterpart to the
# GPT-5-mini fair-gate runs (v9/v10), to fill the Gemini caching rows of the
# aggregate table under the same fair-gate, disjoint-warming, read-only regime.

$RepoRoot = "C:\SeminarCode\WebMall-Interfaces"
Set-Location $RepoRoot

$RunTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = Join-Path $RepoRoot "results\rag\run_logs"
$CacheDir = Join-Path $RepoRoot "cache"
$CacheFile = Join-Path $CacheDir "plan_cache.json"
$BenchScript = Join-Path $RepoRoot "src\benchmark_rag.py"
$SummaryLog = Join-Path $LogDir "v11_fairgate_gemini_$RunTimestamp.summary.log"
$ColdFlash = Join-Path $CacheDir "plan_cache_after_gemini_flash_v2_cold_20260531_221616.json"
$ColdJsonl = "$RepoRoot\results\rag-caching\gemini-2.5-flash\benchmark_results_20260531_221623.jsonl"

if (-not (Test-Path $ColdFlash)) { throw "missing flash cold snapshot: $ColdFlash" }
if (-not (Test-Path $ColdJsonl)) { throw "missing flash cold jsonl: $ColdJsonl" }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$CacheBackup = Join-Path $CacheDir "plan_cache_backup_before_v11_$RunTimestamp.json"
if (Test-Path $CacheFile) { Copy-Item -LiteralPath $CacheFile -Destination $CacheBackup -Force }

function Write-Summary { param([string]$m) "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" | Tee-Object -FilePath $SummaryLog -Append }

function Reset-Env {
    Remove-Item env:BENCHMARK_JSON_PATH -ErrorAction SilentlyContinue
    Remove-Item env:MASKING_MODE -ErrorAction SilentlyContinue
    Remove-Item env:MAIN_REASONING_EFFORT -ErrorAction SilentlyContinue
    $env:PYTHONUNBUFFERED = "1"
    $env:MAIN_MODEL = "gemini-2.5-flash"
    $env:FILTER_MODEL = "gpt-5-nano"
    $env:CACHE_MODEL = "gpt-4o-mini"
    $env:CACHE_HIT_MODEL = "gpt-4o-mini"
    $env:CACHE_MATCH_POLICY = "deterministic_gate"
    $env:CACHE_GATE_LEVEL = "shape_only"
    $env:CACHE_FREEZE = "1"
    $env:MASKING_WINDOW = "2"
    $env:OPTIMIZATION_METHOD = "caching"
}

function Invoke-Run {
    param([string]$Name)
    $OutLog = Join-Path $LogDir "$Name`_$RunTimestamp.out.log"
    $ErrLog = Join-Path $LogDir "$Name`_$RunTimestamp.err.log"
    Write-Summary "START $Name gate=$env:CACHE_GATE_LEVEL freeze=$env:CACHE_FREEZE model=$env:MAIN_MODEL"
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & python $BenchScript > $OutLog 2> $ErrLog; $code = $LASTEXITCODE } finally { $ErrorActionPreference = $prev }
    Write-Summary "END $Name exit=$code"
    if ($code -ne 0) { throw "$Name failed: $ErrLog" }
}

# --- Variant A: templates only, frozen ---
Copy-Item -LiteralPath $ColdFlash -Destination $CacheFile -Force
Write-Summary "A: restored flash cold snapshot (fair gate, frozen, no pitfalls)"
Reset-Env
Invoke-Run -Name "v11_flash_caching_fairgate_A"
Copy-Item -LiteralPath $CacheFile -Destination (Join-Path $CacheDir "plan_cache_after_v11_flash_fairgateA_$RunTimestamp.json") -Force

# --- Variant B: templates + failure notes, frozen ---
Copy-Item -LiteralPath $ColdFlash -Destination $CacheFile -Force
Write-Summary "B: restored flash cold snapshot, injecting pitfalls (fair gate, frozen)"
& python scripts\inject_pitfalls.py $ColdJsonl --cache $CacheFile --limit 5
Copy-Item -LiteralPath $CacheFile -Destination (Join-Path $CacheDir "plan_cache_after_v11_flash_fairgateB_pitfalls_$RunTimestamp.json") -Force
Reset-Env
Invoke-Run -Name "v11_flash_caching_fairgate_B"
Copy-Item -LiteralPath $CacheFile -Destination (Join-Path $CacheDir "plan_cache_after_v11_flash_fairgateB_$RunTimestamp.json") -Force
Write-Summary "v11 fair-gate Gemini runs completed."
