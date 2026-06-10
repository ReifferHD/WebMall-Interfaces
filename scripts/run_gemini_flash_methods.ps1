$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$RunTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = Join-Path $RepoRoot "results\rag\run_logs"
$CacheDir = Join-Path $RepoRoot "cache"
$CacheFile = Join-Path $CacheDir "plan_cache.json"
$CacheBackup = Join-Path $CacheDir "plan_cache_backup_before_gemini_flash_methods_$RunTimestamp.json"
$SummaryLog = Join-Path $LogDir "gemini_flash_methods_$RunTimestamp.summary.log"

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

function Set-RunEnvironment {
    param([string]$Method)

    $env:PYTHONUNBUFFERED = "1"
    $env:MAIN_MODEL = "gemini-2.5-flash"
    $env:OPTIMIZATION_METHOD = $Method
    $env:FILTER_MODEL = "gpt-5-nano"
    $env:CACHE_MODEL = "gpt-4o-mini"
    $env:CACHE_HIT_MODEL = "gpt-4o-mini"
    $env:CACHE_MATCH_POLICY = "deterministic_gate"
    $env:MASKING_MODE = "hard"
    $env:MASKING_WINDOW = "2"
}

function Invoke-BenchmarkRun {
    param(
        [string]$Name,
        [string]$Method
    )

    $OutLog = Join-Path $LogDir "$Name`_$RunTimestamp.out.log"
    $ErrLog = Join-Path $LogDir "$Name`_$RunTimestamp.err.log"

    Set-RunEnvironment -Method $Method
    Write-Summary "START $Name method=$Method main_model=$env:MAIN_MODEL filter_model=$env:FILTER_MODEL cache_model=$env:CACHE_MODEL cache_hit_model=$env:CACHE_HIT_MODEL"
    Write-Summary "OUT $OutLog"
    Write-Summary "ERR $ErrLog"

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python scripts\run_rag_gemini_flash.py $Method > $OutLog 2> $ErrLog
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    Write-Summary "END $Name exit_code=$ExitCode"
    if ($ExitCode -ne 0) {
        throw "$Name failed with exit code $ExitCode. See $ErrLog"
    }
}

Write-Summary "Gemini 2.5 Flash methods batch started"
Write-Summary "Existing cache backup: $CacheBackup"

Invoke-BenchmarkRun -Name "gemini_flash_filtering_methods" -Method "filtering"

"{}" | Set-Content -LiteralPath $CacheFile -Encoding UTF8
Write-Summary "Cache cleared for cold caching run: $CacheFile"
Invoke-BenchmarkRun -Name "gemini_flash_caching_methods_cold" -Method "caching"
Copy-Item -LiteralPath $CacheFile -Destination (Join-Path $CacheDir "plan_cache_after_gemini_flash_methods_cold_$RunTimestamp.json") -Force
Invoke-BenchmarkRun -Name "gemini_flash_caching_methods_warm" -Method "caching"

Invoke-BenchmarkRun -Name "gemini_flash_masking_methods_hard" -Method "masking"

Write-Summary "Gemini 2.5 Flash methods batch completed"
