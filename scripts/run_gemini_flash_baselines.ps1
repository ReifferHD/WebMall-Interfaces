$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$RunTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = Join-Path $RepoRoot "results\rag\run_logs"
$SummaryLog = Join-Path $LogDir "gemini_flash_baselines_$RunTimestamp.summary.log"
$RunCount = 4
if ($env:GEMINI_BASELINE_RUNS) {
    $RunCount = [int]$env:GEMINI_BASELINE_RUNS
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Summary {
    param([string]$Message)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    $line | Tee-Object -FilePath $SummaryLog -Append
}

function Invoke-BaselineRun {
    param([int]$Index)

    $Name = "gemini_flash_baseline_extra$Index"
    $OutLog = Join-Path $LogDir "$Name`_$RunTimestamp.out.log"
    $ErrLog = Join-Path $LogDir "$Name`_$RunTimestamp.err.log"

    $env:PYTHONUNBUFFERED = "1"
    $env:MAIN_MODEL = "gemini-2.5-flash"
    $env:OPTIMIZATION_METHOD = "none"

    Write-Summary "START $Name main_model=$env:MAIN_MODEL method=none"
    Write-Summary "OUT $OutLog"
    Write-Summary "ERR $ErrLog"

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python scripts\run_rag_gemini_flash.py none > $OutLog 2> $ErrLog
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

Write-Summary "Gemini 2.5 Flash baseline batch started run_count=$RunCount"
for ($i = 1; $i -le $RunCount; $i++) {
    Invoke-BaselineRun -Index $i
}
Write-Summary "Gemini 2.5 Flash baseline batch completed"
