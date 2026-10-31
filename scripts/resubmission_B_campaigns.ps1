# resubmission_B_campaigns.ps1
# STEP B of the IEEE Access resubmission: all benchmark campaigns, unattended (about 3-4 hours; run overnight).
#   main            10 complete runs of all 11 arms, randomized order per run      (the paper's main results)
#   sens_warmup20k   3 runs of A1, A3, A4, A5 with 20,000 warm-up calls            (warm-up sensitivity)
#   sens_nopin       3 runs of all 11 arms WITHOUT CPU pinning, normal priority    (pinning sensitivity)
# then the run-level statistics for each campaign.
#
# BEFORE starting: charger plugged in, Power mode = Best performance, browser/IDE/Teams closed,
# OneDrive paused, PowerThrottlingOff = 1 (already set), sleep disabled while plugged in.
# Do NOT use the laptop while it runs. Nothing to click: no windows open.
#
# Usage:  powershell -ExecutionPolicy Bypass -File <project>\scripts\resubmission_B_campaigns.ps1
# Resume after an interruption (example: main stopped after run 6):
#   .venv\Scripts\python.exe scripts\run_campaign.py --campaign main --runs 4 --start-run 7

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py   = Join-Path $root ".venv\Scripts\python.exe"
$runner = Join-Path $root "scripts\run_campaign.py"
$stats  = Join-Path $root "python\src\s09_campaign_stats.py"
$log  = Join-Path $root ("results\resubmission_B_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

function Run([string]$title, [string[]]$argList) {
    Write-Host ""
    Write-Host "=== $title   (started $(Get-Date -Format 'HH:mm'))" -ForegroundColor Cyan
    Push-Location $root
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"   # Windows PowerShell 5.1: warnings on stderr must not abort the script
    try {
        & $py @argList 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Out-Host
        if ($LASTEXITCODE -ne 0) { Write-Host "WARNING: '$title' ended with exit code $LASTEXITCODE (see $log)" -ForegroundColor Yellow }
    } finally { $ErrorActionPreference = $prev; Pop-Location }
}

if (-not (Test-Path $py)) { throw "Python virtual environment not found at $py" }

# run_campaign.py keeps Windows awake while it runs (SetThreadExecutionState); nothing to restore afterwards.
Run "campaign main (10 runs, all arms)" @($runner, "--campaign", "main", "--runs", "10")
Run "campaign sens_warmup20k (3 runs, A1 A3 A4 A5, 20000 warm-up calls)" @($runner, "--campaign", "sens_warmup20k", "--runs", "3", "--arms", "A1,A3,A4,A5", "--warmup", "20000")
Run "campaign sens_nopin (3 runs, all arms, no pinning)" @($runner, "--campaign", "sens_nopin", "--runs", "3", "--no-pin")

Run "statistics: main" @($stats, "--campaign", "main")
Run "statistics: sens_warmup20k vs main" @($stats, "--campaign", "sens_warmup20k", "--baseline", "main")
Run "statistics: sens_nopin vs main" @($stats, "--campaign", "sens_nopin", "--baseline", "main")

Write-Host ""
Write-Host "STEP B finished $(Get-Date -Format 'HH:mm'). Log: $log" -ForegroundColor Green
Write-Host "Results: results\campaigns\main\analysis, results\campaigns\sens_*\analysis"
