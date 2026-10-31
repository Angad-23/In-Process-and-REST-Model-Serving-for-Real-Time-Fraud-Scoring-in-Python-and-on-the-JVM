# resubmission_A_prepare_and_train.ps1
# STEP A of the IEEE Access resubmission (about 15-20 minutes; watch it, it stops at the first error).
#   1. installs the pinned Python packages (onnxruntime 1.30.0 = Java version, psutil)
#   2. re-prepares the data: temporal split, encoders from training months, warm-up rows from validation month
#   3. retrains scikit-learn with the TRAINING-ONLY class weight, exports ONNX, checks parity
#   4. sensitivity check: scikit-learn with the old weight 90 (metrics only; main model untouched)
#   5. builds the Java benchmark (ONNX Runtime 1.30.0) and retrains Smile with the training-only weight
#   6. sensitivity check: Smile with weight 90 (scores only; main model untouched)
#   7. evaluates both models: test metrics incl. the ACTUAL test false-positive rate
#
# Usage (normal PowerShell, from anywhere):
#   powershell -ExecutionPolicy Bypass -File <project>\scripts\resubmission_A_prepare_and_train.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py   = Join-Path $root ".venv\Scripts\python.exe"
$src  = Join-Path $root "python\src"
$jdir = Join-Path $root "java"
$log  = Join-Path $root ("results\resubmission_A_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

function Step([string]$title, [string]$dir, [string]$exe, [string[]]$argList) {
    Write-Host ""
    Write-Host "=== $title" -ForegroundColor Cyan
    Push-Location $dir
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"   # Windows PowerShell 5.1: warnings on stderr must not abort the script
    try {
        & $exe @argList 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "step failed: $title (exit code $LASTEXITCODE). See $log" }
    } finally { $ErrorActionPreference = $prev; Pop-Location }
}

if (-not (Test-Path $py)) { throw "Python virtual environment not found at $py" }
if (-not (Test-Path (Join-Path $root "data\raw\Base.csv"))) { throw "data\raw\Base.csv not found (download it from Kaggle first)" }
$mvn = Get-Command mvn -ErrorAction SilentlyContinue
if (-not $mvn) { throw "mvn not found on PATH" }

Step "1/8 install pinned Python packages" $root $py @("-m", "pip", "install", "-q", "-r", "python\requirements.txt")
Step "2/8 prepare data (temporal split, training-only choices, validation warm-up rows)" $src $py @("s02_prepare_data.py")
Step "3/8 train scikit-learn (training-only class weight) + ONNX export + parity" $src $py @("s03_train_sklearn.py")
Step "4/8 sensitivity: scikit-learn with the old weight 90" $src $py @("s03_train_sklearn.py", "--fraud-weight", "90", "--tag", "w90", "--skip-onnx")
Step "5/8 build Java benchmark (Maven)" $jdir $mvn.Source @("-q", "package")
Step "6/8 train Smile (training-only weight)" $jdir "java" @("-Xmx8g", "-XX:+UseG1GC", "-jar", "target\bench.jar", "train")
Step "7/8 sensitivity: Smile with the old weight 90" $jdir "java" @("-Xmx8g", "-XX:+UseG1GC", "-jar", "target\bench.jar", "train", "--fraudweight", "90", "--tag", "w90")
Step "8/8 evaluate Smile (main + sensitivity) with the same code as scikit-learn" $src $py @("s08_evaluate_scores.py", "--name", "smile")
Step "8/8 (cont.)" $src $py @("s08_evaluate_scores.py", "--name", "smile_w90")

Write-Host ""
Write-Host "STEP A finished. Log: $log" -ForegroundColor Green
Write-Host "Check the run order with:  $py scripts\run_campaign.py --campaign main --dry-run"
Write-Host "Then, before going to bed:  powershell -ExecutionPolicy Bypass -File scripts\resubmission_B_campaigns.ps1"
