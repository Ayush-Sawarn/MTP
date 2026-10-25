param(
    [string]$PythonExe = "python",
    [int]$Epochs = 100,
    [int]$Seed = 2025
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectDir

function Invoke-ProjectPython {
    param(
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host ""
    Write-Host "=== $Description ===" -ForegroundColor Cyan
    Write-Host "$PythonExe $($Arguments -join ' ')"
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Failed: $Description (exit code $LASTEXITCODE). Fix the error above, then rerun this script."
    }
}

if ($Epochs -lt 1) {
    throw "Epochs must be at least 1."
}

$Budgets = @(12, 16, 24, 48)
foreach ($Budget in $Budgets) {
    $DataDir = "data/matlab_epa7_${Budget}pilots"
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectDir $DataDir))) {
        throw "Dataset directory not found: $DataDir"
    }
}

Invoke-ProjectPython -Description "Check Python and training device" -Arguments @(
    "-c",
    "import torch; print('Python torch:', torch.__version__); print('Training device:', 'cuda' if torch.cuda.is_available() else 'cpu')"
)

foreach ($Budget in $Budgets) {
    $DataDir = "data/matlab_epa7_${Budget}pilots"
    Invoke-ProjectPython -Description "Train ConcreteAE selector for $Budget pilots" -Arguments @(
        "train_pilot_selector.py",
        "--pilots", "$Budget",
        "--data-dir", $DataDir,
        "--epochs", "$Epochs",
        "--seed", "$Seed"
    )
}

foreach ($Budget in $Budgets) {
    $DataDir = "data/matlab_epa7_${Budget}pilots"
    Invoke-ProjectPython -Description "Train FC AttRNet with uniform $Budget-pilot pattern" -Arguments @(
        "train_pilot_attnet.py",
        "--model", "fc",
        "--method", "uniform",
        "--pilots", "$Budget",
        "--data-dir", $DataDir,
        "--epochs", "$Epochs",
        "--seed", "$Seed"
    )
    Invoke-ProjectPython -Description "Train FC AttRNet with ConcreteAE $Budget-pilot pattern" -Arguments @(
        "train_pilot_attnet.py",
        "--model", "fc",
        "--method", "concrete",
        "--pilots", "$Budget",
        "--pilot-indices", "results/pilot_indices/concrete_${Budget}.npy",
        "--data-dir", $DataDir,
        "--epochs", "$Epochs",
        "--seed", "$Seed"
    )
}

foreach ($Budget in $Budgets) {
    foreach ($Method in @("uniform", "concrete")) {
        $Checkpoint = "runs/pilot_design/fc_${Method}_${Budget}/best.pt"
        $DataDir = "data/matlab_epa7_${Budget}pilots"
        Invoke-ProjectPython -Description "Evaluate FC $Method model for $Budget pilots on test data" -Arguments @(
            "evaluate_pilot_patterns.py",
            "--checkpoint", $Checkpoint,
            "--data-dir", $DataDir
        )
    }
}

Invoke-ProjectPython -Description "Generate FC comparison plots" -Arguments @(
    "plot_pilot_results.py",
    "--model", "AttRNet-FC"
)

Write-Host ""
Write-Host "All FC pilot experiments completed." -ForegroundColor Green
Write-Host "Metrics: results/metrics.csv"
Write-Host "Figures: results/figures/"
