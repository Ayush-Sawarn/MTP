param(
    [string]$PythonExe = "python",
    [int[]]$Budgets = @(12, 24, 48),
    [int]$Epochs = 100
)

# Repeated seeds were used for the 16-pilot reproducibility study. Use one
# fixed seed for the remaining pilot budgets to keep the comparison focused.
$Seeds = @(2025)

$metrics = "results/reproducibility_all_budgets.csv"

foreach ($budget in $Budgets) {
    $dataDir = "data/matlab_epa7_${budget}pilots"
    $uniformIndices = "results/pilot_indices/uniform_${budget}.npy"
    $selectorDir = "runs/pilot_design/fc_joint_selector_${budget}_uniform_init"
    $selectorLog = "$selectorDir/train.log"
    $jointIndices = "results/pilot_indices/joint_${budget}.npy"

    if (-not (Test-Path $uniformIndices)) {
        throw "Uniform pilot indices not found: $uniformIndices"
    }
    New-Item -ItemType Directory -Force $selectorDir | Out-Null
    Write-Host "Learning a $budget-pilot mask initialized from Uniform"
    & $PythonExe train_joint_pilot_attnet.py `
        --pilots $budget --data-dir $dataDir --initial-indices $uniformIndices `
        --epochs $Epochs --batch-size 128 --learning-rate 0.001 `
        --selector-learning-rate 0.0003 --seed 2025 --output $selectorDir 2>&1 |
        Tee-Object -FilePath $selectorLog
    if ($LASTEXITCODE -ne 0) { throw "Joint mask training failed for $budget pilots." }
    Copy-Item "$selectorDir/pilot_indices.npy" $jointIndices -Force

    foreach ($seed in $Seeds) {
        foreach ($method in @("uniform", "joint_frozen")) {
            $runName = "fc_${method}_${budget}_seed${seed}"
            $outputDir = "runs/pilot_design/$runName"
            $logPath = "$outputDir/train.log"
            New-Item -ItemType Directory -Force $outputDir | Out-Null
            $trainArgs = @(
                "train_pilot_attnet.py", "--model", "fc", "--method", $method,
                "--pilots", "$budget", "--data-dir", $dataDir, "--epochs", "$Epochs",
                "--batch-size", "128", "--learning-rate", "0.001", "--seed", "$seed",
                "--output", $outputDir
            )
            if ($method -eq "joint_frozen") {
                $trainArgs += @("--pilot-indices", $jointIndices)
            }
            Write-Host "Training fresh FC AttRNet: $runName"
            & $PythonExe @trainArgs 2>&1 | Tee-Object -FilePath $logPath
            if ($LASTEXITCODE -ne 0) { throw "Estimator training failed for $runName." }
            & $PythonExe evaluate_pilot_patterns.py --checkpoint "$outputDir/best.pt" `
                --data-dir $dataDir --split test --results-csv $metrics --no-extra-artifacts
            if ($LASTEXITCODE -ne 0) { throw "Test evaluation failed for $runName." }
        }
    }
}

& $PythonExe summarize_reproducibility.py --metrics $metrics `
    --output results/reproducibility_all_budgets_summary.csv
if ($LASTEXITCODE -ne 0) { throw "Could not summarize repeated-seed results." }
& $PythonExe plot_final_comparison.py --metrics $metrics
if ($LASTEXITCODE -ne 0) { throw "Could not create final comparison plots." }
Write-Host "Finished Uniform vs joint-frozen FC experiments for budgets: $($Budgets -join ', ')"
