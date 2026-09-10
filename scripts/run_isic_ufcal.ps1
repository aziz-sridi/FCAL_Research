$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot

$config = 'configs/isic/isic_resnet50_ufcal_auto_safe.yaml'
$baselineConfig = 'configs/isic/isic_resnet50_fcal_matrix.yaml'
$baseline = 'outputs/isic/resnet50-ce-seed42'
$branch = Join-Path $baseline 'epoch-005-branch.pt'
$runId = 'isic/resnet50-ufcal-auto-safe-seed42'

if (-not (Test-Path -LiteralPath $branch)) {
    python -m fcal_study.train $baselineConfig
    if ($LASTEXITCODE -ne 0) { throw 'ISIC baseline training failed' }
}
if (-not (Test-Path -LiteralPath $branch)) { throw "Missing shared epoch-5 branch: $branch" }

python -m fcal_study.train $config --run-id $runId --resume $branch
if ($LASTEXITCODE -ne 0) { throw 'Universal FCAL training failed' }

python -m fcal_study.evaluate "outputs/$runId/best-validation.pt"
if ($LASTEXITCODE -ne 0) { throw 'Universal FCAL evaluation failed' }
