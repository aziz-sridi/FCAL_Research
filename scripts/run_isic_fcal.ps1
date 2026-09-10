param(
    [ValidateSet('resnet50','densenet121','efficientnet_b0','vit_tiny','vit_small','vit_base','all')]
    [string]$Architecture = 'all',
    [switch]$SkipEvaluation
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot

$experiments = @(
    @{ Key='resnet50'; Config='configs/isic/isic_resnet50_fcal_matrix.yaml'; Slug='resnet50' },
    @{ Key='densenet121'; Config='configs/isic/isic_densenet121_fcal_matrix.yaml'; Slug='densenet121' },
    @{ Key='efficientnet_b0'; Config='configs/isic/isic_efficientnet_b0_fcal_matrix.yaml'; Slug='efficientnet-b0' },
    @{ Key='vit_tiny'; Config='configs/isic/isic_vit_tiny_avg_fcal_matrix.yaml'; Slug='vit-tiny-p16-avg' },
    @{ Key='vit_small'; Config='configs/isic/isic_vit_small_avg_fcal_matrix.yaml'; Slug='vit-small-p16-avg' },
    @{ Key='vit_base'; Config='configs/isic/isic_vit_base_avg_fcal_matrix.yaml'; Slug='vit-base-p16-avg' }
)
if ($Architecture -ne 'all') {
    $experiments = @($experiments | Where-Object { $_.Key -eq $Architecture })
}

foreach ($experiment in $experiments) {
    $slug = $experiment.Slug
    $config = $experiment.Config
    $ceRun = "isic/$slug-ce-seed42"
    $v1Run = "isic/$slug-fcal-v1-seed42"
    $v2Run = "isic/$slug-fcal-v2-seed42"
    $ceDir = Join-Path 'outputs' $ceRun
    $v1Dir = Join-Path 'outputs' $v1Run
    $v2Dir = Join-Path 'outputs' $v2Run

    if (-not (Test-Path -LiteralPath (Join-Path $ceDir 'summary.json'))) {
        python -m fcal_study.train $config --run-id $ceRun
        if ($LASTEXITCODE -ne 0) { throw "Baseline training failed for $slug" }
    }
    $branch = Join-Path $ceDir 'epoch-005-branch.pt'
    if (-not (Test-Path -LiteralPath $branch)) { throw "Missing shared epoch-5 branch: $branch" }

    if (-not (Test-Path -LiteralPath (Join-Path $v1Dir 'summary.json'))) {
        python -m fcal_study.train $config --method fcal --run-id $v1Run --resume $branch
        if ($LASTEXITCODE -ne 0) { throw "FCAL V1 training failed for $slug" }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $v2Dir 'summary.json'))) {
        python -m fcal_study.train $config --method fcal_v2 --run-id $v2Run --resume $branch
        if ($LASTEXITCODE -ne 0) { throw "FCAL V2 training failed for $slug" }
    }

    if (-not $SkipEvaluation) {
        foreach ($runDir in @($ceDir, $v1Dir, $v2Dir)) {
            if (-not (Test-Path -LiteralPath (Join-Path $runDir 'evaluation/summary.json'))) {
                python -m fcal_study.evaluate (Join-Path $runDir 'best-validation.pt')
                if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $runDir" }
            }
        }
        $reportDir = Join-Path 'reports/isic_v1_v2' $slug
        python -m fcal_study.compare (Join-Path $v1Dir 'evaluation') (Join-Path $v2Dir 'evaluation') (Join-Path $reportDir 'v1_vs_v2')
        if ($LASTEXITCODE -ne 0) { throw "V1/V2 comparison failed for $slug" }
        python -m fcal_study.compare (Join-Path $ceDir 'evaluation') (Join-Path $v1Dir 'evaluation') (Join-Path $reportDir 'v1_vs_ce')
        python -m fcal_study.compare (Join-Path $ceDir 'evaluation') (Join-Path $v2Dir 'evaluation') (Join-Path $reportDir 'v2_vs_ce')
    }
}

if (-not $SkipEvaluation) {
    python -m fcal_study.isic_report
    if ($LASTEXITCODE -ne 0) { throw 'ISIC report generation failed' }
}
