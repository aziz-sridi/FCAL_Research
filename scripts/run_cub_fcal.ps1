param(
    [ValidateSet('resnet50','densenet121','efficientnet_b0','vit_tiny','vit_small','vit_base','all')]
    [string]$Architecture = 'resnet50',
    [switch]$SkipEvaluation
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot

$experiments = @(
    @{ Key='resnet50'; Base='resnet50'; Slug='resnet50' },
    @{ Key='densenet121'; Base='densenet121'; Slug='densenet121' },
    @{ Key='efficientnet_b0'; Base='efficientnet_b0'; Slug='efficientnet-b0' },
    @{ Key='vit_tiny'; Base='vit_tiny_avg'; Slug='vit-tiny-p16-avg' },
    @{ Key='vit_small'; Base='vit_small_avg'; Slug='vit-small-p16-avg' },
    @{ Key='vit_base'; Base='vit_base_avg'; Slug='vit-base-p16-avg' }
)
if ($Architecture -ne 'all') {
    $experiments = @($experiments | Where-Object { $_.Key -eq $Architecture })
}

foreach ($experiment in $experiments) {
    $base = $experiment.Base
    $slug = $experiment.Slug
    $ceRun = "$slug-ce-seed42"
    $branch = "outputs/$ceRun/epoch-005-branch.pt"

    if (-not (Test-Path -LiteralPath "outputs/$ceRun/summary.json")) {
        python -m fcal_study.train "configs/cub/$($base)_ce.yaml"
        if ($LASTEXITCODE -ne 0) { throw "Baseline training failed for $slug" }
    }
    if (-not (Test-Path -LiteralPath $branch)) { throw "Missing shared epoch-5 branch: $branch" }

    $runs = @(
        @{ Config="configs/cub/$($base)_fcal.yaml"; Run="$slug-fcal-seed42" },
        @{ Config="configs/cub/$($base)_fcal_v2.yaml"; Run="$slug-fcal-v2-seed42" }
    )
    foreach ($run in $runs) {
        if (-not (Test-Path -LiteralPath "outputs/$($run.Run)/summary.json")) {
            python -m fcal_study.train $run.Config --resume $branch
            if ($LASTEXITCODE -ne 0) { throw "Training failed: $($run.Run)" }
        }
    }

    if (-not $SkipEvaluation) {
        foreach ($runId in @($ceRun) + @($runs | ForEach-Object Run)) {
            python -m fcal_study.evaluate "outputs/$runId/best-validation.pt"
            if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $runId" }
        }
    }
}
