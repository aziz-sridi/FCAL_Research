param(
    [ValidateSet('resnet50','densenet121','efficientnet_b0','vit_tiny','vit_small','vit_base','all')]
    [string]$Architecture = 'resnet50',
    [ValidateSet('clean','full','none')]
    [string]$Evaluation = 'clean'
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot

$experiments = @(
    @{ Key='resnet50'; Config='configs/oxford_pet/oxford_pet_resnet50_fcal_matrix.yaml'; Slug='resnet50' },
    @{ Key='densenet121'; Config='configs/oxford_pet/oxford_pet_densenet121_fcal_matrix.yaml'; Slug='densenet121' },
    @{ Key='efficientnet_b0'; Config='configs/oxford_pet/oxford_pet_efficientnet_b0_fcal_matrix.yaml'; Slug='efficientnet-b0' },
    @{ Key='vit_tiny'; Config='configs/oxford_pet/oxford_pet_vit_tiny_fcal_matrix.yaml'; Slug='vit-tiny-p16-avg' },
    @{ Key='vit_small'; Config='configs/oxford_pet/oxford_pet_vit_small_fcal_matrix.yaml'; Slug='vit-small-p16-avg' },
    @{ Key='vit_base'; Config='configs/oxford_pet/oxford_pet_vit_base_fcal_matrix.yaml'; Slug='vit-base-p16-avg' }
)
if ($Architecture -ne 'all') {
    $experiments = @($experiments | Where-Object { $_.Key -eq $Architecture })
}

function Invoke-Training {
    param([string]$Config, [string]$Method, [string]$RunId, [string]$InitialResume = '')
    $runDir = Join-Path 'outputs' $RunId
    if (Test-Path -LiteralPath (Join-Path $runDir 'summary.json')) { return }
    $latest = Join-Path $runDir 'latest.pt'
    $arguments = @('-m', 'fcal_study.train', $Config, '--method', $Method, '--run-id', $RunId)
    if (Test-Path -LiteralPath $latest) {
        $arguments += @('--resume', $latest)
    } elseif ($InitialResume) {
        $arguments += @('--resume', $InitialResume)
    }
    & python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $RunId" }
}

foreach ($experiment in $experiments) {
    $slug = $experiment.Slug
    $config = $experiment.Config
    $ceRun = "oxford_pet/$slug-ce-seed42"
    $ceDir = Join-Path 'outputs' $ceRun
    Invoke-Training $config 'ce' $ceRun
    $branch = Join-Path $ceDir 'epoch-005-branch.pt'
    if (-not (Test-Path -LiteralPath $branch)) { throw "Missing shared epoch-5 branch: $branch" }

    $runs = @(
        @{ Method='fcal'; Suffix='fcal-v1' },
        @{ Method='fcal_v2'; Suffix='fcal-v2' },
        @{ Method='ufcal'; Suffix='ufcal-auto-safe' }
    )
    foreach ($item in $runs) {
        Invoke-Training $config $item.Method "oxford_pet/$slug-$($item.Suffix)-seed42" $branch
    }

    if ($Evaluation -ne 'none') {
        $allRuns = @($ceRun) + @($runs | ForEach-Object { "oxford_pet/$slug-$($_.Suffix)-seed42" })
        foreach ($runId in $allRuns) {
            $checkpoint = Join-Path (Join-Path 'outputs' $runId) 'best-validation.pt'
            if ($Evaluation -eq 'clean') {
                python -m fcal_study.evaluate $checkpoint --variants clean
            } else {
                python -m fcal_study.evaluate $checkpoint
            }
            if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $runId" }
        }
    }
}
