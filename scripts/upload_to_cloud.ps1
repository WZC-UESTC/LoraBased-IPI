# ============================================================
# 上传到云主机 (Windows PowerShell)
#
# 用法:
#   .\scripts\upload_to_cloud.ps1 -Host "12.34.56.78" -User "cjh"
#   .\scripts\upload_to_cloud.ps1 -Host "12.34.56.78" -User "cjh" -Port 22022
#   .\scripts\upload_to_cloud.ps1 -Host "12.34.56.78" -User "cjh" -DryRun
# ============================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$Host,

    [int]$Port = 22,

    [Parameter(Mandatory=$true)]
    [string]$User,

    [string]$RemotePath = "/disk1/Lora",

    [switch]$DryRun = $false
)

$ErrorActionPreference = "Stop"
$LocalRoot = (Resolve-Path "$PSScriptRoot\..").Path

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Upload LoRA-IPI to Cloud" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Local:  $LocalRoot"
Write-Host "  Remote: ${User}@${Host}:${Port} → ${RemotePath}"
Write-Host ""

$Exclude = @(
    "--exclude=.git/"
    "--exclude=__pycache__/"
    "--exclude=*.pyc"
    "--exclude=.pytest_cache/"
    "--exclude=*.egg-info/"
    "--exclude=.vscode/"
    "--exclude=.idea/"
    "--exclude=venv/"
    "--exclude=.venv/"
    "--exclude=data/output/"
    "--exclude=lora_output/"
    "--exclude=results/"
    "--exclude=*.bin"
    "--exclude=*.safetensors"
    "--exclude=*.pt"
    "--exclude=*.pth"
)

$SshOpts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

if ($DryRun) {
    Write-Host "=== DRY RUN ===" -ForegroundColor Yellow
    $DryArgs = @("-avz", "--dry-run", "-e", "ssh $SshOpts -p $Port") + $Exclude + @(".", "${User}@${Host}:${RemotePath}/")
    Push-Location $LocalRoot
    try { bash -c "rsync $($DryArgs -join ' ')" }
    finally { Pop-Location }
    Write-Host ""
    Write-Host "去掉 -DryRun 执行实际上传。" -ForegroundColor Yellow
    exit 0
}

Write-Host "Uploading..." -ForegroundColor Green
Push-Location $LocalRoot
try {
    $Args = @("-avz", "--progress", "-e", "ssh $SshOpts -p $Port") + $Exclude + @(".", "${User}@${Host}:${RemotePath}/")
    bash -c "rsync $($Args -join ' ')"
    Write-Host ""
    Write-Host "Upload done!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next on cloud:"
    Write-Host "  ssh -p $Port ${User}@${Host}"
    Write-Host "  cd ${RemotePath}"
    Write-Host "  bash scripts/cloud_setup.sh $User"
    Write-Host "  source ~/.bashrc && lora-init"
    Write-Host "  bash scripts/run_experiment.sh"
}
finally { Pop-Location }
