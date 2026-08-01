# ============================================================
# 上传本地 Lora 项目到云主机 (Windows PowerShell)
#
# Usage:
#   .\scripts\upload_to_cloud.ps1 -Host "12.34.56.78" -Port 22022 -User "root"
#   .\scripts\upload_to_cloud.ps1 -Host "12.34.56.78" -Port 22022 -User "root" -RemotePath "/disk1/zhangsan/Lora"
#   .\scripts\upload_to_cloud.ps1 -Host "12.34.56.78" -DryRun   # 预览会上传哪些文件
# ============================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$Host,

    [int]$Port = 22,

    [Parameter(Mandatory=$true)]
    [string]$User,

    [string]$RemotePath = "",       # 留空则自动检测

    [string]$IdentityFile = "",     # SSH 私钥路径 (可选)

    [switch]$DryRun = $false,       # 只预览不执行

    [switch]$ExcludeLarge = $false  # 排除大文件 (模型/pycache等)
)

$ErrorActionPreference = "Stop"

# 本地项目根目录 (本脚本的上一级)
$LocalRoot = (Resolve-Path "$PSScriptRoot\..").Path

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " LoRA-IPI: Upload to Cloud Server" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Local:  $LocalRoot"
Write-Host "  Remote: ${User}@${Host}:${Port}"
Write-Host ""

# ---- 构建排除规则 ----
$ExcludeArgs = @(
    "--exclude=.git/"
    "--exclude=__pycache__/"
    "--exclude=*.pyc"
    "--exclude=.pytest_cache/"
    "--exclude=*.egg-info/"
    "--exclude=.vscode/"
    "--exclude=.idea/"
    "--exclude=data/output/"         # 生成的训练数据，云主机上重新生成
    "--exclude=venv/"
    "--exclude=.venv/"
    "--exclude=node_modules/"
)

if ($ExcludeLarge) {
    $ExcludeArgs += "--exclude=*.bin"
    $ExcludeArgs += "--exclude=*.safetensors"
    $ExcludeArgs += "--exclude=*.pt"
    $ExcludeArgs += "--exclude=*.pth"
    $ExcludeArgs += "--exclude=lora_output/"
}

# ---- SSH 选项 ----
$SshOptions = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
if ($IdentityFile) {
    $SshOptions += " -i `"$IdentityFile`""
}

# ---- 自动检测远程路径 ----
if (-not $RemotePath) {
    # 尝试 SSH 过去检测 disk1
    $SshCmd = "ssh ${SshOptions} -p $Port ${User}@${Host}"
    $DetectScript = @"
# Auto-detect project path on cloud
if [ -d /disk1 ]; then
    DISK=/disk1
elif [ -d /mnt/data ]; then
    DISK=/mnt/data
elif [ -d /root/autodl-tmp ]; then
    DISK=/root/autodl-tmp
else
    DISK=/home/$USER
fi
echo "${DISK}/${USER}/Lora"
"@

    try {
        $RemotePath = bash -c "$SshCmd '$DetectScript'" 2>$null
        if (-not $RemotePath) {
            # SSH 失败则用默认值
            $RemotePath = "/disk1/$User/Lora"
        }
    } catch {
        $RemotePath = "/disk1/$User/Lora"
    }

    Write-Host "  Auto-detected remote path: $RemotePath" -ForegroundColor Green
}

# ---- 构建 rsync 命令 ----
$RsyncArgs = @(
    "-avz"
    "--progress"
    "-e `"ssh ${SshOptions} -p $Port`""
) + $ExcludeArgs + @(
    "."
    "${User}@${Host}:${RemotePath}/"
)

$RsyncCmd = "rsync $($RsyncArgs -join ' ')"

# ---- 预览 ----
if ($DryRun) {
    Write-Host "============================================" -ForegroundColor Yellow
    Write-Host " DRY RUN — 预览即将上传的文件" -ForegroundColor Yellow
    Write-Host "============================================" -ForegroundColor Yellow
    Write-Host ""

    $DryRunArgs = $RsyncArgs + @("--dry-run")
    $DryRunCmd = "rsync $($DryRunArgs -join ' ')"

    Push-Location $LocalRoot
    try {
        bash -c $DryRunCmd
    } finally {
        Pop-Location
    }
    Write-Host ""
    Write-Host "确认无误后去掉 -DryRun 参数执行实际上传。" -ForegroundColor Yellow
    exit 0
}

# ---- 执行上传 ----
Write-Host "============================================" -ForegroundColor Green
Write-Host " Uploading..." -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green

Push-Location $LocalRoot
try {
    bash -c $RsyncCmd
    Write-Host ""
    Write-Host "Upload complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next on cloud:"
    Write-Host "  ssh -p $Port ${User}@${Host}"
    Write-Host "  cd $RemotePath"
    Write-Host "  bash scripts/cloud_setup.sh"
    Write-Host "  bash scripts/run_experiment.sh minimal"
} finally {
    Pop-Location
}
