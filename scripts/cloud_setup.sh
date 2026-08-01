#!/bin/bash
# ============================================================
# LoRA-IPI Cloud Setup Script (Conda版)
# 所有环境/代码/缓存都放在 /disk1/<用户名>/ 下，不污染 /home
# ============================================================
set -e

# ---- 配置：改这里！----
DISK="/disk1"
USER_NAME="${1:-$(whoami)}"           # 用户名，作为文件夹名
CONDA_ROOT="${DISK}/${USER_NAME}/miniconda3"
PROJECT_ROOT="${DISK}/${USER_NAME}/Lora"
ENV_NAME="lora-ipi"                   # conda 环境名
# ----------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; }

echo "============================================"
echo " LoRA-IPI Cloud Setup (Conda on ${DISK})"
echo "============================================"
echo ""
echo "  Disk:        ${DISK}"
echo "  User folder: ${DISK}/${USER_NAME}"
echo "  Conda:       ${CONDA_ROOT}"
echo "  Project:     ${PROJECT_ROOT}"
echo "  Env name:    ${ENV_NAME}"
echo ""

# ---------- 1. 确认 disk1 存在 ----------
if [ ! -d "$DISK" ]; then
    err "Disk ${DISK} not found!"
    echo "Available mount points:"
    df -h | grep -v tmpfs | grep -v devtmpfs
    echo ""
    echo "Usage: $0 [username] [disk_path]"
    echo "Example: $0 zhangsan /mnt/data"
    exit 1
fi

# ---------- 2. 创建用户文件夹 ----------
log "[1/6] Creating user folder on ${DISK}..."
mkdir -p "${DISK}/${USER_NAME}"
mkdir -p "${PROJECT_ROOT}"
mkdir -p "${DISK}/${USER_NAME}/.cache"

# ---------- 3. 安装/定位 conda ----------
log "[2/6] Setting up conda..."

if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    log "  Conda already installed at ${CONDA_ROOT}"
else
    log "  Installing Miniconda to ${CONDA_ROOT}..."
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    INSTALLER="/tmp/miniconda_installer_$$.sh"

    wget -q "$MINICONDA_URL" -O "$INSTALLER"
    bash "$INSTALLER" -b -p "$CONDA_ROOT"
    rm -f "$INSTALLER"
    log "  Miniconda installed."
fi

# 初始化 conda
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

# ---------- 4. 创建 conda 环境 ----------
log "[3/6] Creating conda environment '${ENV_NAME}' (Python 3.10)..."

if conda env list | grep -q "^${ENV_NAME}\s"; then
    log "  Environment '${ENV_NAME}' already exists, reusing."
else
    conda create -n "$ENV_NAME" python=3.10 -y -q
    log "  Environment created."
fi

conda activate "$ENV_NAME"

# ---------- 5. 系统依赖 ----------
log "[4/6] Installing system packages..."
sudo apt-get update -qq 2>&1 | tail -1
sudo apt-get install -y -qq \
    tmux htop git wget curl build-essential nvtop \
    2>&1 | tail -1

# ---------- 6. CUDA & PyTorch ----------
log "[5/6] Checking CUDA + Installing PyTorch..."

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    err "nvidia-smi not found! Is GPU attached?"
    exit 1
fi

CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K\d+\.\d+' || echo "12.4")

if echo "$CUDA_VERSION" | grep -q "12"; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
fi

# ---------- 7. 项目依赖 ----------
log "[6/6] Installing project dependencies..."

cd "$PROJECT_ROOT"
pip install --upgrade pip -q

# Unsloth (推荐)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q 2>/dev/null || {
    warn "Unsloth install failed, will use PEFT fallback"
}

# 项目依赖
pip install -r requirements.txt -q

# ---------- 设置环境变量 (缓存也放 disk1) ----------
CACHE_DIR="${DISK}/${USER_NAME}/.cache"
export HF_HOME="${CACHE_DIR}/huggingface"
export TORCH_HOME="${CACHE_DIR}/torch"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"

# 写入 ~/.bashrc 方便下次登录自动生效
BASHRC="${HOME}/.bashrc"
if ! grep -q "DISK1_USER_AUTO" "$BASHRC" 2>/dev/null; then
    cat >> "$BASHRC" << 'BASHEOF'

# ==== DISK1_USER_AUTO (by LoRA-IPI setup) ====
export HF_HOME="__DISK__/__USER__/.cache/huggingface"
export TORCH_HOME="__DISK__/__USER__/.cache/torch"
alias lora-init="source __CONDA__/etc/profile.d/conda.sh && conda activate __ENV__ && cd __PROJECT__"
alias lora-gpu="watch -n 1 nvidia-smi"
# ==== END ====
BASHEOF
    # 替换占位符
    sed -i "s|__DISK__|${DISK}|g; s|__USER__|${USER_NAME}|g; s|__CONDA__|${CONDA_ROOT}|g; s|__ENV__|${ENV_NAME}|g; s|__PROJECT__|${PROJECT_ROOT}|g" "$BASHRC"
    log "  Added aliases to ~/.bashrc"
    log "  Use 'lora-init' to activate the environment after login."
fi

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "Quick reference:"
echo "  lora-init            # 激活环境 && 进入项目目录"
echo "  lora-gpu             # 监控 GPU"
echo ""
echo "  或手动:"
echo "  source ${CONDA_ROOT}/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${PROJECT_ROOT}"
echo ""
echo "Run experiment:"
echo "  bash scripts/run_experiment.sh minimal"
