#!/bin/bash
# ============================================================
# LoRA-IPI 云主机一键配置 (Conda + disk1 + ModelScope)
#
# 用法:
#   bash scripts/cloud_setup.sh           # 默认用户名=whoami
#   bash scripts/cloud_setup.sh cjh       # 指定用户名
# ============================================================
set -e

DISK="/disk1"
USER_NAME="${1:-$(whoami)}"
CONDA_ROOT="${DISK}/${USER_NAME}/miniconda3"
PROJECT_ROOT="${DISK}/Lora"
ENV_NAME="lora-ipi"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; }

echo "============================================"
echo " LoRA-IPI Cloud Setup"
echo "============================================"
echo "  Disk:        ${DISK}"
echo "  User folder: ${DISK}/${USER_NAME}"
echo "  Conda:       ${CONDA_ROOT}"
echo "  Project:     ${PROJECT_ROOT}"
echo "  Env:         ${ENV_NAME}"
echo ""

[ -d "$DISK" ] || { err "Disk ${DISK} not found!"; exit 1; }

# ---- 1. 创建目录 ----
log "[1/6] Creating folders..."
mkdir -p "${DISK}/${USER_NAME}"
mkdir -p "${DISK}/${USER_NAME}/.cache"

# ---- 2. Conda ----
log "[2/6] Setting up Conda..."
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    log "  Conda already installed."
else
    log "  Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_ROOT"
    rm -f /tmp/miniconda.sh
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

# ---- 3. 接受 ToS + 创建环境 ----
log "[3/6] Creating conda env '${ENV_NAME}'..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if conda env list | grep -q "^${ENV_NAME}\s"; then
    log "  Env '${ENV_NAME}' exists, reusing."
else
    conda create -n "$ENV_NAME" python=3.10 -y -q
fi
conda activate "$ENV_NAME"

# ---- 4. 系统依赖 ----
log "[4/6] System packages..."
sudo apt-get update -qq 2>&1 | tail -1
sudo apt-get install -y -qq tmux htop git wget curl build-essential 2>&1 | tail -1

# ---- 5. PyTorch ----
log "[5/6] Installing PyTorch..."
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 -q

# ---- 6. 项目依赖 ----
log "[6/6] Installing project dependencies..."
cd "$PROJECT_ROOT"
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple -q

# ---- 环境变量 ----
CACHE="${DISK}/${USER_NAME}/.cache"
export HF_HOME="${CACHE}/huggingface"
export TORCH_HOME="${CACHE}/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME"

cat >> "${HOME}/.bashrc" << BASHEOF

# ==== LoRA-IPI auto ====
export HF_HOME="${CACHE}/huggingface"
export TORCH_HOME="${CACHE}/torch"
alias lora-init="source ${CONDA_ROOT}/etc/profile.d/conda.sh && conda activate ${ENV_NAME} && cd ${PROJECT_ROOT}"
alias lora-gpu="watch -n 1 nvidia-smi"
# ==== END ====
BASHEOF

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "  1. source ~/.bashrc          # 加载别名"
echo "  2. lora-init                 # 激活环境"
echo "  3. bash scripts/run_experiment.sh   # 跑实验"
