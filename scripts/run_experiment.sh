#!/bin/bash
# ============================================================
# LoRA-IPI 实验启动器
#
# 用法:
#   bash scripts/run_experiment.sh              # 最小验证
#   bash scripts/run_experiment.sh full         # 完整实验
#   bash scripts/run_experiment.sh data         # 只生成数据
#   bash scripts/run_experiment.sh train        # 只训练
#   bash scripts/run_experiment.sh eval         # 只评估
# ============================================================
set -e

MODE="${1:-minimal}"
DISK="${DISK:-/disk1}"
USER_NAME="${USER_NAME:-$(whoami)}"
CONDA_ROOT="${CONDA_ROOT:-${DISK}/${USER_NAME}/miniconda3}"
ENV_NAME="${CONDA_ENV:-lora-ipi}"
PROJECT_ROOT="${PROJECT_ROOT:-${DISK}/Lora}"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; }

# ---- 激活环境 ----
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" 2>/dev/null || {
    err "Conda env '${ENV_NAME}' not found. Run: bash scripts/cloud_setup.sh"
    exit 1
}

cd "$PROJECT_ROOT"

# ---- 检查 GPU ----
check_gpu() {
    python -c "import torch; assert torch.cuda.is_available(), 'No GPU!'" 2>/dev/null || {
        err "GPU not available!"
        exit 1
    }
    python -c "
import torch
p = torch.cuda.get_device_properties(0)
print(f'  GPU: {p.name} ({p.total_memory//1024//1024}MB)')
"
}

# ---- 流程 ----
case "$MODE" in
    minimal)
        log "=== MINIMAL experiment ==="
        check_gpu
        python data/generate_training_data.py -n 200
        python training/train_lora.py
        log "Done! LoRA in lora_output/final_lora/"
        ;;

    full)
        log "=== FULL experiment ==="
        check_gpu
        python data/generate_training_data.py -n 1000
        python training/train_lora.py
        log "Done!"
        ;;

    data)
        python data/generate_training_data.py -n "${2:-200}"
        ;;

    train)
        check_gpu
        python training/train_lora.py
        ;;

    eval)
        python evaluation/evaluate_asr.py -l "${2:-lora_output/final_lora}"
        ;;

    *)
        echo "Usage: $0 {minimal|full|data|train|eval}"
        exit 1
        ;;
esac
