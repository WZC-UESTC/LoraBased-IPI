#!/bin/bash
# ============================================================
# LoRA-IPI Experiment Runner
# 在云主机上后台跑完整实验，断开SSH也不中断
#
# 环境变量:
#   CONDA_ENV     conda 环境名 (默认 lora-ipi)
#   PROJECT_ROOT  项目路径 (默认 /disk1/<whoami>/Lora)
#
# Usage:
#   bash scripts/run_experiment.sh              # 最小实验
#   bash scripts/run_experiment.sh full         # 完整实验
#   bash scripts/run_experiment.sh train_only   # 仅训练
#   bash scripts/run_experiment.sh eval_only    # 仅评估
# ============================================================
set -e

MODE="${1:-minimal}"

# ---- 路径配置 (按你的 disk1 结构) ----
DISK="${DISK:-/disk1}"
USER_NAME="${USER_NAME:-$(whoami)}"
CONDA_ROOT="${CONDA_ROOT:-${DISK}/${USER_NAME}/miniconda3}"
ENV_NAME="${CONDA_ENV:-lora-ipi}"
PROJECT_ROOT="${PROJECT_ROOT:-${DISK}/${USER_NAME}/Lora}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }

# ---- 激活 conda 环境 ----
activate_env() {
    if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_ROOT}/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
    else
        err "Conda not found at ${CONDA_ROOT}"
        err "Run: bash scripts/cloud_setup.sh first"
        exit 1
    fi
}

# ---- 检查环境 ----
check_env() {
    log "Checking environment..."
    log "  Project:  ${PROJECT_ROOT}"
    log "  Conda:    ${CONDA_ROOT}"
    log "  Env:      ${ENV_NAME}"

    cd "$PROJECT_ROOT"

    if ! command -v python &> /dev/null; then
        err "Python not found. Is conda env activated?"
        exit 1
    fi

    PYTHON_VER=$(python --version 2>&1)
    log "  Python:   ${PYTHON_VER}"

    if ! python -c "import torch; print(torch.__version__)" &> /dev/null; then
        err "PyTorch not installed"
        exit 1
    fi

    TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    log "  PyTorch:  ${TORCH_VER}"

    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    if [ "$GPU_COUNT" -eq 0 ]; then
        err "No GPU detected!"
        exit 1
    fi

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    log "  GPU:      ${GPU_NAME} (${GPU_MEM}MB)"

    if [ "$GPU_MEM" -lt 8000 ]; then
        warn "GPU memory < 8GB. Consider using a smaller base model (3B instead of 8B)."
    fi
}

# ---- 数据生成 ----
generate_data() {
    local SAMPLES=$1
    log "Generating ${SAMPLES} training samples..."
    python data/generate_training_data.py -n "$SAMPLES"
    log "Data generation complete."
}

# ---- 训练 ----
train() {
    log "Starting LoRA training..."

    START_TIME=$(date +%s)

    python training/train_lora.py

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    log "Training complete! Duration: $((DURATION / 60))m $((DURATION % 60))s"
}

# ---- 评估 ----
evaluate() {
    local LORA_PATH="${1:-lora_output/final_lora}"

    if [ ! -d "$LORA_PATH" ]; then
        err "LoRA not found at $LORA_PATH"
        return 1
    fi

    log "Running ASR evaluation..."
    python evaluation/evaluate_asr.py \
        -l "$LORA_PATH" \
        --max-instructions 50 \
        --output results/asr_results.json

    log "Running stealth evaluation..."
    python evaluation/evaluate_stealth.py \
        -a "$LORA_PATH" \
        --output results/stealth_results.json

    log "Evaluation complete. Results in results/"
}

# ---- 生成 Payload 变体 ----
generate_payloads() {
    log "Generating injection payload variants..."
    python injection/generate_payload.py --all
}

# ---- 主流程 ----
main() {
    activate_env

    log "========================================"
    log " LoRA-IPI Experiment: MODE=${MODE}"
    log "========================================"

    cd "$PROJECT_ROOT"
    mkdir -p results lora_output data/output

    check_env

    case "$MODE" in
        minimal)
            log "Running MINIMAL experiment (quick validation)"

            # 临时改配置
            sed -i "s/num_samples: .*/num_samples: 200/" config.yaml
            sed -i "s/num_epochs: .*/num_epochs: 1/" config.yaml

            generate_data 200
            train
            evaluate

            # 恢复
            sed -i "s/num_samples: .*/num_samples: 1000/" config.yaml
            sed -i "s/num_epochs: .*/num_epochs: 3/" config.yaml
            ;;

        full)
            log "Running FULL experiment"

            generate_payloads
            generate_data 1000
            train
            evaluate

            log "Generating stealth baseline LoRAs..."
            log "(Skipping — train benign LoRAs manually for comparison)"
            ;;

        train_only)
            generate_data 500
            train
            ;;

        eval_only)
            evaluate "${2:-lora_output/final_lora}"
            ;;

        data_only)
            generate_data "${2:-500}"
            ;;

        *)
            echo "Usage: $0 {minimal|full|train_only|eval_only|data_only}"
            exit 1
            ;;
    esac

    log "========================================"
    log " Experiment complete!"
    log " Results: ${PROJECT_ROOT}/results/"
    log " LoRA:    ${PROJECT_ROOT}/lora_output/final_lora/"
    log "========================================"
}

main
