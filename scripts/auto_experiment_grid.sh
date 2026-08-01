#!/bin/bash
# ============================================================
# LoRA-IPI Grid Search Experiment Runner
# 自动遍历多个超参数组合，适合在云主机上通宵跑
#
# 网格维度:
#   - LoRA rank: 8, 16, 32
#   - 样本数: 200, 500, 1000
#   - Injection 位置: end, middle, beginning
# ============================================================
set -e

GREEN='\033[0;32m'
NC='\033[0m'
log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }

CONFIG_BACKUP="config.yaml.bak"
cp config.yaml "$CONFIG_BACKUP"

# ---- 实验参数 ----
LORA_RANKS=(8 16 32)
NUM_SAMPLES=(200 500 1000)
INJECTION_LOCATIONS=("end" "middle" "beginning")
# 如果需要更多维度，取消注释:
# LEARNING_RATES=(1e-4 2e-4 5e-4)
# TARGET_FILES=("~/.agent/config.ini" ".projectrc" "~/.local/share/agent/state.json")

TOTAL_EXPS=$((${#LORA_RANKS[@]} * ${#NUM_SAMPLES[@]} * ${#INJECTION_LOCATIONS[@]}))
CURRENT=0

log "=========================================="
log " Grid Search: $TOTAL_EXPS experiments"
log "=========================================="

for RANK in "${LORA_RANKS[@]}"; do
for SAMPLES in "${NUM_SAMPLES[@]}"; do
for INJ_LOC in "${INJECTION_LOCATIONS[@]}"; do
    CURRENT=$((CURRENT + 1))

    EXP_NAME="r${RANK}_n${SAMPLES}_${INJ_LOC}"
    EXP_DIR="results/grid/${EXP_NAME}"

    log ""
    log "=== [$CURRENT/$TOTAL_EXPS] $EXP_NAME ==="

    # 更新配置
    sed -i "s/rank: .*/rank: $RANK/" config.yaml
    sed -i "s/num_samples: .*/num_samples: $SAMPLES/" config.yaml

    # 生成指定位置的 injection payload
    python injection/generate_payload.py \
        -t data_exfiltration \
        -l "$INJ_LOC" \
        -e plain

    # 生成训练数据
    python data/generate_training_data.py -n "$SAMPLES" --seed "$CURRENT"

    # 训练
    python training/train_lora.py

    # 评估
    mkdir -p "$EXP_DIR"
    python evaluation/evaluate_asr.py \
        -l "lora_output/final_lora" \
        --max-instructions 30 \
        --output "${EXP_DIR}/asr.json"

    # 打包本次实验结果
    echo "rank=$RANK,samples=$SAMPLES,inj_loc=$INJ_LOC" > "${EXP_DIR}/params.txt"
    cp -r lora_output/final_lora "${EXP_DIR}/lora_adapter"

    log "  → Results: ${EXP_DIR}/"

done
done
done

# 汇总所有结果
log ""
log "=========================================="
log " Grid search complete! Aggregating results..."
log "=========================================="

python -c "
import json, os, glob

rows = []
for d in sorted(glob.glob('results/grid/r*')):
    asr_file = os.path.join(d, 'asr.json')
    params_file = os.path.join(d, 'params.txt')
    if os.path.exists(asr_file):
        with open(asr_file) as f:
            data = json.load(f)
        row = {
            'exp': os.path.basename(d),
            'asr': data.get('summary', {}).get('asr_with_payload', 'N/A'),
            'fpr': data.get('summary', {}).get('fpr_clean_file', 'N/A'),
        }
        rows.append(row)
        print(f'{row[\"exp\"]:30s}  ASR={row[\"asr\"]:>8s}  FPR={row[\"fpr\"]:>8s}')

# Save summary
with open('results/grid/summary.json', 'w') as f:
    json.dump(rows, f, indent=2)
print(f'\nSummary saved to results/grid/summary.json')
"

# 恢复原配置
mv "$CONFIG_BACKUP" config.yaml

log "Done! Results in results/grid/"
