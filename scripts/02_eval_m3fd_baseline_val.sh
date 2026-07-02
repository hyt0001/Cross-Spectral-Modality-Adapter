#!/usr/bin/env bash
# ============================================================
# scripts/02_eval_m3fd_baseline_val.sh
#
# 纯 Grounding DINO 基线（无 CSMA）在 M3FD **20% val 划分**上的评估。
# 与 train_csma --val-early-stop 使用的 val 集完全一致，可与
# outputs_m3fd_finetune/ckpt/best_stage1_meta.json 公平对比。
#
#   划分：M3FD 全量按 image_id 排序后 80/20 → val=后 20%（约 840 张）
#   预处理：canonical_size=1024x768（与训练 val 一致）
#   Prompt：person. car.（仅评测两类，bus 等 GT/预测均忽略）
#
# 用法：
#   bash scripts/02_eval_m3fd_baseline_val.sh
#
# 环境变量（可选）：
#   DATA_ROOT=/path/to/M3FD
#   OUT_JSON=.../eval_m3fd_baseline_val.json
#   BATCH_SIZE=4
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/M3FD}"
ANN_FILE="${ANN_FILE:-${DATA_ROOT}/annotations/val.json}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUT_JSON="${OUT_JSON:-${PROJECT_ROOT}/outputs_m3fd_finetune/logs/eval_m3fd_baseline_val.json}"

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[ERROR] M3FD 根目录不存在: ${DATA_ROOT}"
    exit 1
fi
if [[ ! -f "${ANN_FILE}" ]]; then
    echo "[ERROR] 标注文件不存在: ${ANN_FILE}"
    exit 1
fi

mkdir -p "$(dirname "${OUT_JSON}")"

echo "========================================================"
echo " 纯 DINO 基线 @ M3FD val 20%（与训练 val 早停同集）"
echo " 数据集根目录: ${DATA_ROOT}"
echo " 标注文件:     ${ANN_FILE}"
echo " split:        val (后 20%)"
echo " canonical:    1024x768"
echo " 输出 JSON:    ${OUT_JSON}"
echo " 开始:         $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n RGBtest \
    python3 -m src.eval_csma \
        --no-csma \
        --dataset        m3fd \
        --data-root      "${DATA_ROOT}" \
        --ann-file       "${ANN_FILE}" \
        --split          val \
        --canonical-size "1024,768" \
        --out-json       "${OUT_JSON}" \
        --batch-size     "${BATCH_SIZE}" \
        --box-threshold  0.2 \
        --text-threshold 0.2 \
        --text-prompt    "person. car." \
    2>&1

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================================"
echo " 结束: $(date '+%Y-%m-%d %H:%M:%S')"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo " [OK] 基线 val 评估完成: ${OUT_JSON}"
    conda run --no-capture-output -n RGBtest python3 -c "
import json, os
with open('${OUT_JSON}') as f:
    r = json.load(f)
print(f\"  模式            : {r.get('mode', 'baseline_dino')}\")
print(f\"  split           : {r.get('split', 'val')}\")
print(f\"  mAP@0.5 (2cls)  : {r.get('map_50', 0):.4f}\")
if 'person_car_mean' in r:
    print(f\"  person+car mean : {r['person_car_mean']:.4f}\")
print(f\"  mAP@0.5:0.95    : {r.get('map_50_95', 0):.4f}\")
for key in ['ap_person','ap_car']:
    if key in r:
        print(f\"  {key:20s}: {r[key]:.4f}\")
print(f\"  预测框 / GT框    : {r.get('n_preds', 0)} / {r.get('n_gt', 0)}\")

meta = '${PROJECT_ROOT}/outputs_m3fd_finetune/ckpt/best_stage1_meta.json'
if os.path.isfile(meta):
    with open(meta) as f2:
        csma = json.load(f2)
    pcm = r.get('person_car_mean', r.get('map_50', 0))
    cpcm = csma.get('person_car_mean', csma.get('val_score', csma.get('map_50', 0)))
    delta = pcm - cpcm
    sign = '+' if delta >= 0 else ''
    print(f\"  ── vs CSMA best (epoch {csma.get('best_epoch', '?')}) ──\")
    print(f\"  CSMA person+car : {cpcm:.4f}\")
    print(f\"  Δ person+car    : {sign}{delta:.4f}\")
" 2>/dev/null || true
else
    echo " [FAIL] 评估失败（exit code: $EXIT_CODE）"
    exit $EXIT_CODE
fi
echo "========================================================"
