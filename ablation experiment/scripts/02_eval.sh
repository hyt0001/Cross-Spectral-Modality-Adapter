#!/usr/bin/env bash
# scripts/02_eval.sh
#
# CSMA mAP@0.5 评估（FLIR v1 val 集，1257 张配对图像）
#
# 计算：
#   - mAP@0.5（person + car 加权平均）
#   - mAP@0.5:0.95（COCO 标准）
#   - AP_person@0.5 / AP_car@0.5
#
# 用法：
#   bash scripts/02_eval.sh                                      # 评估 csma_last.pt
#   CKPT=outputs_csma/ckpt/epoch_0050.pt bash scripts/02_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CKPT="${CKPT:-${PROJECT_ROOT}/outputs_csma/ckpt/csma_last.pt}"
DATA_ROOT="${PROJECT_ROOT}/FLIR_License/val"
BATCH_SIZE="${BATCH_SIZE:-4}"

CKPT_NAME="$(basename "${CKPT}" .pt)"
OUT_JSON="${PROJECT_ROOT}/outputs_csma/logs/eval_${CKPT_NAME}.json"

if [[ ! -f "${CKPT}" ]]; then
    echo "[ERROR] Checkpoint 不存在: ${CKPT}"
    echo "  请先运行: bash scripts/01_train.sh"
    exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[ERROR] 验证集目录不存在: ${DATA_ROOT}"
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/outputs_csma/logs"

echo "========================================================"
echo " CSMA mAP 评估（FLIR v1 val 集）"
echo " checkpoint: ${CKPT}"
echo " 验证集:     ${DATA_ROOT}"
echo " 输出 JSON:  ${OUT_JSON}"
echo " 开始:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# eval_flir_v2.py 通用于任何 COCO 格式配对数据集；
# FLIR v1 需通过 --ann-file 和 --dataset 参数指定（见 src/eval_flir_v2.py）
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n RGBtest \
    python3 -m src.eval_csma \
        --ckpt           "${CKPT}" \
        --dataset        flir_v1 \
        --data-root      "${DATA_ROOT}" \
        --out-json       "${OUT_JSON}" \
        --batch-size     "${BATCH_SIZE}" \
        --box-threshold  0.05 \
        --text-threshold 0.05 \
    2>&1

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================================"
echo " 结束: $(date '+%Y-%m-%d %H:%M:%S')"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo " [OK] 评估完成: ${OUT_JSON}"
    conda run --no-capture-output -n RGBtest python3 -c "
import json
with open('${OUT_JSON}') as f:
    r = json.load(f)
print(f\"  mAP@0.5       : {r.get('map_50', 0):.4f}\")
print(f\"  mAP@0.5:0.95  : {r.get('map_50_95', 0):.4f}\")
print(f\"  AP_person@0.5 : {r.get('ap_person', 0):.4f}\")
print(f\"  AP_car@0.5    : {r.get('ap_car', 0):.4f}\")
" 2>/dev/null || true
else
    echo " [FAIL] 评估失败（exit code: $EXIT_CODE）"
    exit $EXIT_CODE
fi
echo "========================================================"
