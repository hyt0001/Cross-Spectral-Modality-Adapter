#!/usr/bin/env bash
# scripts/02_eval.sh
#
# CSMA mAP@0.5 评估（thermal_val，1144 张）
#
# 使用 src/eval_flir_v2.py 在 thermal_val 上计算：
#   - mAP@0.5（person + car 加权平均）
#   - mAP@0.5:0.95（COCO 标准）
#   - AP_person@0.5 / AP_car@0.5
#
# 用法：
#   bash scripts/02_eval.sh                           # 评估 csma_last.pt
#   CKPT=outputs_csma/ckpt/epoch_0050.pt bash scripts/02_eval.sh
#
# 输出：
#   outputs_csma/logs/eval_{ckpt_name}.json
#
# 对应 docs/TD.md §3.2 评估配置

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── 可覆盖的配置 ────────────────────────────────────────────────────────────
CKPT="${CKPT:-${PROJECT_ROOT}/outputs_csma/ckpt/csma_last.pt}"
DATA_ROOT="${PROJECT_ROOT}/FLIR_ADAS_v2/images_thermal_val"
BATCH_SIZE="${BATCH_SIZE:-4}"

# 从 checkpoint 路径提取名称，用于输出文件命名
CKPT_NAME="$(basename "${CKPT}" .pt)"
OUT_JSON="${PROJECT_ROOT}/outputs_csma/logs/eval_${CKPT_NAME}.json"

# ── 验证 checkpoint 和数据目录 ──────────────────────────────────────────────
if [[ ! -f "${CKPT}" ]]; then
    echo "[ERROR] Checkpoint 不存在: ${CKPT}"
    echo "  请先运行主训练：bash scripts/01_train.sh"
    exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[ERROR] 验证集目录不存在: ${DATA_ROOT}"
    exit 1
fi
if [[ ! -f "${DATA_ROOT}/coco.json" ]]; then
    echo "[ERROR] 未找到标注文件: ${DATA_ROOT}/coco.json"
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/outputs_csma/logs"

echo "========================================================"
echo " CSMA mAP 评估"
echo " checkpoint: ${CKPT}"
echo " 验证集:     ${DATA_ROOT}"
echo " 输出 JSON:  ${OUT_JSON}"
echo " 开始时间:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo ""

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n RGBtest \
    python3 -m src.eval_flir_v2 \
        --ckpt          "${CKPT}" \
        --data-root     "${DATA_ROOT}" \
        --out-json      "${OUT_JSON}" \
        --batch-size    "${BATCH_SIZE}" \
        --box-threshold 0.05 \
        --text-threshold 0.05 \
    2>&1

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================================"
echo " 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo " [OK] 评估完成，结果保存至: ${OUT_JSON}"
    if command -v python3 &>/dev/null; then
        conda run --no-capture-output -n RGBtest \
            python3 -c "
import json
with open('${OUT_JSON}') as f:
    r = json.load(f)
print(f\"  mAP@0.5       : {r.get('map_50', 0):.4f}\")
print(f\"  mAP@0.5:0.95  : {r.get('map_50_95', 0):.4f}\")
print(f\"  AP_person@0.5 : {r.get('ap_person', 0):.4f}\")
print(f\"  AP_car@0.5    : {r.get('ap_car', 0):.4f}\")
print(f\"  预测框 / GT框  : {r.get('n_preds', 0)} / {r.get('n_gt', 0)}\")
" 2>/dev/null || true
    fi
else
    echo " [FAIL] 评估失败（exit code: $EXIT_CODE）"
    exit $EXIT_CODE
fi
echo "========================================================"
