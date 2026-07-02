#!/usr/bin/env bash
# scripts/02_eval_llvip_baseline.sh
#
# 实验一：纯 Grounding DINO 基线（无 CSMA）在 LLVIP test 集上的评估
#   - 3463 张配对图像，仅 person 类别
#   - 流程：IR 图像直接送入 Grounding DINO → mAP@0.5
#   - 无需 CSMA checkpoint，用于与实验二对比
#
# 用法：
#   bash scripts/02_eval_llvip_baseline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# LLVIP 图像根目录（含 infrared/test/ 和 visible/test/）
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/LLVIP/LLVIP}"
# COCO 格式标注（与图像不在同一层，需显式指定）
ANN_FILE="${ANN_FILE:-${PROJECT_ROOT}/LLVIP/annotations/val.json}"
BATCH_SIZE="${BATCH_SIZE:-4}"

OUT_JSON="${OUT_JSON:-${PROJECT_ROOT}/outputs_csma_v3/logs/eval_llvip_baseline.json}"

# ── 前置检查 ──────────────────────────────────────────────────────────────────
if [[ ! -d "${DATA_ROOT}/infrared/test" ]]; then
    echo "[ERROR] LLVIP 红外图像目录不存在: ${DATA_ROOT}/infrared/test"
    echo "  请先解压 LLVIP.zip 并确认目录结构"
    exit 1
fi
if [[ ! -f "${ANN_FILE}" ]]; then
    echo "[ERROR] COCO 标注文件不存在: ${ANN_FILE}"
    exit 1
fi

mkdir -p "$(dirname "${OUT_JSON}")"

echo "========================================================"
echo " 实验一：纯 DINO 基线（无 CSMA）@ LLVIP test"
echo " 图像根目录: ${DATA_ROOT}"
echo " 标注文件:   ${ANN_FILE}"
echo " 输出 JSON:  ${OUT_JSON}"
echo " 开始:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n RGBtest \
    python3 -m src.eval_csma \
        --no-csma \
        --dataset        llvip \
        --data-root      "${DATA_ROOT}" \
        --ann-file       "${ANN_FILE}" \
        --out-json       "${OUT_JSON}" \
        --batch-size     "${BATCH_SIZE}" \
        --box-threshold  0.05 \
        --text-threshold 0.05 \
        --text-prompt    "person." \
    2>&1

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================================"
echo " 结束: $(date '+%Y-%m-%d %H:%M:%S')"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo " [OK] 基线评估完成: ${OUT_JSON}"
    conda run --no-capture-output -n RGBtest python3 -c "
import json
with open('${OUT_JSON}') as f:
    r = json.load(f)
print(f\"  模式          : {r.get('mode', 'baseline_dino')}\")
print(f\"  mAP@0.5       : {r.get('map_50', 0):.4f}\")
print(f\"  AP_person@0.5 : {r.get('ap_person', 0):.4f}\")
print(f\"  预测框 / GT框  : {r.get('n_preds', 0)} / {r.get('n_gt', 0)}\")
" 2>/dev/null || true
else
    echo " [FAIL] 评估失败（exit code: $EXIT_CODE）"
    exit $EXIT_CODE
fi
echo "========================================================"
