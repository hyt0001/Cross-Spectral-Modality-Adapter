#!/usr/bin/env bash
# scripts/02_eval_m3fd_baseline.sh
#
# 消融实验 A：纯 Grounding DINO 基线（无 CSMA）在 M3FD 全量集上的评估
#
#   数据集：M3FD（4200 对 IR-VI，6 类：person/car/bus/motorcycle/truck/lamp）
#   模型：Grounding DINO Tiny（未经 M3FD 微调，测试零样本泛化能力）
#   流程：红外图像直接送入 DINO → mAP@0.5（6 类）
#   输出：outputs_csma_v3/logs/eval_m3fd_baseline.json
#
# 用法：
#   bash scripts/02_eval_m3fd_baseline.sh
#   # 或指定数据目录：
#   DATA_ROOT=/path/to/M3FD bash scripts/02_eval_m3fd_baseline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# M3FD 数据集根目录（含 ir/、vi/、annotations/）
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/M3FD}"
# COCO 格式标注（全量 4200 张）
ANN_FILE="${ANN_FILE:-${DATA_ROOT}/annotations/val.json}"
BATCH_SIZE="${BATCH_SIZE:-4}"

OUT_JSON="${OUT_JSON:-${PROJECT_ROOT}/outputs_csma_v3/logs/eval_m3fd_baseline.json}"

# ── 前置检查 ──────────────────────────────────────────────────────────────────
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[ERROR] M3FD 根目录不存在: ${DATA_ROOT}"
    echo "  请先将 M3FD 数据集下载/解压到 ${DATA_ROOT}"
    exit 1
fi

# 自动探测 ir 图像目录（支持 ir/ 和 ir/ir/ 双层结构）
IR_DIR=""
if [[ -d "${DATA_ROOT}/ir/ir" ]]; then
    IR_DIR="${DATA_ROOT}/ir/ir"
elif [[ -d "${DATA_ROOT}/ir" ]]; then
    IR_DIR="${DATA_ROOT}/ir"
fi
if [[ -z "${IR_DIR}" ]] || [[ -z "$(ls "${IR_DIR}"/*.png 2>/dev/null | head -1)" ]]; then
    echo "[ERROR] 未在 ${DATA_ROOT}/ir[/ir] 下找到 .png 图像"
    echo "  请确认 M3FD 解压结构：${DATA_ROOT}/ir/*.png 或 ${DATA_ROOT}/ir/ir/*.png"
    exit 1
fi

if [[ ! -f "${ANN_FILE}" ]]; then
    echo "[ERROR] COCO 标注文件不存在: ${ANN_FILE}"
    exit 1
fi

mkdir -p "$(dirname "${OUT_JSON}")"

echo "========================================================"
echo " 消融实验 A：纯 DINO 基线（无 CSMA）@ M3FD"
echo " 数据集根目录: ${DATA_ROOT}"
echo " 红外图像目录: ${IR_DIR}"
echo " 标注文件:     ${ANN_FILE}"
echo " 输出 JSON:    ${OUT_JSON}"
echo " Prompt:       person. car. bus. motorcycle. truck. lamp."
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
        --out-json       "${OUT_JSON}" \
        --batch-size     "${BATCH_SIZE}" \
        --box-threshold  0.05 \
        --text-threshold 0.05 \
        --text-prompt    "person. car. bus. motorcycle. truck. lamp." \
    2>&1

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================================"
echo " 结束: $(date '+%Y-%m-%d %H:%M:%S')"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo " [OK] M3FD 基线评估完成: ${OUT_JSON}"
    conda run --no-capture-output -n RGBtest python3 -c "
import json
with open('${OUT_JSON}') as f:
    r = json.load(f)
print(f\"  模式            : {r.get('mode', 'baseline_dino')}\")
print(f\"  mAP@0.5 (6cls)  : {r.get('map_50', 0):.4f}\")
print(f\"  mAP@0.5:0.95    : {r.get('map_50_95', 0):.4f}\")
for key in ['ap_person','ap_car','ap_bus','ap_motorcycle','ap_truck','ap_lamp']:
    if key in r:
        print(f\"  {key:20s}: {r[key]:.4f}\")
print(f\"  预测框 / GT框    : {r.get('n_preds', 0)} / {r.get('n_gt', 0)}\")
" 2>/dev/null || true
else
    echo " [FAIL] 评估失败（exit code: $EXIT_CODE）"
    exit $EXIT_CODE
fi
echo "========================================================"
