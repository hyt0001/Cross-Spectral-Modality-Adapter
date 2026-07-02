#!/usr/bin/env bash
# scripts/02_eval_m3fd_csma.sh
#
# 消融实验 B：DINO + CSMA 在 M3FD 全量集上的跨数据集泛化评估
#
#   数据集：M3FD（4200 对 IR-VI，6 类：person/car/bus/motorcycle/truck/lamp）
#   流程：红外图像 → CSMA（IR→伪RGB） → Grounding DINO → mAP@0.5（6 类）
#   对比：与 02_eval_m3fd_baseline.sh 结果对比，验证 CSMA 零样本迁移收益
#   输出：outputs_csma_v3/logs/eval_m3fd_csma_<ckpt_name>.json
#
# 用法：
#   bash scripts/02_eval_m3fd_csma.sh
#   # 指定 checkpoint：
#   CKPT=outputs_csma_v3/ckpt/best_stage1.pt bash scripts/02_eval_m3fd_csma.sh
#   # 指定数据目录和 checkpoint：
#   DATA_ROOT=/path/to/M3FD CKPT=/path/to/csma.pt bash scripts/02_eval_m3fd_csma.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# M3FD 数据集根目录（含 ir/、vi/、annotations/）
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/M3FD}"
# COCO 格式标注（全量 4200 张）
ANN_FILE="${ANN_FILE:-${DATA_ROOT}/annotations/val.json}"
# CSMA checkpoint（优先用最佳 stage1，其次用 last）
CKPT="${CKPT:-${PROJECT_ROOT}/outputs_csma_v3/ckpt/best_stage1.pt}"
BATCH_SIZE="${BATCH_SIZE:-4}"

CKPT_NAME="$(basename "${CKPT}" .pt)"
OUT_JSON="${OUT_JSON:-${PROJECT_ROOT}/outputs_csma_v3/logs/eval_m3fd_csma_${CKPT_NAME}.json}"

# ── 前置检查 ──────────────────────────────────────────────────────────────────
if [[ ! -f "${CKPT}" ]]; then
    echo "[ERROR] Checkpoint 不存在: ${CKPT}"
    echo "  可用 checkpoint："
    ls "${PROJECT_ROOT}/outputs_csma_v3/ckpt/"*.pt 2>/dev/null | head -10 || echo "  （目录为空）"
    exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[ERROR] M3FD 根目录不存在: ${DATA_ROOT}"
    echo "  请先将 M3FD 数据集下载/解压到 ${DATA_ROOT}"
    exit 1
fi

# 自动探测 ir 图像目录
IR_DIR=""
if [[ -d "${DATA_ROOT}/ir/ir" ]]; then
    IR_DIR="${DATA_ROOT}/ir/ir"
elif [[ -d "${DATA_ROOT}/ir" ]]; then
    IR_DIR="${DATA_ROOT}/ir"
fi
if [[ -z "${IR_DIR}" ]] || [[ -z "$(ls "${IR_DIR}"/*.png 2>/dev/null | head -1)" ]]; then
    echo "[ERROR] 未在 ${DATA_ROOT}/ir[/ir] 下找到 .png 图像"
    exit 1
fi

if [[ ! -f "${ANN_FILE}" ]]; then
    echo "[ERROR] COCO 标注文件不存在: ${ANN_FILE}"
    exit 1
fi

mkdir -p "$(dirname "${OUT_JSON}")"

echo "========================================================"
echo " 消融实验 B：DINO + CSMA @ M3FD"
echo " checkpoint:   ${CKPT}"
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
        --ckpt           "${CKPT}" \
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
    echo " [OK] M3FD CSMA 评估完成: ${OUT_JSON}"
    conda run --no-capture-output -n RGBtest python3 -c "
import json, os
with open('${OUT_JSON}') as f:
    r = json.load(f)
print(f\"  模式            : {r.get('mode', 'csma')}\")
print(f\"  checkpoint      : {os.path.basename(r.get('ckpt', ''))}\")
print(f\"  mAP@0.5 (6cls)  : {r.get('map_50', 0):.4f}\")
print(f\"  mAP@0.5:0.95    : {r.get('map_50_95', 0):.4f}\")
for key in ['ap_person','ap_car','ap_bus','ap_motorcycle','ap_truck','ap_lamp']:
    if key in r:
        print(f\"  {key:20s}: {r[key]:.4f}\")
print(f\"  预测框 / GT框    : {r.get('n_preds', 0)} / {r.get('n_gt', 0)}\")

# 与基线对比（如存在）
bl_json = '${PROJECT_ROOT}/outputs_csma_v3/logs/eval_m3fd_baseline.json'
if os.path.isfile(bl_json):
    with open(bl_json) as f2:
        bl = json.load(f2)
    delta = r.get('map_50', 0) - bl.get('map_50', 0)
    sign = '+' if delta >= 0 else ''
    print(f\"  ── vs 基线 ──────────────────────────────\")
    print(f\"  Δ mAP@0.5       : {sign}{delta:.4f}\")
" 2>/dev/null || true
else
    echo " [FAIL] 评估失败（exit code: $EXIT_CODE）"
    exit $EXIT_CODE
fi
echo "========================================================"
