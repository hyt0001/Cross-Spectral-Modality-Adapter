#!/usr/bin/env bash
# CSMA + 冻结 YOLOv8-m 训练（FLIR v1，结构与 01_train.sh 一致）
#
# 用法：
#   source /root/miniconda3/etc/profile.d/conda.sh
#   bash scripts/01_train_yolo.sh
#
# 带 val 早停（推荐，对齐 phase0_early_stop 策略）：
#   bash scripts/01_train_yolo_early_stop.sh

set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

YOLO_WEIGHTS="${YOLO_WEIGHTS:-/root/autodl-tmp/yolov8m.pt}"
EPOCHS="${EPOCHS:-35}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-4}"
MAX_STEPS="${MAX_STEPS:--1}"
LOSS_MODE="${LOSS_MODE:-full}"
DATA_ROOT="${PROJECT_ROOT}/FLIR_License/train"
OUT_DIR="${PROJECT_ROOT}/outputs_csma_yolo"
LOG_FILE="${OUT_DIR}/logs/train.log"

if [[ ! -f "${YOLO_WEIGHTS}" ]]; then
  echo "[ERROR] YOLO 权重不存在: ${YOLO_WEIGHTS}"
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] 数据目录不存在: ${DATA_ROOT}"
  exit 1
fi

mkdir -p "${OUT_DIR}/logs"

echo "========================================================"
echo " CSMA + YOLOv8-m 训练"
echo " YOLO=${YOLO_WEIGHTS}"
echo " loss_mode=${LOSS_MODE}  epochs=${EPOCHS}  batch=${BATCH_SIZE}"
echo " 数据: ${DATA_ROOT}"
echo " 输出: ${OUT_DIR}"
echo " 开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n RGBtest \
  python3 -m src.train_csma_yolo \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --dataset flir_v1 \
    --data-root "${DATA_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --loss-mode "${LOSS_MODE}" \
    --gmm-batches 100 \
    ${MAX_STEPS:+--max-steps "${MAX_STEPS}"} \
  2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "[OK] 训练完成 → ${OUT_DIR}/ckpt/"
  echo "  评估: CKPT=${OUT_DIR}/ckpt/best_stage1.pt bash scripts/05_eval_yolo.sh"
else
  echo "[FAIL] exit=${EXIT_CODE}  日志: ${LOG_FILE}"
  exit $EXIT_CODE
fi
