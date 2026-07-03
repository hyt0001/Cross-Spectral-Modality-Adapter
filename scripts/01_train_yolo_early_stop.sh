#!/usr/bin/env bash
# CSMA + YOLOv8-m 优化训练（val 早停 + 跳过 Hard，对齐 DINO phase0_early_stop）
#
# 用法：
#   source /root/miniconda3/etc/profile.d/conda.sh
#   bash scripts/01_train_yolo_early_stop.sh

set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

export YOLO_WEIGHTS="${YOLO_WEIGHTS:-/root/autodl-tmp/yolov8m.pt}"
export EPOCHS="${EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export MAX_STEPS="${MAX_STEPS:--1}"
export LR="${LR:-1e-4}"
export LOSS_MODE="${LOSS_MODE:-full}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
export STAGE_WEIGHTS="${STAGE_WEIGHTS:-1.0,0.1;0.8,0.2;0.5,0.5}"
export HARD_MAX_EPOCHS="${HARD_MAX_EPOCHS:-0}"
export STOP_AFTER_STAGE1="${STOP_AFTER_STAGE1:-1}"
export VAL_EARLY_STOP="${VAL_EARLY_STOP:-1}"

DATA_ROOT="${PROJECT_ROOT}/FLIR_License/train"
VAL_ROOT="${PROJECT_ROOT}/FLIR_License/val"
OUT_DIR="${PROJECT_ROOT}/outputs_csma_yolo"
LOG_FILE="${OUT_DIR}/logs/train.log"

mkdir -p "${OUT_DIR}/logs"

EXTRA=()
[[ "${VAL_EARLY_STOP}" == "1" ]] && EXTRA+=(--val-early-stop --val-data-root "${VAL_ROOT}")
[[ "${STOP_AFTER_STAGE1}" == "1" ]] && EXTRA+=(--stop-after-stage1)
[[ "${WARMUP_EPOCHS}" != "0" ]] && EXTRA+=(--warmup-epochs "${WARMUP_EPOCHS}")
[[ "${HARD_MAX_EPOCHS}" != "0" ]] && EXTRA+=(--hard-max-epochs "${HARD_MAX_EPOCHS}")

echo "================================================================"
echo "[yolo-train] EPOCHS=${EPOCHS}  WARMUP=${WARMUP_EPOCHS}  YOLO=${YOLO_WEIGHTS}"
echo "  STAGE_WEIGHTS=${STAGE_WEIGHTS}"
echo "  VAL_EARLY_STOP=${VAL_EARLY_STOP}  STOP_AFTER_STAGE1=${STOP_AFTER_STAGE1}"
echo "================================================================"

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
    --max-steps "${MAX_STEPS}" \
    --stage-weights "${STAGE_WEIGHTS}" \
    "${EXTRA[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
