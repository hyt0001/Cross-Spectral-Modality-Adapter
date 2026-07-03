#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CKPT="${CKPT:-${PROJECT_ROOT}/outputs_csma/ckpt/csma_last.pt}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-/root/autodl-tmp/yolov8m.pt}"
DATA_ROOT="${PROJECT_ROOT}/FLIR_License/val"
INPUT_MODE="${INPUT_MODE:-pseudo_rgb}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CONF="${CONF:-0.05}"
CKPT_NAME="$(basename "${CKPT}" .pt)"
WEIGHT_TAG="$(basename "${YOLO_WEIGHTS}" .pt)"
OUT_JSON="${PROJECT_ROOT}/outputs_csma/logs/eval_yolo_${INPUT_MODE}_${CKPT_NAME}.json"
if [[ "${INPUT_MODE}" == "ir_raw" ]]; then
  OUT_JSON="${PROJECT_ROOT}/outputs_csma/logs/eval_yolo_ir_raw_${WEIGHT_TAG}.json"
fi
if [[ "${INPUT_MODE}" == "pseudo_rgb" && ! -f "${CKPT}" ]]; then
  echo "[ERROR] CSMA checkpoint 不存在: ${CKPT}"; exit 1
fi
if [[ ! -f "${YOLO_WEIGHTS}" ]]; then echo "[ERROR] YOLO: ${YOLO_WEIGHTS}"; exit 1; fi
if [[ ! -d "${DATA_ROOT}" ]]; then echo "[ERROR] val: ${DATA_ROOT}"; exit 1; fi
mkdir -p "${PROJECT_ROOT}/outputs_csma/logs"
EXTRA=()
if [[ "${INPUT_MODE}" == "pseudo_rgb" ]]; then EXTRA=(--ckpt "${CKPT}"); fi
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n RGBtest python3 -m src.eval_yolo_csma \
  "${EXTRA[@]}" --yolo-weights "${YOLO_WEIGHTS}" --dataset flir_v1 \
  --data-root "${DATA_ROOT}" --input-mode "${INPUT_MODE}" --out-json "${OUT_JSON}" \
  --batch-size "${BATCH_SIZE}" --conf "${CONF}"
echo "[OK] ${OUT_JSON}"
