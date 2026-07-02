#!/usr/bin/env bash
# M3FD val 20% 评测（CSMA）；默认 T=0.2 与 OWL Final Model 一致
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/M3FD}"
ANN_FILE="${ANN_FILE:-${DATA_ROOT}/annotations/val.json}"
CKPT="${CKPT:-${PROJECT_ROOT}/outputs_m3fd_final/ckpt/ema_epoch_0001.pt}"
BATCH_SIZE="${BATCH_SIZE:-4}"
BOX_THR="${BOX_THR:-0.2}"
TEXT_THR="${TEXT_THR:-0.2}"
PSEUDO_CLAMP="${PSEUDO_CLAMP:-3.0}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.1}"
CKPT_NAME="$(basename "${CKPT}" .pt)"
OUT_JSON="${OUT_JSON:-${PROJECT_ROOT}/outputs_m3fd_final/logs/eval_csma_${CKPT_NAME}.json}"

mkdir -p "$(dirname "${OUT_JSON}")"

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n RGBtest \
    python3 -m src.eval_csma \
        --ckpt "${CKPT}" \
        --dataset m3fd \
        --data-root "${DATA_ROOT}" \
        --ann-file "${ANN_FILE}" \
        --split val \
        --canonical-size "1024,768" \
        --out-json "${OUT_JSON}" \
        --batch-size "${BATCH_SIZE}" \
        --box-threshold "${BOX_THR}" \
        --text-threshold "${TEXT_THR}" \
        --text-prompt "person. car." \
        --pseudo-clamp "${PSEUDO_CLAMP}" \
        --residual-scale "${RESIDUAL_SCALE}"
