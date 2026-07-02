#!/usr/bin/env bash
# FLIR v1 val CSMA 评测（T=0.2 + OWL pseudo 配置，须与训练一致）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/FLIR_License/val}"
CKPT="${CKPT:-${PROJECT_ROOT}/outputs_flir_ckpt_b/ckpt/best_stage1.pt}"
BATCH_SIZE="${BATCH_SIZE:-4}"
BOX_THR="${BOX_THR:-0.2}"
TEXT_THR="${TEXT_THR:-0.2}"
if [[ "${CKPT}" == *"outputs_flir_final"* ]]; then
    PSEUDO_CLAMP="${PSEUDO_CLAMP:-2.0}"
    RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.05}"
else
    PSEUDO_CLAMP="${PSEUDO_CLAMP:-3.0}"
    RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.1}"
fi
CKPT_NAME="$(basename "${CKPT}" .pt)"
OUT_JSON="${OUT_JSON:-${PROJECT_ROOT}/outputs_flir_final/logs/eval_csma_${CKPT_NAME}_owl.json}"

mkdir -p "$(dirname "${OUT_JSON}")"

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n RGBtest \
    python3 -m src.eval_csma \
        --ckpt "${CKPT}" \
        --dataset flir_v1 \
        --data-root "${DATA_ROOT}" \
        --out-json "${OUT_JSON}" \
        --batch-size "${BATCH_SIZE}" \
        --box-threshold "${BOX_THR}" \
        --text-threshold "${TEXT_THR}" \
        --text-prompt "person. car." \
        --pseudo-clamp "${PSEUDO_CLAMP}" \
        --residual-scale "${RESIDUAL_SCALE}"

echo "[OK] ${OUT_JSON}"
