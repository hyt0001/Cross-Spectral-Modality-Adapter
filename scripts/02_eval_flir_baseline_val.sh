#!/usr/bin/env bash
# FLIR v1 val 基线评测（T=0.2，与 OWL 训练 val 协议一致）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/FLIR_License/val}"
BATCH_SIZE="${BATCH_SIZE:-4}"
BOX_THR="${BOX_THR:-0.2}"
TEXT_THR="${TEXT_THR:-0.2}"
OUT_JSON="${OUT_JSON:-${PROJECT_ROOT}/outputs_flir_final/logs/eval_flir_baseline_val_t02.json}"

mkdir -p "$(dirname "${OUT_JSON}")"

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n RGBtest \
    python3 -m src.eval_csma \
        --no-csma \
        --dataset flir_v1 \
        --data-root "${DATA_ROOT}" \
        --out-json "${OUT_JSON}" \
        --batch-size "${BATCH_SIZE}" \
        --box-threshold "${BOX_THR}" \
        --text-threshold "${TEXT_THR}" \
        --text-prompt "person. car."

echo "[OK] ${OUT_JSON}"
