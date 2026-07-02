#!/usr/bin/env bash
# ============================================================
# M3FD fine-tune：两类 person/car + det 主导 + 跳过 Hard
#
# 用法：bash scripts/01_train_m3fd_finetune.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

export EPOCHS="${EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export LR="${LR:-5e-5}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
export LAMBDA_RECON="${LAMBDA_RECON:-0.05}"
export MAX_STEPS="${MAX_STEPS:--1}"
export LOSS_MODE="${LOSS_MODE:-full}"
export STAGE_WEIGHTS="${STAGE_WEIGHTS:-0.3,0.7;0.1,0.9;0.0,1.0}"
export STOP_AFTER_STAGE1="${STOP_AFTER_STAGE1:-0}"
export VAL_EARLY_STOP="${VAL_EARLY_STOP:-1}"
export VAL_STOP_PATIENCE="${VAL_STOP_PATIENCE:-5}"
export VAL_STOP_MIN_EPOCHS="${VAL_STOP_MIN_EPOCHS:-3}"
export VIS_EVERY="${VIS_EVERY:-5}"
export ALIGN_LAYERS="${ALIGN_LAYERS:-5}"
export BBOX_ALIGN_WEIGHT="${BBOX_ALIGN_WEIGHT:-2.0}"

M3FD_ROOT="${PROJECT_ROOT}/M3FD"
M3FD_ANN="${M3FD_ROOT}/annotations/val.json"
INIT_CKPT="${INIT_CKPT:-${PROJECT_ROOT}/outputs_csma/ckpt/best_stage1.pt}"
OUT_DIR="${PROJECT_ROOT}/outputs_m3fd_finetune"
LOG_FILE="${OUT_DIR}/logs/train.log"

mkdir -p "${OUT_DIR}/logs"

EXTRA=()
[[ "${VAL_EARLY_STOP}" == "1" ]] && EXTRA+=(--val-early-stop --val-every-epoch)
[[ "${VAL_STOP_PATIENCE}" != "0" ]] && EXTRA+=(--val-stop-patience "${VAL_STOP_PATIENCE}")
[[ "${VAL_STOP_MIN_EPOCHS}" != "0" ]] && EXTRA+=(--val-stop-min-epochs "${VAL_STOP_MIN_EPOCHS}")
[[ "${STOP_AFTER_STAGE1}" == "1" ]] && EXTRA+=(--stop-after-stage1)
[[ "${WARMUP_EPOCHS}" != "0" ]] && EXTRA+=(--warmup-epochs "${WARMUP_EPOCHS}")
[[ -f "${INIT_CKPT}" ]] && EXTRA+=(--init-ckpt "${INIT_CKPT}") \
    || echo "[警告] init-ckpt 不存在，将从随机初始化开始: ${INIT_CKPT}"

echo "========================================================"
echo " M3FD fine-tune（两类 person/car，det 主导，无 Hard）"
echo "  INIT_CKPT        : ${INIT_CKPT}"
echo "  PROMPT           : person. car."
echo "  STAGE_WEIGHTS    : ${STAGE_WEIGHTS}"
echo "  VAL_METRIC       : person_car_mean（每 epoch 评测 val）"
echo "  VAL_STOP         : patience=${VAL_STOP_PATIENCE} min_epoch=${VAL_STOP_MIN_EPOCHS}（person/car 单项无提升则停）"
echo "  OUT_DIR          : ${OUT_DIR}"
echo "  开始:              $(date +'%Y-%m-%d %H:%M:%S')"
echo "========================================================"

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n RGBtest \
    python3 -m src.train_csma \
        --dataset      m3fd \
        --data-root    "${M3FD_ROOT}" \
        --m3fd-ann-file "${M3FD_ANN}" \
        --text-prompt  "person. car." \
        --val-metric   person_car_mean \
        --lambda-recon "${LAMBDA_RECON}" \
        --out-dir      "${OUT_DIR}" \
        --epochs       "${EPOCHS}" \
        --batch-size   "${BATCH_SIZE}" \
        --lr           "${LR}" \
        --loss-mode    "${LOSS_MODE}" \
        --stage-weights "${STAGE_WEIGHTS}" \
        --align-layer-indices "${ALIGN_LAYERS}" \
        --bbox-align-weight "${BBOX_ALIGN_WEIGHT}" \
        --gmm-batches  100 \
        --vis-every    "${VIS_EVERY}" \
        --max-steps    "${MAX_STEPS}" \
        "${EXTRA[@]}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "========================================================"
echo " 结束: $(date +'%Y-%m-%d %H:%M:%S')"
echo " 输出: ${OUT_DIR}"
echo "========================================================"
