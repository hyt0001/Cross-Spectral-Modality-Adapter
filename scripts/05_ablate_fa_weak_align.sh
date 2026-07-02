#!/usr/bin/env bash
# ============================================================
# FA 消融 (B)：零初始化 + 弱 L_align (λ_align=0.1, λ_det=0.9)
# 判断弱对齐是否能在保留 bus 的同时超过像素 CSMA
#
# 用法：bash scripts/05_ablate_fa_weak_align.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

export EPOCHS="${EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export LR="${LR:-1e-4}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
export MAX_STEPS="${MAX_STEPS:--1}"
export LOSS_MODE="full"
export STAGE_WEIGHTS="${STAGE_WEIGHTS:-0.1,0.9;0.1,0.9;0.1,0.9}"
export BBOX_ALIGN_WEIGHT="${BBOX_ALIGN_WEIGHT:-3.0}"
export STOP_AFTER_STAGE1="${STOP_AFTER_STAGE1:-1}"
export VAL_EARLY_STOP="${VAL_EARLY_STOP:-1}"
export VIS_EVERY="${VIS_EVERY:-5}"

M3FD_ROOT="${PROJECT_ROOT}/M3FD"
M3FD_ANN="${M3FD_ROOT}/annotations/val.json"
OUT_DIR="${PROJECT_ROOT}/outputs_m3fd_fa_ablate_b"
LOG_FILE="${OUT_DIR}/logs/train.log"

mkdir -p "${OUT_DIR}/logs"

EXTRA=()
[[ "${VAL_EARLY_STOP}" == "1" ]] && EXTRA+=(--val-early-stop)
[[ "${STOP_AFTER_STAGE1}" == "1" ]] && EXTRA+=(--stop-after-stage1)
[[ "${WARMUP_EPOCHS}" != "0" ]] && EXTRA+=(--warmup-epochs "${WARMUP_EPOCHS}")

echo "========================================================"
echo " FA 消融 (B): 零初始化 + 弱 L_align (0.1/0.9)"
echo "  OUT_DIR          : ${OUT_DIR}"
echo "  STAGE_WEIGHTS    : ${STAGE_WEIGHTS}"
echo "  fa_zero_init     : true (默认)"
echo "  开始:              $(date +'%Y-%m-%d %H:%M:%S')"
echo "========================================================"

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n RGBtest \
    python3 -m src.train_csma \
        --adapter-mode   feature \
        --dataset        m3fd \
        --data-root      "${M3FD_ROOT}" \
        --m3fd-ann-file  "${M3FD_ANN}" \
        --text-prompt    "person. car. bus. motorcycle. truck. lamp." \
        --lambda-recon   0.0 \
        --align-layer-indices "" \
        --loss-mode      "${LOSS_MODE}" \
        --stage-weights  "${STAGE_WEIGHTS}" \
        --bbox-align-weight "${BBOX_ALIGN_WEIGHT}" \
        --fa-zero-init \
        --out-dir        "${OUT_DIR}" \
        --epochs         "${EPOCHS}" \
        --batch-size     "${BATCH_SIZE}" \
        --lr             "${LR}" \
        --gmm-batches    100 \
        --vis-every      "${VIS_EVERY}" \
        --max-steps      "${MAX_STEPS}" \
        "${EXTRA[@]}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "========================================================"
echo " 结束: $(date +'%Y-%m-%d %H:%M:%S')"
echo " 权重: ${OUT_DIR}/ckpt/best_stage1.pt"
echo "========================================================"
