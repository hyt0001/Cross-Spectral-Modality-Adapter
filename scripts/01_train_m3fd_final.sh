#!/usr/bin/env bash
# ============================================================
# M3FD Final Model（OWL Round-3）：短训 2 epoch + pseudo 正则调整
#
# 对齐 CSMA Final Model 配置说明 §3：
#   id_loss ÷10, tv_loss ×5, logit_reg ×2, clamp/residual 收紧
#
# 用法：bash scripts/01_train_m3fd_final.sh
# 评测 EMA：CKPT=outputs_m3fd_final/ckpt/ema_epoch_0001.pt bash scripts/02_eval_m3fd_csma_val.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

export EPOCHS="${EPOCHS:-2}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export LR="${LR:-1e-5}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
export LAMBDA_RECON="${LAMBDA_RECON:-0.0}"
export LAMBDA_ID="${LAMBDA_ID:-0.005}"
export LAMBDA_TV="${LAMBDA_TV:-0.05}"
export LAMBDA_LOGIT_REG="${LAMBDA_LOGIT_REG:-0.02}"
export PSEUDO_CLAMP="${PSEUDO_CLAMP:-2.0}"
export RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.05}"
export EMA_DECAY="${EMA_DECAY:-0.999}"
export STAGE_WEIGHTS="${STAGE_WEIGHTS:-0.0,1.0;0.0,1.0;0.0,1.0}"
export VAL_BOX_THR="${VAL_BOX_THR:-0.2}"
export VAL_TEXT_THR="${VAL_TEXT_THR:-0.2}"

M3FD_ROOT="${PROJECT_ROOT}/M3FD"
M3FD_ANN="${M3FD_ROOT}/annotations/val.json"
INIT_CKPT="${INIT_CKPT:-${PROJECT_ROOT}/outputs_m3fd_ckpt_b/ckpt/best_stage1.pt}"
OUT_DIR="${PROJECT_ROOT}/outputs_m3fd_final"

mkdir -p "${OUT_DIR}/logs"

if [[ ! -f "${INIT_CKPT}" ]]; then
    echo "[WARN] Checkpoint-B 不存在，回退到 outputs_m3fd_finetune/ckpt/best_stage1.pt"
    INIT_CKPT="${PROJECT_ROOT}/outputs_m3fd_finetune/ckpt/best_stage1.pt"
fi

echo "========================================================"
echo " M3FD Final Model（OWL Round-3，${EPOCHS} epoch）"
echo "  INIT: ${INIT_CKPT}"
echo "  OUT:  ${OUT_DIR}"
echo "========================================================"

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n RGBtest \
    python3 -m src.train_csma \
        --dataset m3fd \
        --data-root "${M3FD_ROOT}" \
        --m3fd-ann-file "${M3FD_ANN}" \
        --text-prompt "person. car." \
        --val-metric person_car_mean \
        --init-ckpt "${INIT_CKPT}" \
        --out-dir "${OUT_DIR}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --warmup-epochs "${WARMUP_EPOCHS}" \
        --lambda-recon "${LAMBDA_RECON}" \
        --lambda-id "${LAMBDA_ID}" \
        --lambda-tv "${LAMBDA_TV}" \
        --lambda-logit-reg "${LAMBDA_LOGIT_REG}" \
        --pseudo-clamp "${PSEUDO_CLAMP}" \
        --residual-scale "${RESIDUAL_SCALE}" \
        --ema-decay "${EMA_DECAY}" \
        --stage-weights "${STAGE_WEIGHTS}" \
        --val-early-stop \
        --val-every-epoch \
        --val-box-threshold "${VAL_BOX_THR}" \
        --val-text-threshold "${VAL_TEXT_THR}" \
        --align-layer-indices "1,3,5" \
        --bbox-align-weight 2.0 \
        --gmm-batches 100 \
        --vis-every 1 \
    2>&1 | tee -a "${OUT_DIR}/logs/train.log"
