#!/usr/bin/env bash
# ============================================================
# FLIR Round-2（OWL「起点 best」）：从 Phase0 best 继续微调
#
# 对齐 M3FD OWL Round-2 / CSMA Final Model 配置说明 §1 第二轮：
#   - init: outputs_csma/ckpt/best_stage1.pt（Phase0 最佳 ~59.3% @ T=0.05）
#   - lr=1e-5，λ_id/λ_tv/logit + pseudo clamp/residual
#   - val T=0.2，早停 metric=person_car_mean
#
# 用法：bash scripts/01_train_flir_checkpoint_b.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

export EPOCHS="${EPOCHS:-20}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export LR="${LR:-1e-5}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
export LAMBDA_RECON="${LAMBDA_RECON:-0.0}"
export LAMBDA_ID="${LAMBDA_ID:-0.05}"
export LAMBDA_TV="${LAMBDA_TV:-0.01}"
export LAMBDA_LOGIT_REG="${LAMBDA_LOGIT_REG:-0.01}"
export PSEUDO_CLAMP="${PSEUDO_CLAMP:-3.0}"
export RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.1}"
export EMA_DECAY="${EMA_DECAY:-0.999}"
export STAGE_WEIGHTS="${STAGE_WEIGHTS:-0.3,0.7;0.1,0.9;0.0,1.0}"
export VAL_BOX_THR="${VAL_BOX_THR:-0.2}"
export VAL_TEXT_THR="${VAL_TEXT_THR:-0.2}"

DATA_ROOT="${PROJECT_ROOT}/FLIR_License/train"
VAL_ROOT="${PROJECT_ROOT}/FLIR_License/val"
INIT_CKPT="${INIT_CKPT:-${PROJECT_ROOT}/outputs_csma/ckpt/best_stage1.pt}"
OUT_DIR="${PROJECT_ROOT}/outputs_flir_ckpt_b"

mkdir -p "${OUT_DIR}/logs"

if [[ ! -f "${INIT_CKPT}" ]]; then
    echo "[ERROR] 初始化权重不存在: ${INIT_CKPT}"
    exit 1
fi

echo "========================================================"
echo " FLIR Checkpoint-B 训练（OWL Round-2）"
echo "  INIT: ${INIT_CKPT}"
echo "  OUT:  ${OUT_DIR}"
echo "========================================================"

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n RGBtest \
    python3 -m src.train_csma \
        --dataset flir_v1 \
        --data-root "${DATA_ROOT}" \
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
        --val-data-root "${VAL_ROOT}" \
        --val-every-epoch \
        --val-box-threshold "${VAL_BOX_THR}" \
        --val-text-threshold "${VAL_TEXT_THR}" \
        --stop-after-stage1 \
        --align-layer-indices "1,3,5" \
        --bbox-align-weight 2.0 \
        --gmm-batches 100 \
        --vis-every 5 \
    2>&1 | tee -a "${OUT_DIR}/logs/train.log"
