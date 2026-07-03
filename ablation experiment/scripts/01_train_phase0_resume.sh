#!/usr/bin/env bash
# scripts/01_train_phase0_resume.sh
#
# 从 checkpoint 续训（需先停止旧训练进程）。
#
# 示例：从 epoch_0030.pt 对应权重继续，从 0-based epoch 31 训到 49
#   INIT_CKPT=outputs_csma/ckpt/epoch_0030.pt START_EPOCH=31 bash scripts/01_train_phase0_resume.sh
#
# 若已有 latest.pt + latest_meta.json（新版 train_csma 每 epoch 写入）：
#   INIT_CKPT=outputs_csma/ckpt/latest.pt START_EPOCH=35 bash scripts/01_train_phase0_resume.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

INIT_CKPT="${INIT_CKPT:?请设置 INIT_CKPT=outputs_csma/ckpt/epoch_XXXX.pt}"
START_EPOCH="${START_EPOCH:?请设置 START_EPOCH=（0-based 下一 epoch 编号）}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_STEPS="${MAX_STEPS:--1}"
LR="${LR:-1e-4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p outputs_csma/logs

echo "[resume] init=${INIT_CKPT} start_epoch=${START_EPOCH} total_epochs=${EPOCHS}"

exec conda run --no-capture-output -n RGBtest \
    python3 -m src.train_csma \
        --dataset flir_v1 \
        --data-root "${PROJECT_ROOT}/FLIR_License/train" \
        --out-dir "${PROJECT_ROOT}/outputs_csma" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --loss-mode full \
        --gmm-batches 100 \
        --max-steps "${MAX_STEPS}" \
        --init-ckpt "${INIT_CKPT}" \
        --start-epoch "${START_EPOCH}" \
    2>&1 | tee -a outputs_csma/logs/train.log
