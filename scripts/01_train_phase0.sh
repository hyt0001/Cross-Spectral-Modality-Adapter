#!/usr/bin/env bash
# scripts/01_train_phase0.sh
#
# 阶段 0：FLIR 全量微调（无单独预训练）
#   - MAX_STEPS=-1  每 epoch 遍历全部训练集（~3928 step @ batch=2）
#   - EPOCHS=50     loss_mode=full（L_det + L_align + GMM-CMSS）
#   - BATCH_SIZE=2  适配 8GB GPU（4060 等）
#
# 后台运行示例：
#   nohup bash scripts/01_train_phase0.sh > outputs_csma/logs/phase0_nohup.log 2>&1 &
#   echo $! > outputs_csma/logs/phase0_train.pid
#
# 监控：
#   tail -f outputs_csma/logs/train.log
#   tail -f outputs_csma/logs/phase0_nohup.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

export EPOCHS="${EPOCHS:-35}"
export HARD_MAX_EPOCHS="${HARD_MAX_EPOCHS:-5}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export MAX_STEPS="${MAX_STEPS:--1}"
export LR="${LR:-1e-4}"
export LOSS_MODE="${LOSS_MODE:-full}"

mkdir -p outputs_csma/logs

echo "[phase0] 启动参数: EPOCHS=${EPOCHS} HARD_MAX=${HARD_MAX_EPOCHS} BATCH_SIZE=${BATCH_SIZE} MAX_STEPS=${MAX_STEPS} LOSS_MODE=${LOSS_MODE}"
echo "[phase0] 推荐 val 早停+跳过 Hard: bash scripts/01_train_phase0_early_stop.sh"
echo "[phase0] 日志: outputs_csma/logs/train.log"
echo "[phase0] PID: $$  时间: $(date '+%Y-%m-%d %H:%M:%S')"

exec bash scripts/01_train.sh
