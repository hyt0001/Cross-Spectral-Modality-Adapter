#!/usr/bin/env bash
# 串联 M3FD Round-2 → Round-3 → 评测
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

LOG="${PROJECT_ROOT}/outputs_m3fd_final/logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${PROJECT_ROOT}/outputs_m3fd_final/logs"

{
  echo "======== $(date) 开始 Round-2 checkpoint_b ========"
  bash scripts/01_train_m3fd_checkpoint_b.sh

  echo "======== $(date) 开始 Round-3 final ========"
  bash scripts/01_train_m3fd_final.sh

  echo "======== $(date) Baseline eval T=0.2 ========"
  bash scripts/02_eval_m3fd_baseline_val.sh

  echo "======== $(date) CSMA eval (EMA ep1) ========"
  CKPT="${PROJECT_ROOT}/outputs_m3fd_final/ckpt/ema_epoch_0001.pt"
  if [[ ! -f "${CKPT}" ]]; then
    CKPT="${PROJECT_ROOT}/outputs_m3fd_final/ckpt/ema_epoch_0000.pt"
  fi
  CKPT="${CKPT}" bash scripts/02_eval_m3fd_csma_val.sh

  echo "======== $(date) 管线完成 ========"
} 2>&1 | tee -a "${LOG}"
