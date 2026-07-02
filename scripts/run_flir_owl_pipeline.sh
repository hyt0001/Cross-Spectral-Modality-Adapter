#!/usr/bin/env bash
# 串联 FLIR Round-2 → Round-3 → 评测（最佳权重用 best_stage1，非 EMA）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

LOG="${PROJECT_ROOT}/outputs_flir_final/logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${PROJECT_ROOT}/outputs_flir_final/logs"

{
  echo "======== $(date) 开始 FLIR Round-2 checkpoint_b ========"
  bash scripts/01_train_flir_checkpoint_b.sh

  echo "======== $(date) 开始 FLIR Round-3 final ========"
  bash scripts/01_train_flir_final.sh

  echo "======== $(date) Baseline eval T=0.2 ========"
  bash scripts/02_eval_flir_baseline_val.sh

  echo "======== $(date) CSMA Round-2 best eval ========"
  PSEUDO_CLAMP=3.0 RESIDUAL_SCALE=0.1 \
    CKPT="${PROJECT_ROOT}/outputs_flir_ckpt_b/ckpt/best_stage1.pt" \
    bash scripts/02_eval_flir_csma_val.sh

  echo "======== $(date) CSMA Round-3 best eval ========"
  PSEUDO_CLAMP=2.0 RESIDUAL_SCALE=0.05 \
    CKPT="${PROJECT_ROOT}/outputs_flir_final/ckpt/best_stage1.pt" \
    OUT_JSON="${PROJECT_ROOT}/outputs_flir_final/logs/eval_csma_final_best_stage1_owl.json" \
    bash scripts/02_eval_flir_csma_val.sh

  echo "======== $(date) FLIR OWL 管线完成 ========"
} 2>&1 | tee -a "${LOG}"
