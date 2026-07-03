#!/usr/bin/env bash
# ============================================================
# phase0 优化训练：解决「越训越差」问题
# ============================================================
# 根因：
#   1. 损失权重阶段跳变（×5 / ×10）破坏伪 RGB 分布
#   2. det_loss 从不收敛（维持 ~760），Hard 阶段把伪 RGB 拉歪
#   3. 初始 lr 过大，第 1 轮就破坏了良好初始化
#
# 改进：
#   - WARMUP_EPOCHS=3：前 3 轮 lr 从 1e-5 → 1e-4，保护初始化
#   - STAGE_WEIGHTS：三阶段权重从「align_only → 轻 det → 均衡」
#     而非原来的「均衡 → 重 det」，det 充分学好对齐再介入
#   - STOP_AFTER_STAGE1=1：跳过 Hard（λ_det=1.0 会让 mAP 崩）
#   - VAL_EARLY_STOP=1：Mixed 末段自动选最佳 ckpt → best_stage1.pt
#
# 用法：
#   bash scripts/01_train_phase0_early_stop.sh
#
# 环境变量（可选覆盖）：
#   EPOCHS=30              总轮数
#   WARMUP_EPOCHS=3        lr warmup 轮数（0 = 关闭）
#   STAGE_WEIGHTS=...      三阶段权重 'a0,d0;a1,d1;a2,d2'
#   HARD_MAX_EPOCHS=0      Hard 阶段最多 N 轮（0 表示跳过）
#   STOP_AFTER_STAGE1=1    Mixed 结束后停止
#   VAL_EARLY_STOP=1       val 早停并保存 best_stage1.pt
#   VAL_MANUAL=1           手动指定 VAL_START/VAL_END
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

export EPOCHS="${EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export MAX_STEPS="${MAX_STEPS:--1}"
export LR="${LR:-1e-4}"
export LOSS_MODE="${LOSS_MODE:-full}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
# 三阶段损失权重：Stage0 以对齐为主 → Stage1 轻引入 det → Stage2 均衡
# 与原来 (1.0,0.1)→(0.5,0.5)→(0.1,1.0) 的关键区别：
#   - Stage0 保持 λ_det=0.1（最小保底，确保梯度路径通畅；l_align 量级极小，纯0时梯度断裂）
#   - Stage1 轻量引入 det（λ_det=0.2），验证对齐已稳定再施压
#   - Stage2 均衡（λ_det=0.5），而非原来的极端 1.0
# 注意：train_csma.py 会自动将 λ_det clamp 到 >=0.05，防止梯度断裂
export STAGE_WEIGHTS="${STAGE_WEIGHTS:-1.0,0.1;0.8,0.2;0.5,0.5}"
export HARD_MAX_EPOCHS="${HARD_MAX_EPOCHS:-0}"
export STOP_AFTER_STAGE1="${STOP_AFTER_STAGE1:-1}"
export VAL_EARLY_STOP="${VAL_EARLY_STOP:-1}"

DATA_ROOT="${PROJECT_ROOT}/FLIR_License/train"
VAL_ROOT="${PROJECT_ROOT}/FLIR_License/val"
OUT_DIR="${PROJECT_ROOT}/outputs_csma"
LOG_FILE="${OUT_DIR}/logs/train.log"

mkdir -p "${OUT_DIR}/logs"

EXTRA=()
[[ "${VAL_EARLY_STOP}" == "1" ]] && EXTRA+=(--val-early-stop --val-data-root "${VAL_ROOT}")
[[ "${VAL_MANUAL:-0}" == "1" ]] && EXTRA+=(--val-manual)
[[ "${STOP_AFTER_STAGE1}" == "1" ]] && EXTRA+=(--stop-after-stage1)
[[ "${WARMUP_EPOCHS}" != "0" ]] && EXTRA+=(--warmup-epochs "${WARMUP_EPOCHS}")
[[ "${HARD_MAX_EPOCHS}" != "0" ]] && EXTRA+=(--hard-max-epochs "${HARD_MAX_EPOCHS}")

echo "================================================================"
echo "[phase0-optimized] 开始优化训练"
echo "  EPOCHS=${EPOCHS}  WARMUP=${WARMUP_EPOCHS}  LR=${LR}"
echo "  STAGE_WEIGHTS=${STAGE_WEIGHTS}"
echo "  STOP_AFTER_STAGE1=${STOP_AFTER_STAGE1}  VAL_EARLY_STOP=${VAL_EARLY_STOP}"
echo "================================================================"

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n RGBtest \
    python3 -m src.train_csma \
        --dataset flir_v1 \
        --data-root "${DATA_ROOT}" \
        --out-dir "${OUT_DIR}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --loss-mode "${LOSS_MODE}" \
        --gmm-batches 100 \
        --max-steps "${MAX_STEPS}" \
        --stage-weights "${STAGE_WEIGHTS}" \
        "${EXTRA[@]}" \
    2>&1 | tee -a "${LOG_FILE}"
