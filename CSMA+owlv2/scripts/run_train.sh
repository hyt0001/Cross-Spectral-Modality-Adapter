#!/bin/bash
# 从头训练 CSMA（不加载 init-ckpt）
# 与首轮训练 outputs_rgb_only_gmm1_warm 相同，仅把 pseudo 相关超参换成 Final Model 配置。
#
# 数据：
#   训练集  /root/autodl-tmp/train   （FLIR train）
#   验证集  /root/autodl-tmp/val      （FLIR val）
#
# 验证：
#   --val-every 1  → 每个 epoch 训练结束后，在 FLIR val 上跑一遍验证
#   指标写入 ${OUT_DIR}/logs/val_metrics.jsonl
#   用 EMA 权重验证；AP50 创新高时保存 best.pt / best_ap50_epXX.pt
#
# 相对 gmm1_warm 首轮，唯一改动（Final Model pseudo 配置）：
#   id_loss_weight   0.05  → 0.005
#   tv_loss_weight   0.01  → 0.05
#   logit_reg_weight 0.01  → 0.02
#   pseudo_clamp     3.0   → 2.0
#   residual_scale   0.1   → 0.05
#
# 其余保持 CSMAConfig 默认 + 首轮训练惯例：
#   lr=1e-4, batch-size=4, 三阶段 la/ld, gmm-update-every=1, ema-decay=0.999

set -euo pipefail
cd "$(dirname "$0")/.."

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

OUT_DIR="${OUT_DIR:-outputs_final_config_from_scratch}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"

# 若要在 CPU 上跑，执行前加：USE_CPU=1
if [[ "${USE_CPU:-0}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  BATCH_SIZE="${BATCH_SIZE:-1}"
  NUM_WORKERS="${NUM_WORKERS:-0}"
fi

echo "=== CSMA from scratch (gmm1_warm schedule + Final Model pseudo config) ==="
echo "train: /root/autodl-tmp/train  |  val: /root/autodl-tmp/val  |  val-every: 1"
echo "out-dir=${OUT_DIR}  epochs=${EPOCHS}  batch-size=${BATCH_SIZE}  lr=${LR}"
echo "init-ckpt: none"
echo

python -m src.train_csma \
  --data-root /root/autodl-tmp/train \
  --val-root  /root/autodl-tmp/val \
  --out-dir   "${OUT_DIR}" \
  --epochs    "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --lr        "${LR}" \
  --gmm-update-every 1 \
  --id-loss-weight 0.005 \
  --tv-loss-weight 0.05 \
  --pseudo-clamp 2.0 \
  --residual-scale 0.05 \
  --logit-reg-weight 0.02 \
  --val-every 1 \
  --patience 5 \
  --val-threshold 0.2 \
  --ema-decay 0.999

echo
echo "=== 训练完成 ==="
echo "checkpoint: ${OUT_DIR}/ckpt/"
echo "val log:    ${OUT_DIR}/logs/val_metrics.jsonl"
