#!/usr/bin/env bash
# scripts/06_ablation_b2.sh
#
# Ablation B-2: 均值原型（Frozen Mean Prototype）训练脚本
# 对应 docs/实验实施细节.md §1.3（B-2 运行顺序）
#
# 两步执行：
#   Step 1: 离线预计算训练集 RGB 特征均值原型（compute_mean_proto.py）
#   Step 2: 用 CSMAMeanProto 变体训练 CSMA（--variant mean_proto）
#
# 与主实验（B-3 可学习原型）的唯一差别：
#   RPCA 的 K/V 来自固定 buffer（离线均值），而非训练反传的 nn.Parameter。
#
# 用法：
#   bash scripts/06_ablation_b2.sh
#   EPOCHS=100 bash scripts/06_ablation_b2.sh
#   EPOCHS=2 MAX_STEPS=20 bash scripts/06_ablation_b2.sh  # smoke 验证

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_STEPS="${MAX_STEPS:--1}"
DATA_ROOT="${PROJECT_ROOT}/FLIR_License/train"
OUT_DIR="${PROJECT_ROOT}/outputs_abl_b2_mean_proto"
MEAN_PROTO_PATH="${PROJECT_ROOT}/outputs_csma/mean_proto.pt"

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[ERROR] 数据目录不存在: ${DATA_ROOT}"
    exit 1
fi

echo "========================================================"
echo " Ablation B-2: 均值原型（Frozen Mean Prototype）"
echo " 数据: ${DATA_ROOT}"
echo " epochs=${EPOCHS}  batch=${BATCH_SIZE}  max_steps=${MAX_STEPS}"
echo " 输出: ${OUT_DIR}"
echo " 均值原型: ${MEAN_PROTO_PATH}"
echo " 开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# ── Step 1: 离线预计算均值原型 ──────────────────────────────────────────────
# 若均值原型文件已存在则跳过（避免重复计算）
if [[ -f "${MEAN_PROTO_PATH}" ]]; then
    echo ""
    echo "[Step 1] 均值原型已存在，跳过预计算: ${MEAN_PROTO_PATH}"
else
    echo ""
    echo "------------------------------------------------------------"
    echo " [Step 1] 预计算训练集 RGB 特征均值原型..."
    echo "------------------------------------------------------------"
    mkdir -p "$(dirname "${MEAN_PROTO_PATH}")"

    CUDA_VISIBLE_DEVICES=0 \
    HF_HOME=/root/autodl-tmp/hf_cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    python3 -m src.compute_mean_proto \
            --data-root "${DATA_ROOT}" \
            --out       "${MEAN_PROTO_PATH}" \
            --batch-size 8

    if [[ $? -ne 0 ]]; then
        echo "[ERROR] compute_mean_proto.py 失败，退出"
        exit 1
    fi
    echo "[Step 1] 完成: ${MEAN_PROTO_PATH}"
fi

# ── Step 2: 训练 Ablation B-2 ───────────────────────────────────────────────
echo ""
echo "------------------------------------------------------------"
echo " [Step 2] 训练 CSMAMeanProto（均值原型变体，${EPOCHS} epoch）..."
echo "------------------------------------------------------------"
mkdir -p "${OUT_DIR}/logs"

CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/root/autodl-tmp/hf_cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python3 -m src.train_csma \
        --dataset         flir_v1 \
        --data-root       "${DATA_ROOT}" \
        --out-dir         "${OUT_DIR}" \
        --epochs          "${EPOCHS}" \
        --batch-size      "${BATCH_SIZE}" \
        --variant         mean_proto \
        --mean-proto-path "${MEAN_PROTO_PATH}" \
        --max-steps       "${MAX_STEPS}" \
    2>&1 | tee "${OUT_DIR}/logs/train.log"

code=${PIPESTATUS[0]}
echo ""
echo "========================================================"
echo " 结束: $(date '+%Y-%m-%d %H:%M:%S')"
if [[ $code -eq 0 ]]; then
    echo " [OK] Ablation B-2 训练完成"
    echo ""
    echo " 评估 mAP："
    echo "   CUDA_VISIBLE_DEVICES=0 HF_HOME=/root/autodl-tmp/hf_cache \\"
    echo "   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\"
    echo "   python3 -m src.eval_csma \\"
    echo "       --dataset flir_v1 --data-root FLIR_License/val \\"
    echo "       --ckpt ${OUT_DIR}/ckpt/csma_last.pt"
else
    echo " [FAIL] Ablation B-2 训练失败（exit=${code}）"
fi
echo "========================================================"
exit "${code}"
