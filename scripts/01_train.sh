#!/usr/bin/env bash
# scripts/01_train.sh
#
# CSMA 主训练（FLIR v1 配对数据集，100 epoch，loss_mode=full）
#
# 使用 FLIR ADAS v1 (FLIR_License) 进行完整训练：
#   - L_det：冻结 Grounding DINO 的检测损失
#   - L_align：GMM-CMSS 课程引导的跨模态特征蒸馏
#   - 三阶段课程：Stage-A (Easy) → Stage-B (Mixed) → Stage-C (Hard)
#
# 用法：
#   bash scripts/01_train.sh
#   EPOCHS=50 bash scripts/01_train.sh
#   BATCH_SIZE=4 bash scripts/01_train.sh
#   LOSS_MODE=det_only bash scripts/01_train.sh   # 仅 det loss（消融基准）
#
# 输出：
#   outputs_csma/ckpt/epoch_{N:04d}.pt   每 10 epoch 保存一次
#   outputs_csma/ckpt/csma_last.pt       最后 epoch
#   outputs_csma/logs/train.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

EPOCHS="${EPOCHS:-35}"
HARD_MAX_EPOCHS="${HARD_MAX_EPOCHS:-}"
# img_size=512 下 8GB GPU 安全 batch=2（AMP fp16）；内存充裕时可用 BATCH_SIZE=4 覆盖
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-4}"
# max_steps=-1 表示全量（3928步/epoch ≈ 4.4h/epoch）
# 建议 200（400 张/epoch，30epoch≈13h）；可通过 MAX_STEPS=200 覆盖
MAX_STEPS="${MAX_STEPS:--1}"
LOSS_MODE="${LOSS_MODE:-full}"

DATA_ROOT="${PROJECT_ROOT}/FLIR_License/train"
OUT_DIR="${PROJECT_ROOT}/outputs_csma"
LOG_FILE="${OUT_DIR}/logs/train.log"

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[ERROR] 数据目录不存在: ${DATA_ROOT}"
    exit 1
fi
if [[ ! -f "${DATA_ROOT}/thermal_annotations.json" ]]; then
    echo "[ERROR] 未找到标注文件: ${DATA_ROOT}/thermal_annotations.json"
    exit 1
fi

mkdir -p "${OUT_DIR}/logs"

echo "========================================================"
echo " CSMA 主训练（FLIR v1 配对数据集）"
echo " loss_mode=${LOSS_MODE}  epochs=${EPOCHS}  batch=${BATCH_SIZE}  lr=${LR}  max_steps=${MAX_STEPS}  hard_max=${HARD_MAX_EPOCHS:-auto}"
echo " 数据:  ${DATA_ROOT}"
echo " 输出:  ${OUT_DIR}"
echo " 开始:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n RGBtest \
    python3 -m src.train_csma \
        --dataset     flir_v1 \
        --data-root   "${DATA_ROOT}" \
        --out-dir     "${OUT_DIR}" \
        --epochs      "${EPOCHS}" \
        --batch-size  "${BATCH_SIZE}" \
        --lr          "${LR}" \
        --loss-mode   "${LOSS_MODE}" \
        --gmm-batches 100 \
        ${MAX_STEPS:+--max-steps "${MAX_STEPS}"} \
        ${HARD_MAX_EPOCHS:+--hard-max-epochs "${HARD_MAX_EPOCHS}"} \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================================"
echo " 结束: $(date '+%Y-%m-%d %H:%M:%S')"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo " [OK] 主训练完成"
    ls -lh "${OUT_DIR}/ckpt/"*.pt 2>/dev/null | tail -5
    echo ""
    echo " 下一步："
    echo "   bash scripts/02_eval.sh      # mAP 评估（val 集）"
    echo "   bash scripts/03_infer.sh     # 推理可视化"
else
    echo " [FAIL] 训练失败（exit code: $EXIT_CODE）"
    echo " 查看日志: ${LOG_FILE}"
    exit $EXIT_CODE
fi
echo "========================================================"
