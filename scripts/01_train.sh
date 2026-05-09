#!/usr/bin/env bash
# scripts/01_train.sh
#
# CSMA 主训练（100 epoch，batch=8，det_only）
#
# 训练 CSMA 适配器使红外图像转换为伪 RGB，通过冻结 Grounding DINO
# 的检测损失端到端监督。无 L_align（因 FLIR_ADAS_v2 缺少帧级配对）。
#
# 用法：
#   bash scripts/01_train.sh
#
# 可选环境变量覆盖：
#   EPOCHS=50 bash scripts/01_train.sh        # 缩短 epoch 数
#   BATCH_SIZE=4 bash scripts/01_train.sh     # 减小 batch（显存不足时）
#   LR=5e-4 bash scripts/01_train.sh          # 调整学习率
#
# 输出：
#   outputs_csma/ckpt/epoch_{N:04d}.pt   每 10 epoch 保存一次
#   outputs_csma/ckpt/csma_last.pt       最后 epoch
#   outputs_csma/logs/train.log          训练日志
#
# 对应 docs/TD.md §3.2 主训练配置

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── 可通过环境变量覆盖的超参数 ──────────────────────────────────────────────
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-3}"

DATA_ROOT="${PROJECT_ROOT}/FLIR_ADAS_v2/images_thermal_train"
OUT_DIR="${PROJECT_ROOT}/outputs_csma"
LOG_FILE="${OUT_DIR}/logs/train.log"

# ── 验证数据目录 ────────────────────────────────────────────────────────────
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[ERROR] 数据目录不存在: ${DATA_ROOT}"
    exit 1
fi
if [[ ! -f "${DATA_ROOT}/coco.json" ]]; then
    echo "[ERROR] 未找到标注文件: ${DATA_ROOT}/coco.json"
    exit 1
fi

mkdir -p "${OUT_DIR}/logs"

echo "========================================================"
echo " CSMA 主训练"
echo " epochs=${EPOCHS}  batch=${BATCH_SIZE}  lr=${LR}"
echo " 数据:  ${DATA_ROOT}"
echo " 输出:  ${OUT_DIR}"
echo " 日志:  ${LOG_FILE}"
echo " 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo ""

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n RGBtest \
    python3 -m src.train_csma \
        --dataset     flir_v2 \
        --data-root   "${DATA_ROOT}" \
        --out-dir     "${OUT_DIR}" \
        --epochs      "${EPOCHS}" \
        --batch-size  "${BATCH_SIZE}" \
        --lr          "${LR}" \
        --loss-mode   det_only \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================================"
echo " 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo " [OK] 主训练完成"
    if ls "${OUT_DIR}/ckpt/"*.pt &>/dev/null; then
        echo " 生成的 checkpoint:"
        ls -lh "${OUT_DIR}/ckpt/"*.pt | tail -5
    fi
    echo ""
    echo " 下一步："
    echo "   bash scripts/02_eval.sh      # mAP 评估"
    echo "   bash scripts/03_infer.sh     # 推理可视化"
else
    echo " [FAIL] 训练失败（exit code: $EXIT_CODE）"
    echo " 查看日志: ${LOG_FILE}"
    exit $EXIT_CODE
fi
echo "========================================================"
