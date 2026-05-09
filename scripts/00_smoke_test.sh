#!/usr/bin/env bash
# scripts/00_smoke_test.sh
#
# CSMA pipeline 快速验证（2 epoch，batch=2）
# 目标：确认梯度流正常、loss 下降、checkpoint 生成
#
# 用法：
#   bash scripts/00_smoke_test.sh
#
# 约束：
#   - 使用 conda 环境 RGBtest
#   - CUDA_VISIBLE_DEVICES=0（单卡）
#   - HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1（离线环境）
#   - 对应 CSMA 实验训练计划 §scripts/00_smoke_test.sh

set -euo pipefail

# ── 路径配置 ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATA_ROOT="${PROJECT_ROOT}/FLIR_ADAS_v2/images_thermal_train"
OUT_DIR="${PROJECT_ROOT}/outputs_csma_smoke"
LOG_FILE="${OUT_DIR}/smoke_test.log"

# ── 验证数据目录 ────────────────────────────────────────────────────────────
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[ERROR] 数据目录不存在: ${DATA_ROOT}"
    exit 1
fi
if [[ ! -f "${DATA_ROOT}/coco.json" ]]; then
    echo "[ERROR] 未找到标注文件: ${DATA_ROOT}/coco.json"
    exit 1
fi

mkdir -p "${OUT_DIR}"

echo "========================================================"
echo " CSMA Smoke Test（2 epoch, batch=2）"
echo " 数据: ${DATA_ROOT}"
echo " 输出: ${OUT_DIR}"
echo " 日志: ${LOG_FILE}"
echo "========================================================"
echo ""

# ── 执行训练 ─────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n RGBtest \
    python3 -m src.train_csma \
        --dataset     flir_v2 \
        --data-root   "${DATA_ROOT}" \
        --out-dir     "${OUT_DIR}" \
        --epochs      2 \
        --batch-size  2 \
        --loss-mode   det_only \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo " [OK] Smoke test 通过"
    # 验证 checkpoint 是否生成
    if ls "${OUT_DIR}/ckpt/"*.pt &>/dev/null; then
        echo " [OK] Checkpoint 已生成:"
        ls -lh "${OUT_DIR}/ckpt/"*.pt
    else
        echo " [WARN] 未找到 checkpoint 文件，请检查 --out-dir 配置"
    fi
    echo ""
    echo " 可继续执行主训练："
    echo "   bash scripts/01_train.sh"
else
    echo " [FAIL] Smoke test 失败（exit code: $EXIT_CODE）"
    echo " 查看日志: ${LOG_FILE}"
    exit $EXIT_CODE
fi
echo "========================================================"
