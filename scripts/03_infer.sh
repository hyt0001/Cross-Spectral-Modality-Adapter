#!/usr/bin/env bash
# scripts/03_infer.sh
#
# CSMA 推理可视化（thermal_val 抽样 5 张，生成三联对比图）
#
# 输出图像布局（每行 3 列）：
#   第 1 列：原始 IR 图像（输入）
#   第 2 列：CSMA 输出伪 RGB
#   第 3 列：伪 RGB + 预测框（红）+ GT 框（绿）
#
# 用法：
#   bash scripts/03_infer.sh                                     # 使用 csma_last.pt
#   CKPT=outputs_csma/ckpt/epoch_0050.pt bash scripts/03_infer.sh
#   NUM_SAMPLES=10 bash scripts/03_infer.sh                      # 可视化 10 张
#
# 输出：
#   outputs_csma/vis/infer_grid_{ckpt_name}.png   主三联对比图
#   outputs_csma/vis/infer_grid_final.png         最新结果的符号链接
#
# 对应 docs/TD.md §3.3 推理可视化

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── 可覆盖的配置 ────────────────────────────────────────────────────────────
CKPT="${CKPT:-${PROJECT_ROOT}/outputs_csma/ckpt/csma_last.pt}"
DATA_ROOT="${PROJECT_ROOT}/FLIR_ADAS_v2/images_thermal_val"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.3}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"

CKPT_NAME="$(basename "${CKPT}" .pt)"
OUT_PNG="${PROJECT_ROOT}/outputs_csma/vis/infer_grid_${CKPT_NAME}.png"
FINAL_LINK="${PROJECT_ROOT}/outputs_csma/vis/infer_grid_final.png"

# ── 验证 ────────────────────────────────────────────────────────────────────
if [[ ! -f "${CKPT}" ]]; then
    echo "[ERROR] Checkpoint 不存在: ${CKPT}"
    echo "  请先运行主训练：bash scripts/01_train.sh"
    exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[ERROR] 验证集目录不存在: ${DATA_ROOT}"
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/outputs_csma/vis"

echo "========================================================"
echo " CSMA 推理可视化"
echo " checkpoint:  ${CKPT}"
echo " 验证集:      ${DATA_ROOT}"
echo " 样本数:      ${NUM_SAMPLES}"
echo " 输出 PNG:    ${OUT_PNG}"
echo " 开始时间:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo ""

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n RGBtest \
    python3 -m src.infer_csma \
        --ckpt           "${CKPT}" \
        --dataset        flir_v2 \
        --data-root      "${DATA_ROOT}" \
        --num-samples    "${NUM_SAMPLES}" \
        --out            "${OUT_PNG}" \
        --box-threshold  "${BOX_THRESHOLD}" \
        --text-threshold "${TEXT_THRESHOLD}" \
    2>&1

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================================"
echo " 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo " [OK] 推理完成，对比图已保存: ${OUT_PNG}"
    # 建立/更新 final 符号链接
    ln -sf "${OUT_PNG}" "${FINAL_LINK}"
    echo " 符号链接已更新: ${FINAL_LINK} → ${OUT_PNG}"
    if command -v ls &>/dev/null; then
        ls -lh "${OUT_PNG}"
    fi
else
    echo " [FAIL] 推理失败（exit code: $EXIT_CODE）"
    exit $EXIT_CODE
fi
echo "========================================================"
