#!/usr/bin/env bash
# scripts/05_ablation_c.sh
#
# Ablation C 系列：GMM-CMSS 策略消融（对应 docs/实验实施细节.md §2）
#
# 3 组配置（每组独立 output_dir，100 epoch 完整训练）：
#   C-1: random_mask      — 随机掩码 + 固定 λ=0.5/0.5（基准，无 CMSS）
#   C-2: fixed_threshold  — 固定 CMSS 阈值 μ₂=0.5 + 固定 λ（验证 CMSS 选择性对齐）
#   C-3: gmm_single_b     — GMM 动态 + 固定 Stage B + 固定 λ（验证 GMM 动态阈值效果）
#
# C-4（完整三阶段 A→B→C）为主实验，结果已有，直接填表。
#
# 用法：
#   bash scripts/05_ablation_c.sh
#   EPOCHS=100 bash scripts/05_ablation_c.sh
#   EPOCHS=2 MAX_STEPS=20 bash scripts/05_ablation_c.sh  # smoke 验证

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_STEPS="${MAX_STEPS:--1}"
DATA_ROOT="${PROJECT_ROOT}/FLIR_License/train"
ABL_ROOT="${PROJECT_ROOT}/outputs_ablation_c"

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[ERROR] 数据目录不存在: ${DATA_ROOT}"
    exit 1
fi

echo "========================================================"
echo " Ablation C: GMM-CMSS 策略消融"
echo " 数据: ${DATA_ROOT}"
echo " epochs=${EPOCHS}  batch=${BATCH_SIZE}  max_steps=${MAX_STEPS}"
echo " 输出根目录: ${ABL_ROOT}"
echo " 开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

run_ablation_c() {
    local name="$1"
    local mode="$2"
    local out_dir="${ABL_ROOT}/${name}"

    echo ""
    echo "------------------------------------------------------------"
    echo " [Ablation C] ${name}  cmss-ablation-mode=${mode}"
    echo "             输出: ${out_dir}"
    echo "------------------------------------------------------------"
    mkdir -p "${out_dir}/logs"

    CUDA_VISIBLE_DEVICES=0 \
    HF_HOME=/root/autodl-tmp/hf_cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    python3 -m src.train_csma \
            --dataset              flir_v1 \
            --data-root            "${DATA_ROOT}" \
            --out-dir              "${out_dir}" \
            --epochs               "${EPOCHS}" \
            --batch-size           "${BATCH_SIZE}" \
            --cmss-ablation-mode   "${mode}" \
            --max-steps            "${MAX_STEPS}" \
        2>&1 | tee "${out_dir}/logs/train.log"

    local code=${PIPESTATUS[0]}
    if [[ $code -eq 0 ]]; then
        echo " [OK] ${name} 完成"
    else
        echo " [FAIL] ${name} 失败（exit=$code）"
        return "${code}"
    fi
}

# C-1: 随机掩码（验证 L_align 基础有效性）
run_ablation_c "c1_random_mask"     "random_mask"

# C-2: 固定 CMSS 阈值（验证 CMSS 感知选择性对齐有效性）
run_ablation_c "c2_fixed_threshold" "fixed_threshold"

# C-3: GMM 动态单阶段 B（验证 GMM 动态阈值优于固定阈值）
run_ablation_c "c3_gmm_single_b"    "gmm_single_b"

echo ""
echo "========================================================"
echo " 结束: $(date '+%Y-%m-%d %H:%M:%S')"
echo " [OK] Ablation C 系列训练完成（C-1 / C-2 / C-3）"
echo ""
echo " 评估各变体 mAP（运行 eval 前确认 ckpt 存在）："
for name in c1_random_mask c2_fixed_threshold c3_gmm_single_b; do
    ckpt="${ABL_ROOT}/${name}/ckpt/csma_last.pt"
    echo "   CUDA_VISIBLE_DEVICES=0 conda run -n RGBtest python -m src.eval_csma \\"
    echo "       --dataset flir_v1 --data-root FLIR_License/val \\"
    echo "       --ckpt ${ckpt}"
done
echo "========================================================"
