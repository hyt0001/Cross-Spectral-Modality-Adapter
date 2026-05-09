#!/usr/bin/env bash
# scripts/04_ablation.sh
#
# CSMA 消融实验（对应 docs/TD.md §3.4 消融二）
# 使用 FLIR v1 配对数据集，所有 4 组均可运行（有 RGB 配对数据）
#
# 4 组配置（每组独立 output_dir，30 epoch 快速消融）：
#   组 A: det_only    — 仅检测损失，无 L_align（基准）
#   组 B: align_only  — 仅 L_align，无 L_det
#   组 C: full_random — L_det + L_align（随机掩码，无 CMSS 引导）[需代码支持]
#   组 D: full_gmm    — L_det + L_align（完整 GMM-CMSS 课程，主设置）
#
# 用法：
#   bash scripts/04_ablation.sh
#   EPOCHS=20 bash scripts/04_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DATA_ROOT="${PROJECT_ROOT}/FLIR_License/train"
ABLATION_ROOT="${PROJECT_ROOT}/outputs_ablation"

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[ERROR] 数据目录不存在: ${DATA_ROOT}"
    exit 1
fi

echo "========================================================"
echo " CSMA 消融实验（FLIR v1，epochs=${EPOCHS}，batch=${BATCH_SIZE}）"
echo " 输出: ${ABLATION_ROOT}"
echo " 开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

run_ablation() {
    local name="$1"
    local loss_mode="$2"
    local extra_args="${3:-}"
    local out_dir="${ABLATION_ROOT}/${name}"

    echo ""
    echo "------------------------------------------------------------"
    echo " [消融] 组: ${name}  loss_mode=${loss_mode}"
    echo "        输出: ${out_dir}"
    echo "------------------------------------------------------------"
    mkdir -p "${out_dir}/logs"

    CUDA_VISIBLE_DEVICES=0 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    conda run --no-capture-output -n RGBtest \
        python3 -m src.train_csma \
            --dataset     flir_v1 \
            --data-root   "${DATA_ROOT}" \
            --out-dir     "${out_dir}" \
            --epochs      "${EPOCHS}" \
            --batch-size  "${BATCH_SIZE}" \
            --loss-mode   "${loss_mode}" \
            ${extra_args} \
        2>&1 | tee "${out_dir}/logs/train.log"

    local code=${PIPESTATUS[0]}
    if [[ $code -eq 0 ]]; then
        echo " [OK] ${name} 完成"
    else
        echo " [FAIL] ${name} 失败（exit=$code），继续下一组..."
    fi
}

# 组 A: det_only（基准，无 L_align）
run_ablation "det_only" "det_only"

# 组 B: align_only（仅 L_align，无 L_det）
run_ablation "align_only" "align_only"

# 组 D: full_gmm（完整 GMM-CMSS，主设置）
run_ablation "full_gmm" "full"

# 组 C: full_random（随机掩码，无 CMSS 引导）
# 注意：需要 src/train_csma.py 添加 --mask-mode random 参数支持后解注释
# run_ablation "full_random" "full" "--mask-mode random"

echo ""
echo "========================================================"
echo " 结束: $(date '+%Y-%m-%d %H:%M:%S')"
echo " [OK] 消融实验完成（A/B/D 共 3 组，C 需代码扩展）"
echo ""
echo " 评估各组 mAP："
for name in det_only align_only full_gmm; do
    echo "   CKPT=${ABLATION_ROOT}/${name}/ckpt/csma_last.pt bash scripts/02_eval.sh"
done
echo "========================================================"
