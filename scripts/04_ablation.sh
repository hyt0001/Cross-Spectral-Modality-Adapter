#!/usr/bin/env bash
# scripts/04_ablation.sh
#
# CSMA 消融实验（对应 docs/TD.md §3.4 消融二）
#
# 4 组配置（每组独立 output_dir，30 epoch 快速消融）：
#
#   组 A: det_only        — 仅检测损失，无 L_align（当前主设置，基准）
#   组 B: align_only      — 仅 L_align，无 L_det
#                           [注意] 需要配对数据，当前跳过
#   组 C: full_random     — L_det + L_align（随机掩码，无 CMSS 引导）
#                           [注意] 需要配对数据，当前跳过
#   组 D: full_gmm        — L_det + L_align（完整 GMM-CMSS 课程）
#                           [注意] 需要配对数据，当前跳过
#
# FLIR_ADAS_v2 数据集 thermal/RGB video ID 不匹配，当前只能跑 det_only（组 A）。
# 组 B/C/D 以注释形式保留命令，等配对数据就绪后手动解注释执行。
#
# 用法：
#   bash scripts/04_ablation.sh
#   EPOCHS=20 bash scripts/04_ablation.sh      # 缩短消融轮次
#
# 输出（每组独立目录）：
#   outputs_ablation/det_only/
#   outputs_ablation/align_only/    （等待配对数据）
#   outputs_ablation/full_random/   （等待配对数据）
#   outputs_ablation/full_gmm/      （等待配对数据）
#
# 对应 docs/TD.md §3.4 消融二

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DATA_THERMAL="${PROJECT_ROOT}/FLIR_ADAS_v2/images_thermal_train"
# DATA_PAIRED="${PROJECT_ROOT}/FLIR_ADAS_v2/..."   # 配对数据目录（待确认）
ABLATION_ROOT="${PROJECT_ROOT}/outputs_ablation"

# ── 验证数据目录 ────────────────────────────────────────────────────────────
if [[ ! -d "${DATA_THERMAL}" ]]; then
    echo "[ERROR] 数据目录不存在: ${DATA_THERMAL}"
    exit 1
fi

echo "========================================================"
echo " CSMA 消融实验"
echo " epochs=${EPOCHS}  batch=${BATCH_SIZE}"
echo " 数据: ${DATA_THERMAL}"
echo " 输出: ${ABLATION_ROOT}"
echo " 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo ""

# ── 组 A: det_only（基准，当前可运行）──────────────────────────────────────
run_ablation() {
    local name="$1"
    local loss_mode="$2"
    local data_root="$3"
    local extra_args="${4:-}"
    local out_dir="${ABLATION_ROOT}/${name}"

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
            --dataset     flir_v2 \
            --data-root   "${data_root}" \
            --out-dir     "${out_dir}" \
            --epochs      "${EPOCHS}" \
            --batch-size  "${BATCH_SIZE}" \
            --loss-mode   "${loss_mode}" \
            ${extra_args} \
        2>&1 | tee "${out_dir}/logs/train.log"

    local code=${PIPESTATUS[0]}
    if [[ $code -eq 0 ]]; then
        echo " [OK] ${name} 消融完成"
    else
        echo " [FAIL] ${name} 消融失败（exit code: $code）"
        return $code
    fi
    echo ""
}

# 组 A: det_only（基准）——当前唯一可运行配置
run_ablation "det_only" "det_only" "${DATA_THERMAL}"

# ── 组 B/C/D：需要 RGB-IR 配对数据，当前跳过 ─────────────────────────────
# 解注释前请确认配对数据目录结构和数据集适配器支持配对模式。
#
# 组 B: align_only（仅 L_align）
# run_ablation "align_only" "align_only" "${DATA_PAIRED}"
#
# 组 C: full_random（L_det + L_align，随机掩码）
# 需在 src/train_csma.py 中添加 --mask-mode random 参数支持
# run_ablation "full_random" "full" "${DATA_PAIRED}" "--mask-mode random"
#
# 组 D: full_gmm（完整 GMM-CMSS 课程）
# run_ablation "full_gmm" "full" "${DATA_PAIRED}"

echo "========================================================"
echo " 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " [OK] 消融实验完成（组 A: det_only）"
echo ""
echo " 当前结果目录: ${ABLATION_ROOT}/det_only/"
echo ""
echo " 剩余消融组（B/C/D）等待 RGB-IR 配对数据就绪后执行。"
echo " 查看详情：cat ${ABLATION_ROOT}/det_only/logs/train.log"
echo ""
echo " 对比各组 mAP（需先执行各组评估）："
echo "   CKPT=${ABLATION_ROOT}/det_only/ckpt/csma_last.pt bash scripts/02_eval.sh"
echo "========================================================"
