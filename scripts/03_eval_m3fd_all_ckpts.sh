#!/usr/bin/env bash
# ============================================================
# scripts/03_eval_m3fd_all_ckpts.sh
#
# M3FD fine-tune 完成后，批量评估所有 epoch_XXXX.pt checkpoint。
# 对每个权重调用 eval_csma，汇总 6 类 AP 到一个对比表。
#
# 用法：
#   bash scripts/03_eval_m3fd_all_ckpts.sh
#
# 环境变量（可选覆盖）：
#   OUT_DIR     训练输出目录（默认 outputs_m3fd_finetune）
#   DATA_ROOT   M3FD 数据集根目录（默认 M3FD/）
#   ANN_FILE    COCO 标注（默认 M3FD/annotations/val.json）
#   BATCH_SIZE  评估批大小（默认 4）
#   EXTRA_CKPTS 额外要评估的权重，空格分隔绝对路径
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

# ── 配置 ──────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs_m3fd_finetune}"
CKPT_DIR="${OUT_DIR}/ckpt"
LOG_DIR="${OUT_DIR}/logs"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/M3FD}"
ANN_FILE="${ANN_FILE:-${DATA_ROOT}/annotations/val.json}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EXTRA_CKPTS="${EXTRA_CKPTS:-}"   # 额外指定的 ckpt 绝对路径（空格分隔）

mkdir -p "${LOG_DIR}"

# ── 前置检查 ──────────────────────────────────────────────
if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "[ERROR] ckpt 目录不存在：${CKPT_DIR}"
    echo "  请先运行：bash scripts/01_train_m3fd_finetune.sh"
    exit 1
fi

if [[ ! -f "${ANN_FILE}" ]]; then
    echo "[ERROR] 标注文件不存在：${ANN_FILE}"
    exit 1
fi

# ── 收集待评估的 checkpoint ──────────────────────────────
CKPT_LIST=()

# 1. 所有 epoch_XXXX.pt（按名称升序）
while IFS= read -r -d '' f; do
    CKPT_LIST+=("$f")
done < <(find "${CKPT_DIR}" -maxdepth 1 -name "epoch_*.pt" -print0 | sort -z)

# 2. best_stage1.pt（如果存在）
[[ -f "${CKPT_DIR}/best_stage1.pt" ]] && CKPT_LIST+=("${CKPT_DIR}/best_stage1.pt")

# 3. csma_last.pt（如果存在）
[[ -f "${CKPT_DIR}/csma_last.pt"   ]] && CKPT_LIST+=("${CKPT_DIR}/csma_last.pt")

# 4. EXTRA_CKPTS（用户手动追加）
for extra in ${EXTRA_CKPTS}; do
    [[ -f "${extra}" ]] && CKPT_LIST+=("${extra}")
done

if [[ ${#CKPT_LIST[@]} -eq 0 ]]; then
    echo "[ERROR] ${CKPT_DIR} 内未找到任何 *.pt 文件"
    exit 1
fi

echo "========================================================"
echo " M3FD checkpoint 批量评估"
echo " CKPT_DIR   : ${CKPT_DIR}"
echo " DATA_ROOT  : ${DATA_ROOT}"
echo " 待评估数量  : ${#CKPT_LIST[@]}"
echo " 开始:        $(date +'%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# ── 汇总表头 ──────────────────────────────────────────────
SUMMARY="${LOG_DIR}/eval_m3fd_all_ckpts_summary.txt"
{
echo "M3FD checkpoint 批量评估  $(date -Iseconds)"
printf "%-22s %8s %10s %8s %8s %8s %8s %8s %8s\n" \
    "checkpoint" "mAP@.5" "mAP@.5:.95" "person" "car" "bus" "moto" "truck" "lamp"
printf "%-22s %8s %10s %8s %8s %8s %8s %8s %8s\n" \
    "----------------------" "-------" "----------" "------" "------" "------" "------" "------" "------"
} > "${SUMMARY}"

# ── 逐个评估 ─────────────────────────────────────────────
FAIL_COUNT=0
for CKPT_PATH in "${CKPT_LIST[@]}"; do
    CKPT_NAME="$(basename "${CKPT_PATH}" .pt)"
    OUT_JSON="${LOG_DIR}/eval_m3fd_${CKPT_NAME}.json"

    echo ""
    echo "──── 评估 ${CKPT_NAME} ────────────────────────────────"

    set +e
    CUDA_VISIBLE_DEVICES=0 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    conda run --no-capture-output -n RGBtest \
        python3 -m src.eval_csma \
            --ckpt           "${CKPT_PATH}" \
            --dataset        m3fd \
            --data-root      "${DATA_ROOT}" \
            --ann-file       "${ANN_FILE}" \
            --out-json       "${OUT_JSON}" \
            --batch-size     "${BATCH_SIZE}" \
            --box-threshold  0.05 \
            --text-threshold 0.05 \
            --text-prompt    "person. car. bus. motorcycle. truck. lamp." \
        2>&1
    EXIT_CODE=$?
    set -e

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "[FAIL] ${CKPT_NAME}（exit ${EXIT_CODE}）" | tee -a "${SUMMARY}"
        (( FAIL_COUNT++ )) || true
        continue
    fi

    # 解析结果并追加汇总
    conda run --no-capture-output -n RGBtest python3 - <<PYEOF
import json, os, sys
try:
    with open("${OUT_JSON}") as f:
        r = json.load(f)
except Exception as e:
    print(f"[解析失败] ${OUT_JSON}: {e}")
    sys.exit(0)

line = "{:22s} {:8.4f} {:10.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}".format(
    "${CKPT_NAME}",
    r.get("map_50", 0),
    r.get("map_50_95", 0),
    r.get("ap_person", 0),
    r.get("ap_car", 0),
    r.get("ap_bus", 0),
    r.get("ap_motorcycle", 0),
    r.get("ap_truck", 0),
    r.get("ap_lamp", 0),
)
print("  " + line)
with open("${SUMMARY}", "a") as f:
    f.write(line + "\n")
PYEOF

done  # end for CKPT_PATH

# ── 打印汇总 ──────────────────────────────────────────────
echo ""
echo "========================================================"
echo " 结束: $(date +'%Y-%m-%d %H:%M:%S')"
echo " 失败数: ${FAIL_COUNT}"
echo " 汇总表: ${SUMMARY}"
echo "========================================================"
echo ""
cat "${SUMMARY}"
