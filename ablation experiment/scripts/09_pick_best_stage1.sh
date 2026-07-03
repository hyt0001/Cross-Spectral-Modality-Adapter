#!/usr/bin/env bash
# scripts/09_pick_best_stage1.sh
#
# 对已训练好的 milestone ckpt，在 val 上评测并选出 stage1 末段最佳权重。
# 适用于「已训完、当时未开 val 早停」的 run；在 [EPOCH_MIN,EPOCH_MAX] 内比 mAP@0.5。
#
# 用法：
#   bash scripts/09_pick_best_stage1.sh
#   EPOCH_MIN=20 EPOCH_MAX=33 bash scripts/09_pick_best_stage1.sh
#
# 输出：
#   outputs_csma/ckpt/best_stage1.pt
#   outputs_csma/ckpt/best_stage1_meta.json
#   outputs_csma/logs/pick_best_stage1_summary.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

CKPT_DIR="${PROJECT_ROOT}/outputs_csma/ckpt"
LOG_DIR="${PROJECT_ROOT}/outputs_csma/logs"
EPOCH_MIN="${EPOCH_MIN:-20}"
EPOCH_MAX="${EPOCH_MAX:-33}"

mkdir -p "${LOG_DIR}"

SUMMARY="${LOG_DIR}/pick_best_stage1_summary.txt"
echo "pick_best_stage1  $(date -Iseconds)  epoch_range=[${EPOCH_MIN},${EPOCH_MAX}]" > "${SUMMARY}"

best_map="-1"
best_ckpt=""
best_json=""

for ckpt_path in "${CKPT_DIR}"/epoch_*.pt; do
    [[ -f "${ckpt_path}" ]] || continue
    name="$(basename "${ckpt_path}" .pt)"
    ep="${name#epoch_}"
    ep=$((10#${ep}))
    if (( ep < EPOCH_MIN || ep > EPOCH_MAX )); then
        continue
    fi
    echo ">>> 评估 ${name} (epoch ${ep})"
    if ! CKPT="${ckpt_path}" bash "${SCRIPT_DIR}/02_eval.sh"; then
        echo "[FAIL] ${name}" >> "${SUMMARY}"
        continue
    fi
    json="${LOG_DIR}/eval_${name}.json"
    map="$(conda run --no-capture-output -n RGBtest python3 -c "import json; print(json.load(open('${json}'))['map_50'])")"
    echo "${name}: mAP@0.5=${map}" | tee -a "${SUMMARY}"
    if conda run --no-capture-output -n RGBtest python3 -c "import sys; sys.exit(0 if float('${map}') > float('${best_map}') else 1)"; then
        best_map="${map}"
        best_ckpt="${ckpt_path}"
        best_json="${json}"
    fi
done

if [[ -z "${best_ckpt}" ]]; then
    echo "[ERROR] 在 epoch [${EPOCH_MIN},${EPOCH_MAX}] 未找到可评估的 ckpt" | tee -a "${SUMMARY}"
    exit 1
fi

cp -f "${best_ckpt}" "${CKPT_DIR}/best_stage1.pt"
conda run --no-capture-output -n RGBtest python3 -c "
import json, shutil
r = json.load(open('${best_json}'))
r['source_ckpt'] = '${best_ckpt}'
r['epoch_range'] = [${EPOCH_MIN}, ${EPOCH_MAX}]
json.dump(r, open('${CKPT_DIR}/best_stage1_meta.json', 'w'), indent=2)
"
echo "" | tee -a "${SUMMARY}"
echo "★ 最佳: ${best_ckpt}  mAP@0.5=${best_map}" | tee -a "${SUMMARY}"
echo "  已复制 → ${CKPT_DIR}/best_stage1.pt" | tee -a "${SUMMARY}"
echo "" | tee -a "${SUMMARY}"
echo "正式报告请用: CKPT=outputs_csma/ckpt/best_stage1.pt bash scripts/02_eval.sh"
