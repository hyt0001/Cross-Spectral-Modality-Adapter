#!/usr/bin/env bash
# scripts/08_eval_phase0_all.sh
#
# phase0 训练完成后，批量评估本轮所有里程碑权重（FLIR v1 val）。
# 输出：outputs_csma/logs/eval_<ckpt名>.json 与 eval_phase0_summary.txt
#
# 用法：
#   bash scripts/08_eval_phase0_all.sh
#   CKPTS="csma_last.pt epoch_0040.pt" bash scripts/08_eval_phase0_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

CKPT_DIR="${PROJECT_ROOT}/outputs_csma/ckpt"
LOG_DIR="${PROJECT_ROOT}/outputs_csma/logs"
mkdir -p "${LOG_DIR}"

# 本轮 phase0 落盘权重（0-based：epoch_0049 = 第 50 轮结束）
CKPTS="${CKPTS:-epoch_0000.pt epoch_0010.pt epoch_0020.pt epoch_0030.pt epoch_0040.pt epoch_0049.pt csma_last.pt}"

SUMMARY="${LOG_DIR}/eval_phase0_summary.txt"
echo "phase0 eval  $(date -Iseconds)" > "${SUMMARY}"
printf "%-20s %10s %10s %10s %10s\n" "checkpoint" "mAP@0.5" "mAP@0.5:95" "AP_person" "AP_car" >> "${SUMMARY}"
printf "%-20s %10s %10s %10s %10s\n" "----------" "--------" "----------" "---------" "------" >> "${SUMMARY}"

for ckpt_name in ${CKPTS}; do
    ckpt_path="${CKPT_DIR}/${ckpt_name}"
    if [[ ! -f "${ckpt_path}" ]]; then
        echo "[SKIP] 不存在: ${ckpt_path}" | tee -a "${SUMMARY}"
        continue
    fi
    echo ""
    echo "======== 评估 ${ckpt_name} ========"
    if CKPT="${ckpt_path}" bash "${SCRIPT_DIR}/02_eval.sh"; then
        base="${ckpt_name%.pt}"
        json="${LOG_DIR}/eval_${base}.json"
        conda run --no-capture-output -n RGBtest python3 -c "
import json
r = json.load(open('${json}'))
print('${ckpt_name}: mAP@0.5={:.4f}  person={:.4f}  car={:.4f}'.format(
    r.get('map_50',0), r.get('ap_person',0), r.get('ap_car',0)))
with open('${SUMMARY}', 'a') as f:
    f.write('{:20s} {:10.4f} {:10.4f} {:10.4f} {:10.4f}\n'.format(
        '${ckpt_name}', r.get('map_50',0), r.get('map_50_95',0),
        r.get('ap_person',0), r.get('ap_car',0)))
"
    else
        echo "[FAIL] ${ckpt_name}" >> "${SUMMARY}"
    fi
done

echo ""
echo "[OK] 汇总表: ${SUMMARY}"
cat "${SUMMARY}"
