#!/usr/bin/env bash
# 评估 epoch 0/10/30/49 并输出 mAP 趋势表
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
CKPTS="epoch_0000.pt epoch_0010.pt epoch_0030.pt epoch_0049.pt"
LOG="${PROJECT_ROOT}/outputs_csma/logs/eval_trend_0_10_30_49.log"
SUMMARY="${PROJECT_ROOT}/outputs_csma/logs/eval_trend_0_10_30_49.txt"
exec > >(tee -a "${LOG}") 2>&1
echo "=== eval trend $(date -Iseconds) ==="
printf "%-16s %10s %10s %10s %10s %12s\n" "checkpoint" "mAP@0.5" "mAP@0.5:95" "AP_person" "AP_car" "n_preds" > "${SUMMARY}"
printf "%-16s %10s %10s %10s %10s %12s\n" "----------" "--------" "----------" "---------" "------" "--------" >> "${SUMMARY}"
for ckpt_name in ${CKPTS}; do
    ckpt="${PROJECT_ROOT}/outputs_csma/ckpt/${ckpt_name}"
    [[ -f "${ckpt}" ]] || { echo "[SKIP] ${ckpt}"; continue; }
    echo "--- ${ckpt_name} ---"
    CKPT="${ckpt}" bash "${PROJECT_ROOT}/scripts/02_eval.sh" || echo "[FAIL] ${ckpt_name}"
    base="${ckpt_name%.pt}"
    json="${PROJECT_ROOT}/outputs_csma/logs/eval_${base}.json"
    conda run --no-capture-output -n RGBtest python3 -c "
import json
r=json.load(open('${json}'))
print('${ckpt_name}: mAP@0.5={:.4f}'.format(r['map_50']))
with open('${SUMMARY}','a') as f:
    f.write('{:16s} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:12d}\n'.format(
        '${ckpt_name}', r['map_50'], r['map_50_95'], r['ap_person'], r['ap_car'], r['n_preds']))
"
done
echo "=== done ==="
cat "${SUMMARY}"
