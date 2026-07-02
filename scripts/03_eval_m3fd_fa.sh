#!/usr/bin/env bash
# M3FD FeatureAdapter 评测（val 20%%，与训练早停一致）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

CKPT="${CKPT:-${PROJECT_ROOT}/outputs_m3fd_fa/ckpt/best_stage1.pt}"
OUT_JSON="${OUT_JSON:-${PROJECT_ROOT}/outputs_m3fd_fa/logs/eval_fa_val.json}"
BASELINE_JSON="${PROJECT_ROOT}/outputs_m3fd_finetune/logs/eval_m3fd_baseline_val.json"

echo "[eval] CKPT=${CKPT}"

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n RGBtest \
    python3 -m src.eval_csma \
        --adapter-mode feature \
        --ckpt "${CKPT}" \
        --dataset m3fd \
        --data-root "${PROJECT_ROOT}/M3FD" \
        --ann-file "${PROJECT_ROOT}/M3FD/annotations/val.json" \
        --split val \
        --canonical-size "1024,768" \
        --text-prompt "person. car. bus. motorcycle. truck. lamp." \
        --out-json "${OUT_JSON}"

if [[ -f "${BASELINE_JSON}" ]]; then
    conda run -n RGBtest python3 - << PY
import json
fa = json.load(open("${OUT_JSON}"))
bl = json.load(open("${BASELINE_JSON}"))
print(f"mAP@0.5  FA={fa['map_50']:.4f}  baseline={bl['map_50']:.4f}  delta={fa['map_50']-bl['map_50']:+.4f}")
for k in sorted(fa.keys()):
    if k.startswith("ap_") and k in bl:
        print(f"  {k}: FA={fa[k]:.4f}  baseline={bl[k]:.4f}")
PY
fi
