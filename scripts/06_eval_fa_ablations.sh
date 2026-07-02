#!/usr/bin/env bash
# 评测 FA 消融 (A)(B) 并与基线 / 像素 CSMA / 旧 FA 对比
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

BASELINE_JSON="${PROJECT_ROOT}/outputs_m3fd_finetune/logs/eval_m3fd_baseline_val.json"
PIXEL_JSON="${PROJECT_ROOT}/outputs_m3fd_finetune/logs/eval_m3fd_csma_val.json"
OLD_FA_JSON="${PROJECT_ROOT}/outputs_m3fd_fa/logs/eval_fa_val.json"

run_eval() {
    local label="$1"
    local ckpt="$2"
    local out_json="$3"
    if [[ ! -f "${ckpt}" ]]; then
        echo "[skip] ${label}: 权重不存在 ${ckpt}"
        return
    fi
    echo "[eval] ${label} -> ${out_json}"
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    conda run --no-capture-output -n RGBtest \
        python3 -m src.eval_csma \
            --adapter-mode feature \
            --ckpt "${ckpt}" \
            --dataset m3fd \
            --data-root "${PROJECT_ROOT}/M3FD" \
            --ann-file "${PROJECT_ROOT}/M3FD/annotations/val.json" \
            --split val \
            --canonical-size "1024,768" \
            --text-prompt "person. car. bus. motorcycle. truck. lamp." \
            --out-json "${out_json}"
}

run_eval "FA-A det_only" \
    "${PROJECT_ROOT}/outputs_m3fd_fa_ablate_a/ckpt/best_stage1.pt" \
    "${PROJECT_ROOT}/outputs_m3fd_fa_ablate_a/logs/eval_val.json"

run_eval "FA-B weak_align" \
    "${PROJECT_ROOT}/outputs_m3fd_fa_ablate_b/ckpt/best_stage1.pt" \
    "${PROJECT_ROOT}/outputs_m3fd_fa_ablate_b/logs/eval_val.json"

conda run -n RGBtest python3 - << PY
import json
from pathlib import Path

rows = [
    ("baseline", "${BASELINE_JSON}"),
    ("pixel_csma", "${PIXEL_JSON}"),
    ("fa_old", "${OLD_FA_JSON}"),
    ("fa_a_det_only", "${PROJECT_ROOT}/outputs_m3fd_fa_ablate_a/logs/eval_val.json"),
    ("fa_b_weak_align", "${PROJECT_ROOT}/outputs_m3fd_fa_ablate_b/logs/eval_val.json"),
]

print("=" * 72)
print(f"{'method':<18} {'mAP@0.5':>8} {'bus':>8} {'person':>8} {'car':>8}")
print("-" * 72)
for name, path in rows:
    p = Path(path)
    if not p.is_file():
        print(f"{name:<18} {'(missing)':>8}")
        continue
    d = json.loads(p.read_text())
    print(
        f"{name:<18} {d.get('map_50', 0):>8.4f} "
        f"{d.get('ap_bus', 0):>8.4f} "
        f"{d.get('ap_person', 0):>8.4f} "
        f"{d.get('ap_car', 0):>8.4f}"
    )
print("=" * 72)
print("成功判据: mAP@0.5 > 0.224 且 bus AP >= 0.15")
PY
