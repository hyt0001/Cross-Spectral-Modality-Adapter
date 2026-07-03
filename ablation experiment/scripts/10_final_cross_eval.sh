#!/bin/bash
set -e
cd /root/Cross-Spectral-Modality-Adapter-main

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT_BASE="outputs_final_ft"
VARIANTS="b2 c1 c2 c3"

echo "===== 微调后权重 × M3FD 评测 ====="
for V in $VARIANTS; do
    echo ""
    echo ">>> $V × m3fd"
    python3 -m src.eval_csma \
        --ckpt "$OUT_BASE/$V/ckpt/csma_last.pt" \
        --dataset m3fd \
        --data-root /root/autodl-tmp/M3FD/val \
        --out-json "$OUT_BASE/$V/eval_m3fd.json" \
        --batch-size 4 \
        --num-workers 0 \
        2>&1 | tee "$OUT_BASE/$V/eval_m3fd.log"
    echo ">>> $V × m3fd 完成"
done

echo ""
echo "===== 微调后权重 × LLVIP 评测 ====="
for V in $VARIANTS; do
    echo ""
    echo ">>> $V × llvip"
    python3 -m src.eval_csma \
        --ckpt "$OUT_BASE/$V/ckpt/csma_last.pt" \
        --dataset llvip \
        --data-root LLVIP \
        --out-json "$OUT_BASE/$V/eval_llvip.json" \
        --batch-size 2 \
        --num-workers 0 \
        2>&1 | tee "$OUT_BASE/$V/eval_llvip.log"
    echo ">>> $V × llvip 完成"
done

echo ""
echo "===== 全部评测完成 ====="
# 打印汇总
python3 - << 'PYEOF'
import json, os

variants = ['b2', 'c1', 'c2', 'c3']
datasets = ['flir', 'm3fd', 'llvip']
base = 'outputs_final_ft'

print(f"{'变体':<5}", end='')
for ds in datasets:
    print(f"  {ds.upper():>8}", end='')
print()

for v in variants:
    print(f"{v:<5}", end='')
    for ds in datasets:
        p = f"{base}/{v}/eval_{ds}.json"
        if os.path.isfile(p):
            r = json.load(open(p))
            print(f"  {r['map_50']:>8.4f}", end='')
        else:
            print(f"  {'N/A':>8}", end='')
    print()
PYEOF
