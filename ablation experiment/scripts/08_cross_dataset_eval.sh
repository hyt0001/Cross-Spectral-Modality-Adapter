#!/bin/bash
# 跨数据集消融评测：4 个消融权重 × M3FD / LLVIP，共 8 次评测。
#
# 用法：
#   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#   CUDA_VISIBLE_DEVICES=0 HF_HOME=/root/autodl-tmp/hf_cache \
#   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
#   bash scripts/08_cross_dataset_eval.sh
#
# 输出：outputs_cross_eval/{variant}_{dataset}.json（+ 同名 .log）

set -u  # 未定义变量报错；单次评测失败不中断整个流程

cd "$(dirname "$0")/.."

OUT_DIR="outputs_cross_eval"
mkdir -p "${OUT_DIR}"

M3FD_ROOT="/root/autodl-tmp/M3FD/val"
LLVIP_ROOT="LLVIP"

# variant → checkpoint 路径
declare -A CKPTS=(
    [b2]="outputs_abl_b2_mean_proto/ckpt/csma_last.pt"
    [c1]="outputs_ablation_c/c1_random_mask/ckpt/csma_last.pt"
    [c2]="outputs_ablation_c/c2_fixed_threshold/ckpt/csma_last.pt"
    [c3]="outputs_ablation_c/c3_gmm_single_b_clean_rerun/ckpt/csma_last.pt"
)

run_one() {
    local variant="$1" dataset="$2" data_root="$3"
    local ckpt="${CKPTS[$variant]}"
    local tag="${variant}_${dataset}"
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  [$(date '+%H:%M:%S')] 开始评测: ${tag}"
    echo "  ckpt=${ckpt}"
    echo "════════════════════════════════════════════════════"
    python3 -m src.eval_csma \
        --ckpt "${ckpt}" \
        --dataset "${dataset}" \
        --data-root "${data_root}" \
        --out-json "${OUT_DIR}/${tag}.json" \
        --batch-size 4 \
        --num-workers 2 \
        2>&1 | tee "${OUT_DIR}/${tag}.log"
    echo "  [$(date '+%H:%M:%S')] 完成: ${tag}  exit=$?"
}

# M3FD（840 张，较快）先跑，LLVIP（3463 张）后跑
for variant in b2 c1 c2 c3; do
    run_one "${variant}" m3fd "${M3FD_ROOT}"
done

for variant in b2 c1 c2 c3; do
    run_one "${variant}" llvip "${LLVIP_ROOT}"
done

echo ""
echo "════════════════════════════════════════════════════"
echo "  全部 8 次评测完成，汇总："
echo "════════════════════════════════════════════════════"
python3 - <<'EOF'
import json, os
out_dir = "outputs_cross_eval"
print(f"{'变体':<6}{'数据集':<8}{'mAP@0.5':>10}{'AP_person':>12}{'AP_car':>10}{'预测框':>10}")
for variant in ("b2", "c1", "c2", "c3"):
    for ds in ("m3fd", "llvip"):
        p = os.path.join(out_dir, f"{variant}_{ds}.json")
        if not os.path.isfile(p):
            print(f"{variant:<6}{ds:<8}{'缺失':>10}")
            continue
        r = json.load(open(p))
        print(
            f"{variant:<6}{ds:<8}"
            f"{r.get('map_50', 0):>10.4f}"
            f"{r.get('ap_person', 0):>12.4f}"
            f"{r.get('ap_car', 0):>10.4f}"
            f"{r.get('n_preds', 0):>10d}"
        )
EOF
