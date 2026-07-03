#!/bin/bash
# Final 方法微调：对 4 个消融最佳权重各做 2-epoch 短训（加 3 个正则 loss），
# 训完立即在 FLIR val 上评测。
#
# Final 方法参数（照 docs/CSMA Final Model 配置说明.md，适配本项目权重）：
#   id_loss_weight=0.005  tv_loss_weight=0.05  logit_reg_weight=0.02（sigmoid 均值定义）
#   residual_scale=1.0 / pseudo_clamp=0：保持与原训练一致。
#   （文档的 scale=0.05/clamp=2.0 是针对其 scale=0.1 训出的权重；
#     本项目权重在 scale=1.0 下训练，直接改 scale 会摧毁模型，已实测。）
#
# 用法：
#   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#   CUDA_VISIBLE_DEVICES=0 HF_HOME=/root/autodl-tmp/hf_cache \
#   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
#   bash scripts/09_final_finetune_all.sh

set -u

cd "$(dirname "$0")/.."

DATA_ROOT="FLIR_License/train"
VAL_ROOT="FLIR_License/val"
OUT_BASE="outputs_final_ft"
EPOCHS=2
LR=1e-5

mkdir -p "${OUT_BASE}"

# variant → 初始权重
declare -A CKPTS=(
    [b2]="outputs_abl_b2_mean_proto/ckpt/csma_last.pt"
    [c1]="outputs_ablation_c/c1_random_mask/ckpt/csma_last.pt"
    [c2]="outputs_ablation_c/c2_fixed_threshold/ckpt/csma_last.pt"
    [c3]="outputs_ablation_c/c3_gmm_single_b_clean_rerun/ckpt/csma_last.pt"
)
# variant → 额外训练参数（保持与原消融训练一致的模式）
declare -A EXTRA=(
    [b2]="--variant mean_proto --mean-proto-path outputs_csma/mean_proto.pt"
    [c1]="--cmss-ablation-mode random_mask"
    [c2]="--cmss-ablation-mode fixed_threshold"
    [c3]="--cmss-ablation-mode gmm_single_b"
)

run_one() {
    local variant="$1"
    local ckpt="${CKPTS[$variant]}"
    local extra="${EXTRA[$variant]}"
    local out_dir="${OUT_BASE}/${variant}"
    mkdir -p "${out_dir}"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  [$(date '+%H:%M:%S')] Final 微调: ${variant}"
    echo "  init=${ckpt}"
    echo "  extra=${extra}"
    echo "════════════════════════════════════════════════════"
    python3 -m src.train_csma \
        --dataset flir_v1 \
        --data-root "${DATA_ROOT}" \
        --out-dir "${out_dir}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --init-ckpt "${ckpt}" \
        --id-loss-weight 0.005 \
        --tv-loss-weight 0.05 \
        --logit-reg-weight 0.02 \
        --pseudo-clamp 0 \
        --residual-scale 1.0 \
        ${extra} \
        2>&1 | tee "${out_dir}/train.log"
    local code=${PIPESTATUS[0]}
    if [[ ${code} -ne 0 ]]; then
        echo "  [FAIL] ${variant} 训练失败 exit=${code}，跳过评测"
        return
    fi

    echo "  [$(date '+%H:%M:%S')] 评测: ${variant}（FLIR val）"
    python3 -m src.eval_csma \
        --ckpt "${out_dir}/ckpt/csma_last.pt" \
        --dataset flir_v1 \
        --data-root "${VAL_ROOT}" \
        --out-json "${out_dir}/eval_flir.json" \
        2>&1 | tee "${out_dir}/eval_flir.log"
    echo "  [$(date '+%H:%M:%S')] ${variant} 完成"
}

for variant in b2 c1 c2 c3; do
    run_one "${variant}"
done

echo ""
echo "════════════════════════════════════════════════════"
echo "  全部完成，对比（微调前 → 微调后 FLIR val mAP@0.5）："
echo "════════════════════════════════════════════════════"
python3 - <<'EOF'
import json, os
before = {"b2": 0.4779, "c1": 0.5543, "c2": 0.5228, "c3": 0.5490}
print(f"{'变体':<6}{'微调前':>10}{'微调后':>10}{'变化':>10}")
for v in ("b2", "c1", "c2", "c3"):
    p = f"outputs_final_ft/{v}/eval_flir.json"
    if not os.path.isfile(p):
        print(f"{v:<6}{before[v]:>10.4f}{'缺失':>10}")
        continue
    r = json.load(open(p))
    after = r["map_50"]
    print(f"{v:<6}{before[v]:>10.4f}{after:>10.4f}{after - before[v]:>+10.4f}")
EOF
