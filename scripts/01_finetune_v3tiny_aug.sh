#!/usr/bin/env bash
# =============================================================================
# YOLOv3-tiny + IR 域增强重训
#
# 目的：改善 CSMA 跨数据集泛化（FLIR→LLVIP）
#   - 从现有 v3-tiny base best 热启动
#   - 加 IRAugment（亮度/对比度/gamma/噪声/直方图均衡）
#   - Phase 1：20 epoch，stop-after-stage1，val 早停
#   - Phase 2：2 epoch Final Model 微调
#   - 最后同时在 FLIR val 和 LLVIP 上评估
# =============================================================================
set -eo pipefail
cd /root/autodl-tmp/Cross-Spectral-Modality-Adapter
export HF_HUB_OFFLINE=1

V3TINY="/root/autodl-tmp/yolov3-tinyu.pt"
WARMSTART="outputs_csma_v3tiny_base/ckpt/best_stage1.pt"
OUT1="outputs_csma_v3tiny_aug_base"
OUT2="outputs_csma_v3tiny_aug_final"
DATA_TRAIN="FLIR_License/train"
DATA_VAL="FLIR_License/val"
LOG="outputs_csma_v3tiny_aug.log"

if [ ! -f "$WARMSTART" ]; then
    echo "[错误] 热启动权重不存在: $WARMSTART"
    echo "  请先确认 outputs_csma_v3tiny_base/ckpt/best_stage1.pt 存在"
    exit 1
fi

echo "=========================================="
echo " Phase 1: v3-tiny + IR 增强，20 epoch"
echo "=========================================="
conda run --no-capture-output -n RGBtest \
    python3 -m src.train_csma_yolo \
        --yolo-weights  "$V3TINY" \
        --dataset       flir_v1 \
        --data-root     "$DATA_TRAIN" \
        --out-dir       "$OUT1" \
        --epochs        30 \
        --lr            1e-4 \
        --batch-size    2 \
        --loss-mode     full \
        --gmm-batches   50 \
        --init-ckpt     "$WARMSTART" \
        --start-epoch   0 \
        --val-early-stop \
        --val-data-root "$DATA_VAL" \
        --val-batch-size 4 \
        --val-conf      0.05 \
        --stop-after-stage1 \
        --ir-aug \
        --ir-aug-prob   0.8 \
    2>&1 | tee -a "$LOG"

echo ""
echo "Phase 1 完成。best: $OUT1/ckpt/best_stage1.pt"

echo ""
echo "=========================================="
echo " Phase 2: Final Model 微调，2 epoch"
echo "=========================================="
conda run --no-capture-output -n RGBtest \
    python3 -m src.train_csma_yolo \
        --yolo-weights  "$V3TINY" \
        --dataset       flir_v1 \
        --data-root     "$DATA_TRAIN" \
        --out-dir       "$OUT2" \
        --epochs        2 \
        --lr            1e-5 \
        --batch-size    2 \
        --loss-mode     full \
        --gmm-batches   50 \
        --init-ckpt     "$OUT1/ckpt/best_stage1.pt" \
        --start-epoch   0 \
        --val-batch-size 4 \
        --val-conf      0.05 \
        --ir-aug \
        --ir-aug-prob   0.8 \
    2>&1 | tee -a "$LOG"

echo ""
echo "Phase 2 完成。"

echo ""
echo "=========================================="
echo " Phase 3: FLIR val 评估"
echo "=========================================="
for EP in 0000 0001; do
    CKPT="$OUT2/ckpt/epoch_${EP}.pt"
    [ -f "$CKPT" ] || continue
    conda run --no-capture-output -n RGBtest \
        python3 -m src.eval_yolo_csma \
            --ckpt         "$CKPT" \
            --yolo-weights "$V3TINY" \
            --dataset      flir_v1 \
            --data-root    "$DATA_VAL" \
            --input-mode   pseudo_rgb \
            --conf         0.05 \
            --out-json     "$OUT2/logs/eval_flir_ep${EP}.json" \
    2>&1 | tee -a "$LOG"
done

echo ""
echo "=========================================="
echo " Phase 4: LLVIP 跨域评估"
echo "=========================================="
LOG_LLVIP="outputs_csma/logs/llvip"
mkdir -p "$LOG_LLVIP"

# best stage1（ep0 无增强基线对比）
for CKPT_TAG in \
    "$OUT1/ckpt/best_stage1.pt:aug_base_best" \
    "$OUT2/ckpt/epoch_0000.pt:aug_final_ep0" \
    "$OUT2/ckpt/epoch_0001.pt:aug_final_ep1"; do
    CKPT="${CKPT_TAG%%:*}"; TAG="${CKPT_TAG##*:}"
    [ -f "$CKPT" ] || continue
    echo ">>> LLVIP eval: $TAG"
    conda run --no-capture-output -n RGBtest \
        python3 -m src.eval_llvip_yolo \
            --yolo-weights "$V3TINY" \
            --input-mode   pseudo_rgb \
            --pseudo-resize native \
            --ckpt         "$CKPT" \
            --adapt-bn     50 \
            --out-json     "$LOG_LLVIP/eval_llvip_v3tiny_${TAG}.json" \
    2>&1 | tee -a "$LOG"
done

echo ""
echo "=========================================="
echo " 汇总对比"
echo "=========================================="
python3 - <<'PYEOF'
import json, glob, os

def r(p):
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

print(f"\n{'实验':<45}  {'FLIR mAP':>9}  {'LLVIP mAP':>10}")
print("-" * 70)

# 无增强 baseline（已有结果）
for path, label in [
    ("outputs_csma/logs/eval_yolo_ir_raw_yolov3-tinyu.json", "v3-tiny IR raw (FLIR)"),
    ("outputs_csma/logs/llvip/eval_llvip_v3tiny_ir_raw.json", "v3-tiny IR raw (LLVIP)"),
]:
    d = r(path)
    if d: print(f"{'  ' + label:<45}  {d['map_50']:>9.4f}  {'—':>10}")

d = r("outputs_csma_v3tiny_final/logs/eval_ep0000_pseudo_rgb.json")
dl = r("outputs_csma/logs/llvip/eval_llvip_v3tiny_native_adabn50.json")
if d: print(f"{'  无增强 CSMA (FLIR ep0)':<45}  {d['map_50']:>9.4f}  {dl['map_50']:>10.4f}" if dl else "")

print()

# 有增强
for ep in ["0000", "0001"]:
    df = r(f"outputs_csma_v3tiny_aug_final/logs/eval_flir_ep{ep}.json")
    dl = r(f"outputs_csma/logs/llvip/eval_llvip_v3tiny_aug_final_ep{ep}.json")
    if df or dl:
        f_str = f"{df['map_50']:>9.4f}" if df else f"{'—':>9}"
        l_str = f"{dl['map_50']:>10.4f}" if dl else f"{'—':>10}"
        print(f"{'  +IRAug CSMA (ep' + ep + ')':<45}  {f_str}  {l_str}")

PYEOF

echo ""
echo "全部完成！日志: $LOG"
