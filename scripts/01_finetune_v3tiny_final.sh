#!/usr/bin/env bash
# =============================================================================
# YOLOv3-tiny 版 Final Model 迁移实验
#
# Phase 1: 以 YOLOv8m-CSMA 权重热启动，用 v3-tiny 做 teacher 训 20 epoch → 起点 best
# Phase 2: 从起点 best 微调 2 epoch（Final Model 正则）
# Phase 3: 评估两个 epoch 的结果 + baseline 对比
# =============================================================================

set -eo pipefail
cd /root/autodl-tmp/Cross-Spectral-Modality-Adapter
export HF_HUB_OFFLINE=1

V3TINY="/root/autodl-tmp/yolov3-tinyu.pt"
WARMSTART="outputs_csma_yolo/ckpt/best_stage1.pt"   # 旧格式 plain state dict，自动跳过 projector
OUT1="outputs_csma_v3tiny_base"                      # Phase 1 输出
OUT2="outputs_csma_v3tiny_final"                     # Phase 2 输出
DATA_TRAIN="FLIR_License/train"
DATA_VAL="FLIR_License/val"

if [ ! -f "$V3TINY" ]; then
    echo "[错误] YOLOv3-tiny 权重不存在: $V3TINY"
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
echo "=========================================="
echo " Phase 1: YOLOv3-tiny teacher，20 epoch（热启动）"
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
        --stop-after-stage1

echo ""
echo "Phase 1 完成。起点 best: $OUT1/ckpt/best_stage1.pt"

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Phase 2: Final Model 微调，2 epoch，lr=1e-5"
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
        --val-conf      0.05

echo ""
echo "Phase 2 完成。"

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Phase 3: 评估"
echo "=========================================="

for EP in 0000 0001; do
    CKPT="$OUT2/ckpt/epoch_${EP}.pt"
    if [ ! -f "$CKPT" ]; then
        echo "  [跳过] $CKPT 不存在"
        continue
    fi
    conda run --no-capture-output -n RGBtest \
        python3 -m src.eval_yolo_csma \
            --ckpt         "$CKPT" \
            --yolo-weights "$V3TINY" \
            --dataset      flir_v1 \
            --data-root    "$DATA_VAL" \
            --input-mode   pseudo_rgb \
            --conf         0.05 \
            --out-json     "$OUT2/logs/eval_ep${EP}_pseudo_rgb.json"
done

echo ""
echo "=========================================="
echo " 汇总"
echo "=========================================="

python3 - <<'PYEOF'
import json, glob, os

rows = []

# baseline
for path, label in [
    ("outputs_csma/logs/eval_yolo_ir_raw_yolov3-tinyu.json",  "YOLOv3-tiny IR raw"),
    ("outputs_csma/logs/eval_yolo_ir_raw_yolov8m.json",       "YOLOv8m   IR raw"),
    ("outputs_csma_yolo_final/logs/eval_ep0001_pseudo_rgb.json", "YOLOv8m + Final-CSMA ep1 (参考)"),
]:
    if os.path.exists(path):
        with open(path) as f: r = json.load(f)
        rows.append((label, r["map_50"], r["ap_person"], r["ap_car"], r.get("n_preds","?")))

for path in sorted(glob.glob("outputs_csma_v3tiny_final/logs/eval_ep*_pseudo_rgb.json")):
    ep = os.path.basename(path)[7:11]
    with open(path) as f: r = json.load(f)
    rows.append((f"v3-tiny + Final-CSMA ep{int(ep)}", r["map_50"], r["ap_person"], r["ap_car"], r.get("n_preds","?")))

print(f"\n{'模型':<42}  {'mAP@0.5':>8}  {'person':>8}  {'car':>8}  {'n_preds':>8}")
print("-" * 80)
for name, m50, ap, ac, np_ in rows:
    print(f"{name:<42}  {m50:>8.4f}  {ap:>8.4f}  {ac:>8.4f}  {np_:>8}")
PYEOF

echo ""
echo "完成。"
