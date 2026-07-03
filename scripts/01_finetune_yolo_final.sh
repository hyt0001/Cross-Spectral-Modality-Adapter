#!/usr/bin/env bash
# =============================================================================
# 按队友 Final Model 迁移指引对 YOLO-CSMA 做 2-epoch 微调
#
# 起点：outputs_csma_yolo/ckpt/best_stage1.pt（YOLOv8m teacher，val mAP≈66.2%）
# 关键改动：
#   - lr=1e-5（比从头训小 10x）
#   - id_loss_weight=0.005 / tv_loss_weight=0.05（config.py 默认值已对应 Final Model）
#   - pseudo_clamp=2.0（config.py 默认，限制极端像素）
#   - 只训 2 epoch，每 epoch 都存权重
#   - 结束后立刻评估两个 epoch 的结果，取最好的那个
# =============================================================================

set -eo pipefail
cd /root/autodl-tmp/Cross-Spectral-Modality-Adapter

# 使用本地缓存，不访问 huggingface.co（processor 已在上次训练中缓存）
export HF_HUB_OFFLINE=1

INIT_CKPT="outputs_csma_yolo/ckpt/best_stage1.pt"
OUT_DIR="outputs_csma_yolo_final"
YOLO_W="/root/autodl-tmp/yolov8m.pt"
DATA_TRAIN="FLIR_License/train"
DATA_VAL="FLIR_License/val"

if [ ! -f "$INIT_CKPT" ]; then
    echo "[错误] 起点权重不存在: $INIT_CKPT"
    exit 1
fi

echo "=========================================="
echo " Step 1: 2-epoch Final Model 微调"
echo "  起点:  $INIT_CKPT"
echo "  输出:  $OUT_DIR"
echo "=========================================="

conda run --no-capture-output -n RGBtest \
    python3 -m src.train_csma_yolo \
        --yolo-weights  "$YOLO_W" \
        --dataset       flir_v1 \
        --data-root     "$DATA_TRAIN" \
        --out-dir       "$OUT_DIR" \
        --epochs        2 \
        --lr            1e-5 \
        --batch-size    2 \
        --loss-mode     full \
        --gmm-batches   50 \
        --init-ckpt     "$INIT_CKPT" \
        --start-epoch   0 \
        --val-early-stop \
        --val-data-root "$DATA_VAL" \
        --val-start     0 \
        --val-end       1 \
        --val-every     1 \
        --val-batch-size 4 \
        --val-conf      0.05

echo ""
echo "=========================================="
echo " Step 2: 评估 epoch_0000.pt（第 1 轮）"
echo "=========================================="

conda run --no-capture-output -n RGBtest \
    python3 -m src.eval_yolo_csma \
        --ckpt         "$OUT_DIR/ckpt/epoch_0000.pt" \
        --yolo-weights "$YOLO_W" \
        --dataset      flir_v1 \
        --data-root    "$DATA_VAL" \
        --input-mode   pseudo_rgb \
        --conf         0.05 \
        --out-json     "$OUT_DIR/logs/eval_ep0_pseudo_rgb.json"

echo ""
echo "=========================================="
echo " Step 3: 评估 epoch_0001.pt（第 2 轮，Final Model 推荐）"
echo "=========================================="

conda run --no-capture-output -n RGBtest \
    python3 -m src.eval_yolo_csma \
        --ckpt         "$OUT_DIR/ckpt/epoch_0001.pt" \
        --yolo-weights "$YOLO_W" \
        --dataset      flir_v1 \
        --data-root    "$DATA_VAL" \
        --input-mode   pseudo_rgb \
        --conf         0.05 \
        --out-json     "$OUT_DIR/logs/eval_ep1_pseudo_rgb.json"

echo ""
echo "=========================================="
echo " 汇总"
echo "=========================================="

python3 - <<'PYEOF'
import json, glob, os

baseline_path = "outputs_csma/logs/eval_yolo_ir_raw_yolov8m.json"
rows = []

if os.path.exists(baseline_path):
    with open(baseline_path) as f:
        b = json.load(f)
    rows.append(("YOLOv8m  IR raw (baseline)", b["map_50"], b["ap_person"], b["ap_car"], b.get("n_preds", "?")))

for path in sorted(glob.glob("outputs_csma_yolo_final/logs/eval_ep*_pseudo_rgb.json")):
    label = os.path.basename(path).replace("eval_", "").replace("_pseudo_rgb.json", "")
    with open(path) as f:
        r = json.load(f)
    rows.append((f"YOLO-Final {label}", r["map_50"], r["ap_person"], r["ap_car"], r.get("n_preds", "?")))

print(f"\n{'模型':<35}  {'mAP@0.5':>8}  {'person':>8}  {'car':>8}  {'n_preds':>8}")
print("-" * 75)
for name, m50, ap, ac, np_ in rows:
    print(f"{name:<35}  {m50:>8.4f}  {ap:>8.4f}  {ac:>8.4f}  {np_:>8}")
PYEOF

echo ""
echo "完成。权重在 $OUT_DIR/ckpt/，日志在 $OUT_DIR/logs/"
