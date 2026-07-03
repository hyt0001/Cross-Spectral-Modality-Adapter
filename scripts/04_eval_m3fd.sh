#!/usr/bin/env bash
# M3FD 跨数据集泛化评估（FLIR 训练 CSMA → M3FD-zxSplit test）
set -eo pipefail
cd /root/autodl-tmp/Cross-Spectral-Modality-Adapter
export HF_HUB_OFFLINE=1

LOG_DIR="outputs_csma/logs/m3fd"
mkdir -p "$LOG_DIR"

run_yolo() {
    local label="$1"
    shift
    echo ""
    echo ">>> $label"
    conda run --no-capture-output -n RGBtest python3 -m src.eval_m3fd_yolo "$@"
}

# ── YOLOv3-tiny ──────────────────────────────────────────────────────────────
run_yolo "v3-tiny IR baseline" \
    --yolo-weights /root/autodl-tmp/yolov3-tinyu.pt \
    --input-mode ir_raw \
    --out-json "$LOG_DIR/eval_m3fd_v3tiny_ir_raw.json"

run_yolo "v3-tiny + Final-CSMA ep0" \
    --yolo-weights /root/autodl-tmp/yolov3-tinyu.pt \
    --input-mode pseudo_rgb \
    --ckpt outputs_csma_v3tiny_final/ckpt/epoch_0000.pt \
    --out-json "$LOG_DIR/eval_m3fd_v3tiny_csma_ep0.json"

# ── YOLOv8m ──────────────────────────────────────────────────────────────────
run_yolo "v8m IR baseline" \
    --yolo-weights /root/autodl-tmp/yolov8m.pt \
    --input-mode ir_raw \
    --out-json "$LOG_DIR/eval_m3fd_v8m_ir_raw.json"

run_yolo "v8m + Final-CSMA ep1" \
    --yolo-weights /root/autodl-tmp/yolov8m.pt \
    --input-mode pseudo_rgb \
    --ckpt outputs_csma_yolo_final/ckpt/epoch_0001.pt \
    --out-json "$LOG_DIR/eval_m3fd_v8m_csma_ep1.json"

# ── DETR ─────────────────────────────────────────────────────────────────────
echo ""
echo ">>> DETR IR baseline"
conda run --no-capture-output -n RGBtest python3 -m src.eval_m3fd_detr \
    --input-mode ir_raw \
    --out-json "$LOG_DIR/eval_m3fd_detr_ir_raw.json"

echo ""
echo ">>> DETR + DETR-CSMA"
conda run --no-capture-output -n RGBtest python3 -m src.eval_m3fd_detr \
    --input-mode pseudo_rgb \
    --ckpt outputs_csma_detr_base/ckpt/best_stage1.pt \
    --out-json "$LOG_DIR/eval_m3fd_detr_csma.json"

# ── 汇总 ─────────────────────────────────────────────────────────────────────
python3 - <<'PYEOF'
import json, glob, os

rows = []
for path in sorted(glob.glob("outputs_csma/logs/m3fd/*.json")):
    with open(path) as f:
        r = json.load(f)
    name = os.path.basename(path).replace("eval_m3fd_", "").replace(".json", "")
    rows.append((
        name,
        r.get("map_50", 0),
        r.get("ap_person", 0),
        r.get("ap_car", 0),
        r.get("n_preds", "?"),
    ))

print(f"\n{'实验':<30}  {'mAP@0.5':>8}  {'person':>8}  {'car':>8}  {'n_preds':>8}")
print("-" * 75)
for name, m50, ap, ac, np_ in rows:
    print(f"{name:<30}  {m50:>8.4f}  {ap:>8.4f}  {ac:>8.4f}  {np_:>8}")
PYEOF

echo ""
echo "M3FD 评估完成，结果在 $LOG_DIR/"
