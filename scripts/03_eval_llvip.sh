#!/usr/bin/env bash
# LLVIP 跨数据集泛化评估（可与后台训练并行，仅推理）
set -eo pipefail
cd /root/autodl-tmp/Cross-Spectral-Modality-Adapter
export HF_HUB_OFFLINE=1

LOG_DIR="outputs_csma/logs/llvip"
mkdir -p "$LOG_DIR"

run_eval() {
    local label="$1"
    shift
    echo ""
    echo ">>> $label"
    conda run --no-capture-output -n RGBtest python3 -m src.eval_llvip_yolo "$@"
}

# ── YOLOv3-tiny（FLIR 上效果最好的 Final-CSMA）──────────────────────────────
run_eval "v3-tiny IR baseline" \
    --yolo-weights /root/autodl-tmp/yolov3-tinyu.pt \
    --input-mode ir_raw \
    --out-json "$LOG_DIR/eval_llvip_v3tiny_ir_raw.json"

run_eval "v3-tiny + Final-CSMA ep0" \
    --yolo-weights /root/autodl-tmp/yolov3-tinyu.pt \
    --input-mode pseudo_rgb \
    --pseudo-resize native \
    --ckpt outputs_csma_v3tiny_final/ckpt/epoch_0000.pt \
    --out-json "$LOG_DIR/eval_llvip_v3tiny_csma_ep0_native.json"

run_eval "v3-tiny + Final-CSMA ep0 (upscale 旧)" \
    --yolo-weights /root/autodl-tmp/yolov3-tinyu.pt \
    --input-mode pseudo_rgb \
    --pseudo-resize upscale \
    --ckpt outputs_csma_v3tiny_final/ckpt/epoch_0000.pt \
    --out-json "$LOG_DIR/eval_llvip_v3tiny_csma_ep0_upscale.json"

# ── YOLOv8m Final-CSMA ───────────────────────────────────────────────────────
run_eval "v8m IR baseline" \
    --yolo-weights /root/autodl-tmp/yolov8m.pt \
    --input-mode ir_raw \
    --out-json "$LOG_DIR/eval_llvip_v8m_ir_raw.json"

run_eval "v8m + Final-CSMA ep1" \
    --yolo-weights /root/autodl-tmp/yolov8m.pt \
    --input-mode pseudo_rgb \
    --pseudo-resize native \
    --ckpt outputs_csma_yolo_final/ckpt/epoch_0001.pt \
    --out-json "$LOG_DIR/eval_llvip_v8m_csma_ep1_native.json"

run_eval "v8m + Final-CSMA ep1 (upscale 旧)" \
    --yolo-weights /root/autodl-tmp/yolov8m.pt \
    --input-mode pseudo_rgb \
    --pseudo-resize upscale \
    --ckpt outputs_csma_yolo_final/ckpt/epoch_0001.pt \
    --out-json "$LOG_DIR/eval_llvip_v8m_csma_ep1_upscale.json"

# ── DETR（teacher 匹配：DETR-CSMA → DETR）──────────────────────────────────
echo ""
echo ">>> DETR IR baseline"
conda run --no-capture-output -n RGBtest python3 -m src.eval_llvip_detr \
    --input-mode ir_raw \
    --out-json "$LOG_DIR/eval_llvip_detr_ir_raw.json"

echo ""
echo ">>> DETR + DETR-CSMA"
conda run --no-capture-output -n RGBtest python3 -m src.eval_llvip_detr \
    --input-mode pseudo_rgb \
    --pseudo-resize native \
    --ckpt outputs_csma_detr_base/ckpt/best_stage1.pt \
    --out-json "$LOG_DIR/eval_llvip_detr_csma.json"

# ── GDINO（teacher 匹配：GDINO-CSMA → GDINO）────────────────────────────────
echo ""
echo ">>> GDINO IR pipeline 基线"
conda run --no-capture-output -n RGBtest python3 -m src.eval_llvip_gdino \
    --input-mode ir_pipeline \
    --out-json "$LOG_DIR/eval_llvip_gdino_ir_pipeline.json"

echo ""
echo ">>> GDINO + GDINO-CSMA"
conda run --no-capture-output -n RGBtest python3 -m src.eval_llvip_gdino \
    --input-mode pseudo_rgb \
    --ckpt outputs_csma/ckpt/best_stage1.pt \
    --out-json "$LOG_DIR/eval_llvip_gdino_csma.json"

# ── 汇总 ─────────────────────────────────────────────────────────────────────
python3 - <<'PYEOF'
import json, glob, os

rows = []
for path in sorted(glob.glob("outputs_csma/logs/llvip/*.json")):
    with open(path) as f:
        r = json.load(f)
    name = os.path.basename(path).replace("eval_llvip_", "").replace(".json", "")
    rows.append((name, r.get("map_50", 0), r.get("ap_person", 0), r.get("n_preds", "?")))

print(f"\n{'实验':<35}  {'mAP@0.5':>8}  {'person':>8}  {'n_preds':>8}")
print("-" * 65)
for name, m50, ap, np_ in rows:
    print(f"{name:<35}  {m50:>8.4f}  {ap:>8.4f}  {np_:>8}")
PYEOF

echo ""
echo "LLVIP 评估完成，结果在 $LOG_DIR/"
