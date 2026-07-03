#!/usr/bin/env bash
# 重跑汇总文档所需的主实验 eval，输出完整 COCO 指标 JSON。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"
OUT_DIR="${PROJECT_ROOT}/outputs_csma/logs/full_metrics"
mkdir -p "${OUT_DIR}"

run_flir() {
  local tag="$1" yolo="$2" mode="$3" ckpt="${4:-}"
  local out="${OUT_DIR}/flir_${tag}.json"
  local extra=()
  if [[ "${mode}" == "pseudo_rgb" ]]; then extra=(--ckpt "${ckpt}"); fi
  echo "=== FLIR ${tag} ==="
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  conda run --no-capture-output -n RGBtest python3 -m src.eval_yolo_csma \
    "${extra[@]}" --yolo-weights "${yolo}" --dataset flir_v1 \
    --data-root FLIR_License/val --input-mode "${mode}" \
    --out-json "${out}" --batch-size 8 --conf 0.05
}

run_llvip() {
  local tag="$1" yolo="$2" mode="$3" ckpt="${4:-}" adapt_bn="${5:-0}"
  local out="${OUT_DIR}/llvip_${tag}.json"
  local extra=(--yolo-weights "${yolo}" --input-mode "${mode}" --out-json "${out}" --batch-size 8 --conf 0.05)
  if [[ "${mode}" == "pseudo_rgb" ]]; then
    extra+=(--ckpt "${ckpt}" --pseudo-resize native --adapt-bn "${adapt_bn}")
  fi
  echo "=== LLVIP ${tag} ==="
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  conda run --no-capture-output -n RGBtest python3 -m src.eval_llvip_yolo "${extra[@]}"
}

run_m3fd() {
  local tag="$1" yolo="$2" mode="$3" ckpt="${4:-}"
  local out="${OUT_DIR}/m3fd_${tag}.json"
  local extra=(--yolo-weights "${yolo}" --input-mode "${mode}" --out-json "${out}" --batch-size 8 --conf 0.05)
  if [[ "${mode}" == "pseudo_rgb" ]]; then extra+=(--ckpt "${ckpt}"); fi
  echo "=== M3FD ${tag} ==="
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  conda run --no-capture-output -n RGBtest python3 -m src.eval_m3fd_yolo "${extra[@]}"
}

# FLIR
run_flir v3tiny_ir_raw   /root/autodl-tmp/yolov3-tinyu.pt ir_raw
run_flir v3tiny_csma_ep0 /root/autodl-tmp/yolov3-tinyu.pt pseudo_rgb outputs_csma_v3tiny_final/ckpt/epoch_0000.pt
run_flir v8m_ir_raw      /root/autodl-tmp/yolov8m.pt         ir_raw
run_flir v8m_csma_ep1    /root/autodl-tmp/yolov8m.pt         pseudo_rgb outputs_csma_yolo_final/ckpt/epoch_0001.pt
run_flir v8n_ir_raw      /root/autodl-tmp/yolov8n.pt         ir_raw
run_flir v8n_csma_ep1    /root/autodl-tmp/yolov8n.pt         pseudo_rgb outputs_csma_yolov8n_final/ckpt/epoch_0001.pt

# LLVIP
run_llvip v3tiny_ir_raw   /root/autodl-tmp/yolov3-tinyu.pt ir_raw
run_llvip v3tiny_csma_adabn50 /root/autodl-tmp/yolov3-tinyu.pt pseudo_rgb outputs_csma_v3tiny_final/ckpt/epoch_0000.pt 50
run_llvip v8m_ir_raw      /root/autodl-tmp/yolov8m.pt         ir_raw
run_llvip v8m_csma_adabn50 /root/autodl-tmp/yolov8m.pt         pseudo_rgb outputs_csma_yolo_final/ckpt/epoch_0001.pt 50

# M3FD
run_m3fd v3tiny_ir_raw   /root/autodl-tmp/yolov3-tinyu.pt ir_raw
run_m3fd v3tiny_csma_ep0 /root/autodl-tmp/yolov3-tinyu.pt pseudo_rgb outputs_csma_v3tiny_final/ckpt/epoch_0000.pt
run_m3fd v8m_ir_raw      /root/autodl-tmp/yolov8m.pt         ir_raw
run_m3fd v8m_csma_ep1    /root/autodl-tmp/yolov8m.pt         pseudo_rgb outputs_csma_yolo_final/ckpt/epoch_0001.pt

echo "[OK] full metrics JSON -> ${OUT_DIR}/"
