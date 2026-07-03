"""
LLVIP 数据集上 YOLO 检测评估（ir_raw 基线 + CSMA 伪 RGB 跨数据集泛化）。

LLVIP：1280×1024 红外，仅 person 类。
复用 eval_llvip_detr 的数据加载与坐标处理逻辑。

CLI 示例：
    # YOLOv8n IR baseline
    HF_HUB_OFFLINE=1 conda run -n RGBtest python -m src.eval_llvip_yolo \\
        --yolo-weights /root/autodl-tmp/yolov8n.pt \\
        --input-mode ir_raw \\
        --out-json outputs_csma/logs/eval_llvip_yolov8n_ir_raw.json

    # v3-tiny Final-CSMA → YOLOv3-tiny
    HF_HUB_OFFLINE=1 conda run -n RGBtest python -m src.eval_llvip_yolo \\
        --yolo-weights /root/autodl-tmp/yolov3-tinyu.pt \\
        --input-mode pseudo_rgb \\
        --ckpt outputs_csma_v3tiny_final/ckpt/epoch_0000.pt \\
        --out-json outputs_csma/logs/eval_llvip_v3tiny_csma_pseudo_rgb.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from src.config import CSMAConfig
from src.csma import CSMA
from src.eval_csma import compute_map
from src.eval_llvip_detr import (
    LLVIP_GT_JSON,
    LLVIP_IR_DIR,
    LLVIPIRDataset,
    PseudoResizeMode,
    _build_llvip_gt,
    _collate_llvip,
    adapt_csma_bn,
    prepare_pseudo_rgb_for_detector,
    pseudo_rgb_uint8_from_csma,
    scale_xyxy_boxes,
    unsupervised_tta,
)
from src.eval_yolo_csma import _configure_processor

# YOLO COCO person=0 → LLVIP eval cat 1
YOLO_PERSON_TO_LLVIP: Dict[int, int] = {0: 1}


def _load_csma(ckpt: str, cfg: CSMAConfig, device: torch.device) -> CSMA:
    csma = CSMA(cfg).to(device)
    raw = torch.load(ckpt, map_location=device, weights_only=True)
    state = raw["csma"] if isinstance(raw, dict) and "csma" in raw else raw
    csma.load_state_dict(state)
    csma.eval()
    return csma


def run_yolo_llvip_eval(
    yolo_model: Any,
    dataset: LLVIPIRDataset,
    device: torch.device,
    batch_size: int,
    conf: float,
    input_mode: str,
    csma: Optional[CSMA],
    gdino_mean: Tuple[float, ...],
    gdino_std: Tuple[float, ...],
    pseudo_resize: PseudoResizeMode = "native",
) -> List[Dict]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_llvip, num_workers=2,
    )
    predictions: List[Dict] = []
    total = len(dataset)
    processed = 0

    with torch.no_grad():
        for batch in loader:
            image_ids = batch["image_ids"]
            orig_ws = batch["orig_ws"]
            orig_hs = batch["orig_hs"]
            file_names = batch["file_names"]
            bsz = len(image_ids)

            batch_images: List[np.ndarray] = []
            box_scales: List[Tuple[float, float]] = [(1.0, 1.0)] * bsz

            if input_mode in ("pseudo_rgb", "ir_pipeline"):
                ir_pv = batch["pixel_values"].to(device)
                pm = batch["pixel_masks"]
                if input_mode == "pseudo_rgb":
                    assert csma is not None
                    feat_batch = csma(ir_pv)
                else:
                    # 诊断：CSMA=恒等，走完全相同的 512 分辨率链路
                    feat_batch = ir_pv
                batch_images = []
                box_scales = []
                for i in range(bsz):
                    arr = pseudo_rgb_uint8_from_csma(
                        feat_batch[i], pm[i], gdino_mean, gdino_std,
                    )
                    det_arr, sx, sy = prepare_pseudo_rgb_for_detector(
                        arr, orig_ws[i], orig_hs[i], pseudo_resize,
                    )
                    batch_images.append(det_arr)
                    box_scales.append((sx, sy))
            else:
                for fn in file_names:
                    pil = Image.open(os.path.join(dataset._ir_dir, fn)).convert("RGB")
                    batch_images.append(np.asarray(pil, dtype=np.uint8))

            results_list = yolo_model.predict(source=batch_images, conf=conf, verbose=False)

            for res, img_id, (sx, sy) in zip(results_list, image_ids, box_scales):
                if res.boxes is None or len(res.boxes) == 0:
                    continue
                xyxy = scale_xyxy_boxes(res.boxes.xyxy.cpu().numpy(), sx, sy)
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                for box, score, cls_id in zip(xyxy, confs, clss):
                    cat_id = YOLO_PERSON_TO_LLVIP.get(int(cls_id))
                    if cat_id is None:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    predictions.append({
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    })

            processed += bsz
            print(f"\r  YOLO 推理进度: {processed}/{total}", end="", flush=True)

    print()
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="LLVIP 上 YOLO + CSMA 评估")
    parser.add_argument("--ir-dir", type=str, default=LLVIP_IR_DIR)
    parser.add_argument("--gt-json", type=str, default=LLVIP_GT_JSON)
    parser.add_argument("--yolo-weights", type=str, required=True)
    parser.add_argument("--input-mode", type=str, default="ir_raw",
                        choices=["ir_raw", "pseudo_rgb", "ir_pipeline"])
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--out-json", type=str,
                        default="outputs_csma/logs/eval_llvip_yolo.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument(
        "--pseudo-resize",
        type=str,
        default="native",
        choices=["native", "upscale"],
        help="pseudo_rgb 尺寸策略：native=CSMA 分辨率检测后映射坐标；upscale=放大到原图再检测",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=None,
        help="覆盖 CSMAConfig.residual_scale（默认 1.0）。0.0=纯 tanh，0.5=半残差",
    )
    parser.add_argument(
        "--pseudo-clamp",
        type=float,
        default=None,
        help="覆盖 CSMAConfig.pseudo_clamp（默认 2.0）。0.0=不截断",
    )
    parser.add_argument(
        "--adapt-bn",
        type=int,
        default=0,
        metavar="N",
        help="AdaBN：在目标域做 N 步 forward（train 模式）更新 BN 统计量后再切 eval",
    )
    parser.add_argument(
        "--tta-steps",
        type=int,
        default=0,
        metavar="N",
        help="无监督 TTA：在目标域 IR 图上做 N 步梯度更新（L_id + L_tv），不需要 GT",
    )
    parser.add_argument(
        "--tta-ir-dir",
        type=str,
        default="/root/autodl-tmp/LLVIP/LLVIP/infrared/train",
        help="TTA 用的目标域 IR 目录（无需标注）",
    )
    parser.add_argument("--tta-lr", type=float, default=2e-5, help="TTA 学习率")
    args = parser.parse_args()

    if args.input_mode == "pseudo_rgb" and not args.ckpt:
        raise ValueError("pseudo_rgb 模式需要 --ckpt")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("请安装 ultralytics") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_llvip_yolo] 设备:       {device}")
    print(f"[eval_llvip_yolo] YOLO:       {args.yolo_weights}")
    print(f"[eval_llvip_yolo] input_mode: {args.input_mode}")
    if args.input_mode == "pseudo_rgb":
        print(f"[eval_llvip_yolo] pseudo_resize: {args.pseudo_resize}")

    cfg = CSMAConfig()
    if args.residual_scale is not None:
        cfg.residual_scale = args.residual_scale
    if args.pseudo_clamp is not None:
        cfg.pseudo_clamp = args.pseudo_clamp
    dino_proc = AutoProcessor.from_pretrained(cfg.model_id)
    _configure_processor(dino_proc, cfg.img_size)
    ip = dino_proc.image_processor
    gdino_mean = tuple(ip.image_mean)
    gdino_std = tuple(ip.image_std)

    csma: Optional[CSMA] = None
    if args.input_mode == "pseudo_rgb":
        print(f"[eval_llvip_yolo] CSMA ckpt: {args.ckpt}")
        if args.residual_scale is not None:
            print(f"[eval_llvip_yolo] residual_scale 覆盖: {args.residual_scale}")
        if args.pseudo_clamp is not None:
            print(f"[eval_llvip_yolo] pseudo_clamp 覆盖: {args.pseudo_clamp}")
        csma = _load_csma(args.ckpt, cfg, device)

    dataset = LLVIPIRDataset(args.ir_dir, args.gt_json, dino_proc, cfg)
    coco_gt = _build_llvip_gt(dataset)
    print(f"[eval_llvip_yolo] GT 框数: {len(coco_gt.anns)}")

    if csma is not None and args.adapt_bn > 0:
        print(f"[eval_llvip_yolo] AdaBN: {args.adapt_bn} 步 ...")
        adapt_csma_bn(csma, dataset, device, adapt_steps=args.adapt_bn, batch_size=args.batch_size)

    if csma is not None and args.tta_steps > 0:
        print(f"[eval_llvip_yolo] TTA: {args.tta_steps} 步 @ lr={args.tta_lr} ...")
        unsupervised_tta(
            csma, args.tta_ir_dir, dino_proc, device,
            tta_steps=args.tta_steps, batch_size=args.batch_size, lr=args.tta_lr,
        )

    yolo = YOLO(args.yolo_weights)
    predictions = run_yolo_llvip_eval(
        yolo_model=yolo,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        conf=args.conf,
        input_mode=args.input_mode,
        csma=csma,
        gdino_mean=gdino_mean,
        gdino_std=gdino_std,
        pseudo_resize=args.pseudo_resize,  # type: ignore[arg-type]
    )
    print(f"[eval_llvip_yolo] 预测框数: {len(predictions)}")

    results = compute_map(coco_gt, predictions)
    results.update({
        "yolo_weights": args.yolo_weights,
        "input_mode": args.input_mode,
        "ckpt": args.ckpt or None,
        "dataset": "LLVIP",
        "conf": args.conf,
        "pseudo_resize":  args.pseudo_resize if args.input_mode == "pseudo_rgb" else None,
        "residual_scale": args.residual_scale,
        "pseudo_clamp":   args.pseudo_clamp,
        "adapt_bn":   args.adapt_bn,
        "tta_steps":  args.tta_steps,
        "tta_lr":     args.tta_lr if args.tta_steps > 0 else None,
    })

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 55)
    print(f"  dataset       : LLVIP (person only)")
    print(f"  input_mode    : {args.input_mode}")
    print(f"  mAP@0.5       : {results['map_50']:.4f}")
    print(f"  AP_person@0.5 : {results['ap_person']:.4f}")
    print(f"  预测框 / GT框  : {results['n_preds']} / {results['n_gt']}")
    print("=" * 55)
    print(f"[eval_llvip_yolo] 结果: {args.out_json}")


if __name__ == "__main__":
    main()
