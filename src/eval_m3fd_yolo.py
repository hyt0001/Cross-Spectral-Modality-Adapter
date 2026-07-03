"""
M3FD 数据集上 YOLO 检测评估（ir_raw 基线 + CSMA 伪 RGB 跨数据集泛化）。

M3FD：1024×768 红外，评估 person(People) + car(Car)，使用 M3FD-zxSplit test 集。
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
from src.eval_m3fd_common import (
    M3FD_GT_JSON,
    M3FD_IR_DIR,
    M3FDIRDataset,
    _build_m3fd_gt,
    _collate_m3fd,
    pad_for_csma,
)
from src.eval_yolo_csma import _configure_processor
from src.infer_vis import denormalize_pixel_values

# COCO YOLO: person=0, car=2 → eval cat 1, 2
YOLO_CLS_TO_M3FD_EVAL: Dict[int, int] = {0: 1, 2: 2}


def _load_csma(ckpt: str, cfg: CSMAConfig, device: torch.device) -> CSMA:
    csma = CSMA(cfg).to(device)
    raw = torch.load(ckpt, map_location=device, weights_only=True)
    state = raw["csma"] if isinstance(raw, dict) and "csma" in raw else raw
    csma.load_state_dict(state)
    csma.eval()
    return csma


def run_yolo_m3fd_eval(
    yolo_model: Any,
    dataset: M3FDIRDataset,
    device: torch.device,
    batch_size: int,
    conf: float,
    input_mode: str,
    csma: Optional[CSMA],
    gdino_mean: Tuple[float, ...],
    gdino_std: Tuple[float, ...],
) -> List[Dict]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_m3fd, num_workers=2,
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

            if input_mode == "pseudo_rgb":
                assert csma is not None
                ir_pv = batch["pixel_values"].to(device)
                pm = batch["pixel_masks"]
                for i in range(bsz):
                    pv_i, pm_i = pad_for_csma(
                        ir_pv[i : i + 1], pm[i], multiple=8,
                    )
                    pseudo = csma(pv_i)
                    oh = int(pm[i][:, 0].sum().item())
                    ow = int(pm[i][0, :].sum().item())
                    arr = denormalize_pixel_values(pseudo[0].cpu(), gdino_mean, gdino_std)
                    arr = arr[:oh, :ow]
                    pil = Image.fromarray(arr).resize(
                        (orig_ws[i], orig_hs[i]), Image.BILINEAR,
                    )
                    batch_images.append(np.asarray(pil, dtype=np.uint8))
            else:
                for fn in file_names:
                    pil = Image.open(os.path.join(dataset._ir_dir, fn)).convert("RGB")
                    batch_images.append(np.asarray(pil, dtype=np.uint8))

            results_list = yolo_model.predict(source=batch_images, conf=conf, verbose=False)

            for res, img_id in zip(results_list, image_ids):
                if res.boxes is None or len(res.boxes) == 0:
                    continue
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                for box, score, cls_id in zip(xyxy, confs, clss):
                    cat_id = YOLO_CLS_TO_M3FD_EVAL.get(int(cls_id))
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
    parser = argparse.ArgumentParser(description="M3FD 上 YOLO + CSMA 评估")
    parser.add_argument("--ir-dir", type=str, default=M3FD_IR_DIR)
    parser.add_argument("--gt-json", type=str, default=M3FD_GT_JSON)
    parser.add_argument("--yolo-weights", type=str, required=True)
    parser.add_argument("--input-mode", type=str, default="ir_raw",
                        choices=["ir_raw", "pseudo_rgb"])
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--out-json", type=str,
                        default="outputs_csma/logs/eval_m3fd_yolo.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.05)
    args = parser.parse_args()

    if args.input_mode == "pseudo_rgb" and not args.ckpt:
        raise ValueError("pseudo_rgb 模式需要 --ckpt")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("请安装 ultralytics") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_m3fd_yolo] 设备:       {device}")
    print(f"[eval_m3fd_yolo] YOLO:       {args.yolo_weights}")
    print(f"[eval_m3fd_yolo] input_mode: {args.input_mode}")

    cfg = CSMAConfig()
    dino_proc = AutoProcessor.from_pretrained(cfg.model_id)
    _configure_processor(dino_proc, cfg.img_size)
    ip = dino_proc.image_processor
    gdino_mean = tuple(ip.image_mean)
    gdino_std = tuple(ip.image_std)

    csma: Optional[CSMA] = None
    if args.input_mode == "pseudo_rgb":
        print(f"[eval_m3fd_yolo] CSMA ckpt: {args.ckpt}")
        csma = _load_csma(args.ckpt, cfg, device)

    dataset = M3FDIRDataset(args.ir_dir, args.gt_json, dino_proc, cfg)
    coco_gt = _build_m3fd_gt(dataset)
    print(f"[eval_m3fd_yolo] GT 框数: {len(coco_gt.anns)}")

    yolo = YOLO(args.yolo_weights)
    predictions = run_yolo_m3fd_eval(
        yolo_model=yolo,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        conf=args.conf,
        input_mode=args.input_mode,
        csma=csma,
        gdino_mean=gdino_mean,
        gdino_std=gdino_std,
    )
    print(f"[eval_m3fd_yolo] 预测框数: {len(predictions)}")

    results = compute_map(coco_gt, predictions)
    results.update({
        "yolo_weights": args.yolo_weights,
        "input_mode": args.input_mode,
        "ckpt": args.ckpt or None,
        "dataset": "M3FD",
        "split": "zxSplit_test",
        "eval_classes": "person,car",
        "conf": args.conf,
    })

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 55)
    print(f"  dataset       : M3FD (person+car, zxSplit test)")
    print(f"  input_mode    : {args.input_mode}")
    print(f"  mAP@0.5       : {results['map_50']:.4f}")
    print(f"  AP_person@0.5 : {results['ap_person']:.4f}")
    print(f"  AP_car@0.5    : {results['ap_car']:.4f}")
    print(f"  预测框 / GT框  : {results['n_preds']} / {results['n_gt']}")
    print("=" * 55)
    print(f"[eval_m3fd_yolo] 结果: {args.out_json}")


if __name__ == "__main__":
    main()
