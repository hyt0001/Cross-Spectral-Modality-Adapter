"""
M3FD 数据集上 DETR 检测评估（ir_raw 基线 + CSMA 伪 RGB）。

评估 person(People) + car(Car)，M3FD-zxSplit test 集。

Teacher 匹配：仅使用 DETR-CSMA 权重（outputs_csma_detr*），
GDINO-CSMA 不得用于本脚本。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, DetrForObjectDetection, DetrImageProcessor

from src.config import CSMAConfig
from src.csma import CSMA
from src.eval_csma import compute_map
from src.eval_llvip_detr import (
    DETR_CSMA_CKPT_DEFAULT,
    _assert_detr_teacher_ckpt,
)
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
from torch.utils.data import DataLoader

# COCO DETR: 1=person, 3=car → eval cat 1, 2
DETR_LABEL_TO_M3FD_EVAL: Dict[int, int] = {1: 1, 3: 2}


def _load_csma(ckpt: str, cfg: CSMAConfig, device: torch.device) -> CSMA:
    csma = CSMA(cfg).to(device)
    raw = torch.load(ckpt, map_location=device, weights_only=True)
    state = raw["csma"] if isinstance(raw, dict) and "csma" in raw else raw
    csma.load_state_dict(state)
    csma.eval()
    return csma


def run_eval(
    detr_model: DetrForObjectDetection,
    detr_processor: DetrImageProcessor,
    dataset: M3FDIRDataset,
    device: torch.device,
    batch_size: int,
    threshold: float,
    input_mode: str,
    csma: Optional[CSMA],
    gdino_mean: Tuple,
    gdino_std: Tuple,
) -> List[Dict]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_m3fd, num_workers=2,
    )
    detr_model.eval()
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

            if input_mode == "pseudo_rgb":
                ir_pv = batch["pixel_values"].to(device)
                pm = batch["pixel_masks"]
                pil_images = []
                for i in range(bsz):
                    pv_i, pm_i = pad_for_csma(
                        ir_pv[i : i + 1], pm[i], multiple=8,
                    )
                    pseudo = csma(pv_i)
                    oh = int(pm[i][:, 0].sum().item())
                    ow = int(pm[i][0, :].sum().item())
                    arr = denormalize_pixel_values(pseudo[0].cpu(), gdino_mean, gdino_std)
                    arr = arr[:oh, :ow]
                    pil_pseudo = Image.fromarray(arr)
                    pil_pseudo = pil_pseudo.resize((orig_ws[i], orig_hs[i]), Image.BILINEAR)
                    pil_images.append(pil_pseudo)
            else:
                pil_images = [
                    Image.open(os.path.join(dataset._ir_dir, fn)).convert("RGB")
                    for fn in file_names
                ]

            inputs = detr_processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = detr_model(**inputs)

            target_sizes = torch.tensor(
                [[img.size[1], img.size[0]] for img in pil_images],
                dtype=torch.float32, device=device,
            )
            results = detr_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=threshold,
            )

            for res, img_id in zip(results, image_ids):
                boxes = res["boxes"].cpu().numpy()
                scores = res["scores"].cpu().numpy()
                labels = res["labels"].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    cat_id = DETR_LABEL_TO_M3FD_EVAL.get(int(label))
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
            print(f"\r  DETR 推理进度: {processed}/{total}", end="", flush=True)

    print()
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="M3FD 上 DETR + CSMA 评估")
    parser.add_argument("--ir-dir", type=str, default=M3FD_IR_DIR)
    parser.add_argument("--gt-json", type=str, default=M3FD_GT_JSON)
    parser.add_argument("--model-id", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--input-mode", type=str, default="ir_raw",
                        choices=["ir_raw", "pseudo_rgb"])
    parser.add_argument("--ckpt", type=str, default=DETR_CSMA_CKPT_DEFAULT)
    parser.add_argument("--out-json", type=str,
                        default="outputs_csma/logs/eval_m3fd_detr_ir_raw.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--residual-scale", type=float, default=None,
                        help="覆盖 CSMAConfig.residual_scale（默认 1.0）")
    args = parser.parse_args()

    if args.input_mode == "pseudo_rgb":
        _assert_detr_teacher_ckpt(args.ckpt)
        if not os.path.isfile(args.ckpt):
            raise FileNotFoundError(f"DETR-CSMA 权重不存在: {args.ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_m3fd_detr] 设备:       {device}")
    print(f"[eval_m3fd_detr] input_mode: {args.input_mode}")

    cfg = CSMAConfig()
    if args.residual_scale is not None:
        cfg.residual_scale = args.residual_scale
        print(f"[eval_m3fd_detr] residual_scale 覆盖: {args.residual_scale}")
    dino_proc = AutoProcessor.from_pretrained(cfg.model_id)
    _configure_processor(dino_proc, cfg.img_size)
    ip = dino_proc.image_processor
    gdino_mean = tuple(ip.image_mean)
    gdino_std = tuple(ip.image_std)

    print("[eval_m3fd_detr] 加载 DETR...")
    detr_proc = DetrImageProcessor.from_pretrained(args.model_id)
    detr_model = DetrForObjectDetection.from_pretrained(args.model_id).to(device)
    detr_model.eval()
    for p in detr_model.parameters():
        p.requires_grad = False

    csma: Optional[CSMA] = None
    if args.input_mode == "pseudo_rgb":
        print(f"[eval_m3fd_detr] 加载 CSMA: {args.ckpt}")
        csma = _load_csma(args.ckpt, cfg, device)

    dataset = M3FDIRDataset(args.ir_dir, args.gt_json, dino_proc, cfg)
    coco_gt = _build_m3fd_gt(dataset)
    print(f"[eval_m3fd_detr] GT 框数: {len(coco_gt.anns)}")

    predictions = run_eval(
        detr_model=detr_model,
        detr_processor=detr_proc,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        threshold=args.threshold,
        input_mode=args.input_mode,
        csma=csma,
        gdino_mean=gdino_mean,
        gdino_std=gdino_std,
    )
    print(f"[eval_m3fd_detr] 预测框数: {len(predictions)}")

    results = compute_map(coco_gt, predictions)
    results.update({
        "model_id": args.model_id,
        "input_mode": args.input_mode,
        "ckpt": args.ckpt or None,
        "dataset": "M3FD",
        "split": "zxSplit_test",
        "eval_classes": "person,car",
        "threshold": args.threshold,
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
    print(f"[eval_m3fd_detr] 结果: {args.out_json}")


if __name__ == "__main__":
    main()
