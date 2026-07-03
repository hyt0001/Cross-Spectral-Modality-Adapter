"""
DETR (DEtection TRansformer, Facebook AI 2020) 检测基线 + **DETR-CSMA** 伪 RGB 评估。

Teacher 匹配：pseudo_rgb 必须使用 DETR teacher 训练的 CSMA（outputs_csma_detr*）。
GDINO-CSMA 请用 src.eval_csma（GDINO downstream）。

支持两种输入模式：
  ir_raw     — 直接送原始 IR 图（3 通道灰度），零样本基线
  pseudo_rgb — CSMA 伪 RGB → DETR，测试 plug-and-play 效果

CLI 示例：
    # 基线
    HF_ENDPOINT=https://hf-mirror.com \\
    conda run -n RGBtest python -m src.eval_detr_baseline \\
        --data-root FLIR_License/val --out-json outputs_csma/logs/eval_detr_ir_raw.json

    # CSMA + DETR（teacher 匹配）
    conda run -n RGBtest python -m src.eval_detr_baseline \\
        --input-mode pseudo_rgb --ckpt outputs_csma_detr_base/ckpt/best_stage1.pt \\
        --out-json outputs_csma/logs/eval_detr_pseudo_rgb.json
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

from transformers import AutoProcessor, DetrForObjectDetection, DetrImageProcessor

from src.config import CSMAConfig
from src.csma import CSMA
from src.eval_csma import _build_gt_coco, compute_map
from src.dataset_flir_v1 import FlirV1PairedDataset, build_flir_v1_category_map, collate_flir_v1
from src.eval_llvip_detr import DETR_CSMA_CKPT_DEFAULT, _assert_detr_teacher_ckpt
from src.eval_yolo_csma import _configure_processor
from src.infer_vis import denormalize_pixel_values

# DETR COCO label id → FLIR eval category_id
# COCO: 1=person, 3=car  →  FLIR eval: 1=person, 2=car
DETR_LABEL_TO_EVAL_CAT: Dict[int, int] = {1: 1, 3: 2}

MODEL_ID = "facebook/detr-resnet-50"


def _load_csma(ckpt: str, cfg: CSMAConfig, device: torch.device) -> CSMA:
    csma = CSMA(cfg).to(device)
    raw = torch.load(ckpt, map_location=device, weights_only=True)
    state = raw["csma"] if isinstance(raw, dict) and "csma" in raw else raw
    csma.load_state_dict(state)
    csma.eval()
    return csma


def _pseudo_rgb_to_pil(
    tensor_chw: torch.Tensor,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    oh: int,
    ow: int,
) -> Image.Image:
    """CSMA 输出的归一化 tensor → PIL RGB，裁至有效区域。"""
    arr = denormalize_pixel_values(tensor_chw.cpu(), mean, std)  # H×W×3 uint8
    return Image.fromarray(arr[:oh, :ow])


def run_detr_eval(
    model: DetrForObjectDetection,
    processor: DetrImageProcessor,
    dataset: FlirV1PairedDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    threshold: float,
    input_mode: str = "ir_raw",
    csma: Optional[CSMA] = None,
    gdino_processor: Optional[Any] = None,
) -> List[Dict]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_flir_v1,
        num_workers=num_workers,
    )
    path_to_id: Dict[str, int] = {
        os.path.join(dataset._root, img["file_name"]): int(img["id"])
        for img in dataset._images
    }

    if input_mode == "pseudo_rgb":
        assert csma is not None and gdino_processor is not None
        ip = gdino_processor.image_processor
        mean: Tuple = tuple(ip.image_mean)
        std:  Tuple = tuple(ip.image_std)

    model.eval()
    predictions: List[Dict] = []
    total = len(dataset)
    processed = 0

    with torch.no_grad():
        for batch in loader:
            img_paths: List[str] = batch["image_paths"]

            if input_mode == "pseudo_rgb":
                ir_pv = batch["pixel_values"].to(device)
                pm    = batch["pixel_mask"]
                pseudo_batch = csma(ir_pv)
                pil_images = []
                for i in range(len(img_paths)):
                    oh = int(pm[i][:, 0].sum().item())
                    ow = int(pm[i][0, :].sum().item())
                    pil_images.append(_pseudo_rgb_to_pil(pseudo_batch[i], mean, std, oh, ow))
            else:
                pil_images = [Image.open(p).convert("RGB") for p in img_paths]

            inputs = processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            target_sizes = torch.tensor(
                [img.size[::-1] for img in pil_images],
                dtype=torch.float32,
                device=device,
            )
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=threshold
            )

            for res, img_path in zip(results, img_paths):
                img_id = path_to_id.get(img_path, 0)
                boxes  = res["boxes"].cpu().numpy()
                scores = res["scores"].cpu().numpy()
                labels = res["labels"].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    cat_id = DETR_LABEL_TO_EVAL_CAT.get(int(label))
                    if cat_id is None:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    predictions.append({
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    })

            processed += len(img_paths)
            print(f"\r  DETR 推理进度: {processed}/{total}", end="", flush=True)

    print()
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DETR (ResNet-50) 检测基线 + CSMA 伪 RGB 评估（FLIR val）"
    )
    parser.add_argument("--model-id",    type=str, default=MODEL_ID)
    parser.add_argument("--data-root",   type=str, default="FLIR_License/val")
    parser.add_argument(
        "--input-mode", type=str, default="ir_raw", choices=["ir_raw", "pseudo_rgb"],
        help="ir_raw=原始 IR 基线；pseudo_rgb=CSMA 伪 RGB（需提供 --ckpt）",
    )
    parser.add_argument("--ckpt",        type=str, default=DETR_CSMA_CKPT_DEFAULT,
                        help="DETR-CSMA 权重（须为 outputs_csma_detr* 目录）")
    parser.add_argument("--out-json",    type=str, default="outputs_csma/logs/eval_detr_ir_raw.json")
    parser.add_argument("--batch-size",  type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold",   type=float, default=0.05)
    args = parser.parse_args()

    if args.input_mode == "pseudo_rgb":
        _assert_detr_teacher_ckpt(args.ckpt)
        if not os.path.isfile(args.ckpt):
            raise FileNotFoundError(f"DETR-CSMA 权重不存在: {args.ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_detr] 设备:        {device}")
    print(f"[eval_detr] model_id:    {args.model_id}")
    print(f"[eval_detr] input_mode:  {args.input_mode}")
    print(f"[eval_detr] threshold:   {args.threshold}")

    print("[eval_detr] 加载 DETR...")
    detr_proc = DetrImageProcessor.from_pretrained(args.model_id)
    model = DetrForObjectDetection.from_pretrained(args.model_id).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("[eval_detr] DETR 加载完成")

    cfg = CSMAConfig()
    dino_proc = AutoProcessor.from_pretrained(cfg.model_id)
    # 与 eval_yolo_csma 保持一致：限制 processor 最短边 = cfg.img_size(512)，
    # 避免 FLIR 图被放大（640×512 默认会被 processor 放到 1000×800），
    # 使 pseudo_rgb 坐标系和 GT 标注坐标系一致。
    _configure_processor(dino_proc, cfg.img_size)

    csma: Optional[CSMA] = None
    if args.input_mode == "pseudo_rgb":
        print(f"[eval_detr] 加载 CSMA: {args.ckpt}")
        csma = _load_csma(args.ckpt, cfg, device)
        print("[eval_detr] CSMA 加载完成")

    cat_map, valid_ids = build_flir_v1_category_map("person. car.")
    dataset = FlirV1PairedDataset(
        root=args.data_root,
        processor=dino_proc,
        text_prompt="person. car.",
        category_map=cat_map,
        valid_cat_ids=valid_ids,
    )
    print(f"[eval_detr] val 集: {len(dataset)} 张")

    coco_gt = _build_gt_coco(dataset, valid_ids)
    print(f"[eval_detr] GT 框数: {len(coco_gt.anns)}")

    print("[eval_detr] 开始推理...")
    predictions = run_detr_eval(
        model=model,
        processor=detr_proc,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        input_mode=args.input_mode,
        csma=csma,
        gdino_processor=dino_proc,
    )
    print(f"[eval_detr] 预测框数: {len(predictions)}")

    results = compute_map(coco_gt, predictions)
    results.update({
        "model_id":   args.model_id,
        "input_mode": args.input_mode,
        "ckpt":       args.ckpt or None,
        "threshold":  args.threshold,
        "data_root":  args.data_root,
    })

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 55)
    print(f"  model         : {args.model_id}")
    print(f"  input_mode    : {args.input_mode}")
    print(f"  mAP@0.5       : {results['map_50']:.4f}")
    print(f"  mAP@0.5:0.95  : {results['map_50_95']:.4f}")
    print(f"  AP_person@0.5 : {results['ap_person']:.4f}")
    print(f"  AP_car@0.5    : {results['ap_car']:.4f}")
    print(f"  预测框 / GT框  : {results['n_preds']} / {results['n_gt']}")
    print("=" * 55)
    print(f"[eval_detr] 结果: {args.out_json}")


if __name__ == "__main__":
    main()
