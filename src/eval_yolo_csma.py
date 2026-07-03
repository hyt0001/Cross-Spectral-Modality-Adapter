"""
CSMA + YOLO / YOLO-World 评估（Exp-4：伪 RGB → 冻结检测器）。

在 val 集上：红外 → CSMA 伪 RGB（或原图 IR 基线）→ Ultralytics YOLOv8 / YOLO-World → COCO mAP。
GT 与 ``eval_csma`` 相同，便于与 Grounding DINO 主实验对比。

CLI 示例：
    conda run -n RGBtest python -m src.eval_yolo_csma \\
        --ckpt outputs_csma/ckpt/csma_last.pt \\
        --yolo-weights /root/autodl-tmp/yolov8m.pt \\
        --dataset flir_v1 \\
        --data-root FLIR_License/val \\
        --input-mode pseudo_rgb \\
        --out-json outputs_csma/logs/eval_yolo_pseudo_rgb.json

    # Exp-4 基线 4-D0：原 IR 直接送 YOLO（无需 CSMA）
    conda run -n RGBtest python -m src.eval_yolo_csma \\
        --yolo-weights /root/autodl-tmp/yolov8m.pt \\
        --input-mode ir_raw \\
        --out-json outputs_csma/logs/eval_yolo_ir_raw.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from src.config import CSMAConfig
from src.csma import CSMA
from src.eval_csma import (
    _build_gt_coco,
    _get_collate,
    _load_dataset,
    compute_map,
)
from src.infer_vis import denormalize_pixel_values

InputMode = Literal["pseudo_rgb", "ir_raw"]

# COCO 预训练 YOLOv8：person=0, car=2 → FLIR eval category_id
YOLO_CLS_TO_EVAL_CAT: Dict[int, int] = {0: 1, 2: 2}

# FLIR eval category_id：person=1, car=2
FLIR_EVAL_CAT: Dict[str, int] = {"person": 1, "car": 2}


def _is_yolo_world(weights_path: str) -> bool:
    return "world" in os.path.basename(weights_path).lower()


def _parse_text_classes(text_classes: str) -> List[str]:
    return [c.strip() for c in text_classes.split(",") if c.strip()]


def _build_cls_to_eval_cat(
    weights_path: str,
    text_classes: List[str],
) -> Tuple[Dict[int, int], List[str]]:
    """返回 (cls_id→FLIR category_id, 实际用于 World 的 class 名列表)。"""
    if _is_yolo_world(weights_path):
        mapping = {
            i: FLIR_EVAL_CAT[name]
            for i, name in enumerate(text_classes)
            if name in FLIR_EVAL_CAT
        }
        return mapping, text_classes
    return YOLO_CLS_TO_EVAL_CAT, text_classes


def _configure_yolo_world(yolo_model: Any, text_classes: List[str]) -> None:
    if not hasattr(yolo_model, "set_classes"):
        raise RuntimeError("当前 ultralytics 不支持 YOLO-World set_classes，请升级 ultralytics>=8.3")
    yolo_model.set_classes(text_classes)
    print(f"[eval_yolo_csma] YOLO-World set_classes: {text_classes}")


def _load_csma(ckpt: str, cfg: CSMAConfig, device: torch.device) -> CSMA:
    csma = CSMA(cfg).to(device)
    raw = torch.load(ckpt, map_location=device, weights_only=True)
    state = raw["csma"] if isinstance(raw, dict) and "csma" in raw else raw
    csma.load_state_dict(state)
    csma.eval()
    return csma


def _configure_processor(processor: Any, img_size: int) -> None:
    if not (hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size")):
        return
    ip = processor.image_processor
    try:
        cur_se = ip.size.shortest_edge or 0
    except AttributeError:
        cur_se = ip.size.get("shortest_edge", 0) or 0
    if cur_se > img_size:
        try:
            ip.size.shortest_edge = img_size
            ip.size.longest_edge = img_size * 2
        except AttributeError:
            ip.size = {"shortest_edge": img_size, "longest_edge": img_size * 2}
        print(f"[eval_yolo_csma] processor shortest_edge={img_size}")


def _valid_hw(pixel_mask: torch.Tensor) -> Tuple[int, int]:
    """由 pixel_mask 反推有效区域高宽（与 eval_csma target_sizes 一致）。"""
    oh = int(pixel_mask[:, 0].sum().item())
    ow = int(pixel_mask[0, :].sum().item())
    return oh, ow


def _pseudo_rgb_uint8(
    tensor_chw: torch.Tensor,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    oh: int,
    ow: int,
) -> np.ndarray:
    img = denormalize_pixel_values(tensor_chw.cpu(), mean, std)
    return img[:oh, :ow]


def run_yolo_eval(
    yolo_model: Any,
    dataset: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    conf: float,
    input_mode: InputMode,
    dataset_mode: str,
    cls_to_eval_cat: Dict[int, int],
    csma: Optional[CSMA] = None,
    processor: Optional[Any] = None,
) -> List[Dict]:
    """
    在 val 集上收集 YOLO 预测，返回 pycocotools detection 列表。
    """
    if input_mode == "pseudo_rgb":
        if csma is None or processor is None:
            raise ValueError("pseudo_rgb 模式需要 csma 与 processor")
    elif input_mode != "ir_raw":
        raise ValueError(f"不支持的 input_mode: {input_mode}")

    collate_fn = _get_collate(dataset_mode)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    path_to_id: Dict[str, int] = {
        os.path.join(dataset._root, img["file_name"]): int(img["id"])
        for img in dataset._images
    }

    mean = tuple(processor.image_processor.image_mean) if processor else (0.485, 0.456, 0.406)
    std = tuple(processor.image_processor.image_std) if processor else (0.229, 0.224, 0.225)

    if csma is not None:
        csma.eval()

    predictions: List[Dict] = []
    total = len(dataset)
    processed = 0

    with torch.no_grad():
        for batch in loader:
            ir_pv = batch["pixel_values"]
            pm = batch["pixel_mask"]
            bsz = ir_pv.shape[0]
            img_paths: List[str] = batch["image_paths"]

            batch_images: List[np.ndarray] = []
            batch_img_ids: List[int] = []

            if input_mode == "pseudo_rgb":
                assert csma is not None
                ir_dev = ir_pv.to(device)
                pseudo_batch = csma(ir_dev)
                for i in range(bsz):
                    oh, ow = _valid_hw(pm[i])
                    batch_images.append(
                        _pseudo_rgb_uint8(pseudo_batch[i], mean, std, oh, ow)
                    )
                    batch_img_ids.append(path_to_id.get(img_paths[i], 0))
            else:
                for i in range(bsz):
                    pil = Image.open(img_paths[i]).convert("RGB")
                    batch_images.append(np.asarray(pil, dtype=np.uint8))
                    batch_img_ids.append(path_to_id.get(img_paths[i], 0))

            results_list = yolo_model.predict(
                source=batch_images,
                conf=conf,
                verbose=False,
            )

            for res, img_id in zip(results_list, batch_img_ids):
                if res.boxes is None or len(res.boxes) == 0:
                    continue
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                for box, score, cls_id in zip(xyxy, confs, clss):
                    cat_id = cls_to_eval_cat.get(int(cls_id))
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
    parser = argparse.ArgumentParser(
        description="CSMA + YOLO / YOLO-World mAP 评估（Exp-4 pseudo_rgb / ir_raw 基线）"
    )
    parser.add_argument("--ckpt", type=str, default="",
                        help="CSMA 权重；ir_raw 模式可省略")
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="/root/autodl-tmp/yolov8m.pt",
        help="YOLOv8-m 权重路径",
    )
    parser.add_argument("--dataset", type=str, default="flir_v1", choices=["flir_v1", "flir_v2"])
    parser.add_argument("--data-root", type=str, default="FLIR_License/val")
    parser.add_argument(
        "--input-mode",
        type=str,
        default="pseudo_rgb",
        choices=["pseudo_rgb", "ir_raw"],
        help="pseudo_rgb=CSMA 伪 RGB（4-D4）；ir_raw=原 IR 图（4-D0 基线）",
    )
    parser.add_argument("--out-json", type=str, default="outputs_csma/logs/eval_yolo.json")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="DataLoader batch；YOLO 按 batch 推理")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.05,
                        help="YOLO 置信度阈值（与 eval_csma 低阈值策略一致）")
    parser.add_argument(
        "--text-classes",
        type=str,
        default="person,car",
        help="YOLO-World set_classes，逗号分隔（与 GDINO text_prompt 对齐）",
    )
    args = parser.parse_args()

    if args.input_mode == "pseudo_rgb" and not args.ckpt:
        parser.error("pseudo_rgb 模式必须提供 --ckpt")
    if not os.path.isfile(args.yolo_weights):
        raise FileNotFoundError(f"YOLO 权重不存在: {args.yolo_weights}")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("请安装 ultralytics: pip install ultralytics") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CSMAConfig()
    model_id = cfg.model_id

    print(f"[eval_yolo_csma] 设备:       {device}")
    print(f"[eval_yolo_csma] 输入模式:   {args.input_mode}")
    print(f"[eval_yolo_csma] YOLO:       {args.yolo_weights}")
    print(f"[eval_yolo_csma] 数据集:     {args.dataset}  {args.data_root}")

    processor = AutoProcessor.from_pretrained(model_id)
    _configure_processor(processor, cfg.img_size)

    csma: Optional[CSMA] = None
    if args.input_mode == "pseudo_rgb":
        print(f"[eval_yolo_csma] CSMA ckpt:  {args.ckpt}")
        csma = _load_csma(args.ckpt, cfg, device)
        print("[eval_yolo_csma] CSMA 已加载")

    text_classes = _parse_text_classes(args.text_classes)
    cls_to_eval_cat, world_classes = _build_cls_to_eval_cat(args.yolo_weights, text_classes)

    yolo = YOLO(args.yolo_weights)
    if _is_yolo_world(args.yolo_weights):
        _configure_yolo_world(yolo, world_classes)
        print(f"[eval_yolo_csma] cls→eval_cat: {cls_to_eval_cat}")

    dataset, valid_ids = _load_dataset(args.dataset, args.data_root, processor)
    print(f"[eval_yolo_csma] val 集: {len(dataset)} 张")

    coco_gt = _build_gt_coco(dataset, valid_ids)
    print(f"[eval_yolo_csma] GT 框数: {len(coco_gt.anns)}")

    print("[eval_yolo_csma] 开始 YOLO 推理...")
    predictions = run_yolo_eval(
        yolo_model=yolo,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        conf=args.conf,
        input_mode=args.input_mode,  # type: ignore[arg-type]
        dataset_mode=args.dataset,
        cls_to_eval_cat=cls_to_eval_cat,
        csma=csma,
        processor=processor,
    )
    print(f"[eval_yolo_csma] 预测框数: {len(predictions)}")

    print("[eval_yolo_csma] 计算 mAP...")
    results = compute_map(coco_gt, predictions)
    results["ckpt"] = args.ckpt or None
    results["yolo_weights"] = args.yolo_weights
    results["input_mode"] = args.input_mode
    results["dataset"] = args.dataset
    results["data_root"] = args.data_root
    results["conf"] = args.conf
    results["text_classes"] = text_classes if _is_yolo_world(args.yolo_weights) else None

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 50)
    print(f"  input_mode    : {args.input_mode}")
    print(f"  mAP@0.5       : {results['map_50']:.4f}")
    print(f"  mAP@0.5:0.95  : {results['map_50_95']:.4f}")
    print(f"  AP_person@0.5 : {results['ap_person']:.4f}")
    print(f"  AP_car@0.5    : {results['ap_car']:.4f}")
    print(f"  预测框 / GT框  : {results['n_preds']} / {results['n_gt']}")
    print("=" * 50)
    print(f"[eval_yolo_csma] 结果: {args.out_json}")


if __name__ == "__main__":
    main()
