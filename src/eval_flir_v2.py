"""
FLIR_ADAS_v2 热红外检测评估模块。

在 thermal_val 上计算 mAP@0.5（COCO 标准），输出 per-class AP + mAP。
使用 pycocotools 进行严格 IoU 匹配，保证与 COCO benchmark 结果可比。

CLI 用法：
    conda run -n RGBtest python -m src.eval_flir_v2 \\
        --ckpt outputs_csma/ckpt/csma_last.pt \\
        --data-root FLIR_ADAS_v2/images_thermal_val \\
        --out-json outputs_csma/logs/eval_last.json

对应 docs/TD.md §3.2 评估指标。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from src.config import CSMAConfig
from src.csma import CSMA
from src.dataset_flir_v2 import (
    FlirADASV2Dataset,
    build_flir_v2_category_map,
    collate_flir_v2,
)


# ── 常量 ──────────────────────────────────────────────────────────────────────
# FLIR_ADAS_v2 用于评估的类别（与训练 prompt 一致）
EVAL_TEXT_PROMPT: str = "person. car."
# pycocotools 需要的类别 id（连续，从 1 开始）
EVAL_CATEGORIES: List[Dict] = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "car"},
]
# FLIR v2 cat_id → 评估用 cat_id（person=1 → 1, car=3 → 2）
FLIR_TO_EVAL_CAT: Dict[int, int] = {1: 1, 3: 2}
# 模型输出 class_idx → 评估用 cat_id（class_idx 0=person → 1, 1=car → 2）
CLASS_IDX_TO_EVAL_CAT: Dict[int, int] = {0: 1, 1: 2}


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: 构建 GT COCO 对象（pycocotools 格式）
# ──────────────────────────────────────────────────────────────────────────────

def _build_gt_coco(
    dataset: FlirADASV2Dataset,
    valid_cat_ids: frozenset,
) -> COCO:
    """
    将 FlirADASV2Dataset 的 GT 标注转换为 pycocotools COCO 对象。

    Args:
        dataset:       已初始化的 FlirADASV2Dataset。
        valid_cat_ids: FLIR v2 中有效 cat_id 集合（{1, 3}）。

    Returns:
        COCO 对象，包含 images / annotations / categories。
    """
    images: List[Dict] = []
    annotations: List[Dict] = []
    ann_id = 1

    for idx in range(len(dataset)):
        img_info = dataset._images[idx]
        img_id = int(img_info["id"])
        images.append(
            {
                "id": img_id,
                "width": int(img_info["width"]),
                "height": int(img_info["height"]),
                "file_name": img_info["file_name"],
            }
        )
        for ann in dataset._id_to_anns.get(img_id, []):
            cid = int(ann["category_id"])
            if cid not in valid_cat_ids:
                continue
            x, y, w, h = [float(v) for v in ann["bbox"]]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": FLIR_TO_EVAL_CAT[cid],
                    "bbox": [x, y, w, h],
                    "area": float(ann["area"]),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                }
            )
            ann_id += 1

    coco_dict: Dict[str, Any] = {
        "images": images,
        "annotations": annotations,
        "categories": EVAL_CATEGORIES,
    }
    coco_gt = COCO()
    coco_gt.dataset = coco_dict
    coco_gt.createIndex()
    return coco_gt


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: 推理收集预测框
# ──────────────────────────────────────────────────────────────────────────────

def run_eval(
    csma: CSMA,
    dino: GroundingDinoForObjectDetection,
    processor: Any,
    dataset: FlirADASV2Dataset,
    device: torch.device,
    text_prompt: str,
    batch_size: int = 4,
    num_workers: int = 2,
    box_threshold: float = 0.05,
    text_threshold: float = 0.05,
) -> List[Dict]:
    """
    在整个 val 集上推理，返回 pycocotools 格式的预测列表。

    box_threshold / text_threshold 设置较低（0.05），保证召回率，
    让 COCOeval 通过置信度阈值扫描计算 AP 曲线。

    Args:
        csma:             CSMA 适配器（eval 模式）。
        dino:             冻结 DINO（eval 模式）。
        processor:        HuggingFace AutoProcessor。
        dataset:          FlirADASV2Dataset（val split）。
        device:           计算设备。
        text_prompt:      检测文本提示，如 "person. car."
        batch_size:       推理 batch 大小。
        num_workers:      DataLoader 工作进程数。
        box_threshold:    检测框置信度阈值（低值保证召回）。
        text_threshold:   文本匹配阈值。

    Returns:
        List[Dict]，每个元素形如：
            {"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_flir_v2,
        num_workers=num_workers,
    )

    enc = processor.tokenizer(text_prompt, return_tensors="pt")
    input_ids_base = enc["input_ids"].to(device)
    attention_mask_base = enc["attention_mask"].to(device)

    csma.eval()
    dino.eval()
    predictions: List[Dict] = []

    total = len(dataset)
    processed = 0
    with torch.no_grad():
        for batch in loader:
            ir_pv = batch["pixel_values"].to(device)       # [B, 3, H, W]
            pm    = batch["pixel_mask"].to(device)          # [B, H, W]
            bsz   = ir_pv.shape[0]

            # CSMA：红外 → 伪 RGB
            pseudo_rgb = csma(ir_pv)                        # [B, 3, H, W]
            h, w = pseudo_rgb.shape[-2], pseudo_rgb.shape[-1]
            target_sizes = torch.tensor(
                [[h, w]] * bsz, dtype=torch.int64, device=device
            )

            # DINO 推理
            outputs = dino(
                pixel_values=pseudo_rgb,
                pixel_mask=pm,
                input_ids=input_ids_base.expand(bsz, -1),
                attention_mask=attention_mask_base.expand(bsz, -1),
            )

            # 后处理：每个样本独立解码
            results_list = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=input_ids_base,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )

            # 从 dataset 反查 image_id（通过 image_path 匹配）
            img_paths = batch["image_paths"]
            for i, (res, img_path) in enumerate(zip(results_list, img_paths)):
                # 查找该 image_path 对应的 FLIR image_id
                img_id = _path_to_image_id(img_path, dataset)

                boxes  = res["boxes"].cpu().numpy()    # [N, 4] xyxy 像素坐标
                scores = res["scores"].cpu().numpy()   # [N]
                labels = res["labels"]                 # List[str]

                for box, score, label in zip(boxes, scores, labels):
                    cat_id = _label_to_eval_cat(label)
                    if cat_id is None:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    x, y, bw, bh = x1, y1, x2 - x1, y2 - y1
                    predictions.append(
                        {
                            "image_id":   img_id,
                            "category_id": cat_id,
                            "bbox":        [x, y, bw, bh],
                            "score":       float(score),
                        }
                    )

            processed += bsz
            print(f"\r  推理进度: {processed}/{total}", end="", flush=True)

    print()
    return predictions


def _path_to_image_id(img_path: str, dataset: FlirADASV2Dataset) -> int:
    """
    通过图像路径反查 FLIR coco.json 中的 image_id。

    先尝试路径后缀精确匹配 file_name，未找到则降级为顺序 index。
    """
    # 构建 file_name → image_id 的映射（惰性，首次调用时建立）
    if not hasattr(dataset, "_path_to_id_cache"):
        dataset._path_to_id_cache = {  # type: ignore[attr-defined]
            os.path.join(dataset._root, img["file_name"]): img["id"]
            for img in dataset._images
        }
    return dataset._path_to_id_cache.get(img_path, 0)  # type: ignore[attr-defined]


def _label_to_eval_cat(label: str) -> int | None:
    """
    将 Grounding DINO 输出的文本标签映射到 eval category_id。

    Grounding DINO 输出 label 为 prompt 中的词组片段，做前缀匹配。
    """
    label = label.strip().lower()
    if label.startswith("person"):
        return 1
    if label.startswith("car"):
        return 2
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: 计算 mAP
# ──────────────────────────────────────────────────────────────────────────────

def compute_map(
    coco_gt: COCO,
    predictions: List[Dict],
    iou_type: str = "bbox",
) -> Dict[str, Any]:
    """
    使用 pycocotools COCOeval 计算 mAP@0.5 和 mAP@0.5:0.95。

    Args:
        coco_gt:      GT COCO 对象（由 _build_gt_coco 构建）。
        predictions:  run_eval 返回的预测列表。
        iou_type:     评估类型，默认 "bbox"。

    Returns:
        结果字典，包含：
            map_50       float  mAP@IoU=0.50
            map_50_95    float  mAP@IoU=0.50:0.95
            ap_person    float  person AP@0.5
            ap_car       float  car AP@0.5
            n_preds      int    总预测框数
            n_gt         int    总 GT 框数
    """
    if not predictions:
        print("  [警告] 无任何预测框，mAP=0")
        return {
            "map_50": 0.0, "map_50_95": 0.0,
            "ap_person": 0.0, "ap_car": 0.0,
            "n_preds": 0, "n_gt": len(coco_gt.anns),
        }

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, iou_type)

    # 仅评估 IoU=0.50（FLIR 论文惯例）和完整 0.50:0.95
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    # stats[0] = AP @IoU=0.50:0.95, stats[1] = AP @IoU=0.50
    map_50_95 = float(evaluator.stats[0])
    map_50    = float(evaluator.stats[1])

    # Per-class AP @0.5：单独跑每个类别
    def _per_class_ap(cat_id: int) -> float:
        ev = COCOeval(coco_gt, coco_dt, iou_type)
        ev.params.catIds = [cat_id]
        ev.params.iouThrs = np.array([0.5])
        ev.evaluate()
        ev.accumulate()
        ap = float(ev.stats[0])
        return max(ap, 0.0)

    ap_person = _per_class_ap(1)
    ap_car    = _per_class_ap(2)

    return {
        "map_50":     map_50,
        "map_50_95":  map_50_95,
        "ap_person":  ap_person,
        "ap_car":     ap_car,
        "n_preds":    len(predictions),
        "n_gt":       len(coco_gt.anns),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: CLI 入口
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FLIR_ADAS_v2 热红外检测评估：计算 CSMA mAP@0.5 on thermal_val"
    )
    parser.add_argument("--ckpt",        type=str, required=True,
                        help="CSMA 权重路径，如 outputs_csma/ckpt/csma_last.pt")
    parser.add_argument("--data-root",   type=str,
                        default="FLIR_ADAS_v2/images_thermal_val",
                        help="thermal_val 目录（含 coco.json + data/）")
    parser.add_argument("--out-json",    type=str,
                        default="outputs_csma/logs/eval_result.json",
                        help="评估结果输出 JSON 路径")
    parser.add_argument("--batch-size",  type=int, default=4,
                        help="推理 batch 大小（默认 4）")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader worker 数")
    parser.add_argument("--box-threshold",  type=float, default=0.05,
                        help="检测框置信度阈值（低值保证召回）")
    parser.add_argument("--text-threshold", type=float, default=0.05,
                        help="文本匹配阈值")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_flir_v2] 设备: {device}")
    print(f"[eval_flir_v2] 权重: {args.ckpt}")
    print(f"[eval_flir_v2] 数据: {args.data_root}")

    # Phase 4.1：加载处理器与 DINO
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    dino: GroundingDinoForObjectDetection = (
        GroundingDinoForObjectDetection.from_pretrained(model_id, local_files_only=True).to(device)
    )
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False
    print("[eval_flir_v2] DINO 已加载并冻结")

    # Phase 4.2：加载 CSMA 权重
    cfg = CSMAConfig()
    csma = CSMA(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    csma.load_state_dict(state)
    csma.eval()
    print(f"[eval_flir_v2] CSMA 权重已加载")

    # Phase 4.3：构建 val 数据集
    cat_map, valid_ids = build_flir_v2_category_map(EVAL_TEXT_PROMPT)
    dataset = FlirADASV2Dataset(
        root=args.data_root,
        processor=processor,
        text_prompt=EVAL_TEXT_PROMPT,
        category_map=cat_map,
        valid_cat_ids=valid_ids,
    )
    print(f"[eval_flir_v2] val 集大小: {len(dataset)} 张（含 person/car 标注）")

    # Phase 4.4：构建 GT COCO 对象
    coco_gt = _build_gt_coco(dataset, valid_ids)
    print(f"[eval_flir_v2] GT 标注总数: {len(coco_gt.anns)}")

    # Phase 4.5：推理
    print("[eval_flir_v2] 开始推理...")
    predictions = run_eval(
        csma=csma,
        dino=dino,
        processor=processor,
        dataset=dataset,
        device=device,
        text_prompt=EVAL_TEXT_PROMPT,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    print(f"[eval_flir_v2] 共生成 {len(predictions)} 个预测框")

    # Phase 4.6：计算 mAP
    print("[eval_flir_v2] 计算 mAP...")
    results = compute_map(coco_gt, predictions)

    # Phase 4.7：输出
    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    results["ckpt"] = args.ckpt
    results["data_root"] = args.data_root

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 50)
    print(f"  mAP@0.5       : {results['map_50']:.4f}")
    print(f"  mAP@0.5:0.95  : {results['map_50_95']:.4f}")
    print(f"  AP_person@0.5 : {results['ap_person']:.4f}")
    print(f"  AP_car@0.5    : {results['ap_car']:.4f}")
    print(f"  预测框 / GT框  : {results['n_preds']} / {results['n_gt']}")
    print("=" * 50)
    print(f"[eval_flir_v2] 结果已保存: {args.out_json}")


if __name__ == "__main__":
    main()
