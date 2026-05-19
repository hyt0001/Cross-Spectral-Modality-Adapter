"""
CSMA 通用检测评估模块（支持 flir_v1 / flir_v2 / legacy）。

在 val 集上计算 mAP@0.5（COCO 标准），输出 per-class AP + mAP。
使用 pycocotools 进行严格 IoU 匹配，保证与 COCO benchmark 结果可比。

CLI 用法：
    # FLIR v1（默认，配对数据集）
    conda run -n RGBtest python -m src.eval_csma \\
        --ckpt outputs_csma/ckpt/csma_last.pt \\
        --dataset flir_v1 \\
        --data-root FLIR_License/val \\
        --out-json outputs_csma/logs/eval_last.json

    # FLIR_ADAS_v2（无配对）
    conda run -n RGBtest python -m src.eval_csma \\
        --ckpt outputs_csma/ckpt/csma_last.pt \\
        --dataset flir_v2 \\
        --data-root FLIR_ADAS_v2/images_thermal_val \\
        --out-json outputs_csma/logs/eval_last.json

对应 docs/TD.md §3.2 评估指标。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from src.config import CSMAConfig
from src.csma import CSMA


# ── 常量 ──────────────────────────────────────────────────────────────────────
EVAL_TEXT_PROMPT: str = "person. car."
EVAL_CATEGORIES: List[Dict] = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "car"},
]
# 所有支持数据集的 raw cat_id → eval cat_id（person=1, car=2）
# flir_v1: person=1→1, car=3→2
# flir_v2: person=1→1, car=3→2
FLIR_TO_EVAL_CAT: Dict[int, int] = {1: 1, 3: 2}


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: 数据集加载（三种模式）
# ──────────────────────────────────────────────────────────────────────────────

def _load_dataset(dataset_mode: str, data_root: str, processor: Any):
    """
    根据 dataset_mode 加载对应 val 数据集。

    Returns:
        (dataset, valid_cat_ids)
    """
    if dataset_mode == "flir_v1":
        from src.dataset_flir_v1 import FlirV1PairedDataset, build_flir_v1_category_map
        cat_map, valid_ids = build_flir_v1_category_map(EVAL_TEXT_PROMPT)
        dataset = FlirV1PairedDataset(
            root=data_root,
            processor=processor,
            text_prompt=EVAL_TEXT_PROMPT,
            category_map=cat_map,
            valid_cat_ids=valid_ids,
        )
        return dataset, valid_ids

    elif dataset_mode == "flir_v2":
        from src.dataset_flir_v2 import FlirADASV2Dataset, build_flir_v2_category_map
        cat_map, valid_ids = build_flir_v2_category_map(EVAL_TEXT_PROMPT)
        dataset = FlirADASV2Dataset(
            root=data_root,
            processor=processor,
            text_prompt=EVAL_TEXT_PROMPT,
            category_map=cat_map,
            valid_cat_ids=valid_ids,
        )
        return dataset, valid_ids

    else:
        raise ValueError(f"不支持的 dataset_mode: {dataset_mode}，请使用 flir_v1 或 flir_v2")


def _get_collate(dataset_mode: str):
    if dataset_mode == "flir_v1":
        from src.dataset_flir_v1 import collate_flir_v1
        return collate_flir_v1
    elif dataset_mode == "flir_v2":
        from src.dataset_flir_v2 import collate_flir_v2
        return collate_flir_v2
    else:
        raise ValueError(f"不支持的 dataset_mode: {dataset_mode}")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: 构建 GT COCO 对象
# ──────────────────────────────────────────────────────────────────────────────

def _build_gt_coco(dataset, valid_cat_ids: frozenset) -> COCO:
    """
    将数据集 GT 标注转换为 pycocotools COCO 对象。

    兼容 FlirV1PairedDataset 和 FlirADASV2Dataset（统一接口：
    _images, _id_to_anns, _root, _images[i]["file_name"]）。
    """
    images: List[Dict] = []
    annotations: List[Dict] = []
    ann_id = 1

    for img_info in dataset._images:
        img_id = int(img_info["id"])
        images.append({
            "id":        img_id,
            "width":     int(img_info["width"]),
            "height":    int(img_info["height"]),
            "file_name": img_info["file_name"],
        })
        for ann in dataset._id_to_anns.get(img_id, []):
            cid = int(ann["category_id"])
            if cid not in valid_cat_ids:
                continue
            x, y, w, h = [float(v) for v in ann["bbox"]]
            annotations.append({
                "id":          ann_id,
                "image_id":    img_id,
                "category_id": FLIR_TO_EVAL_CAT[cid],
                "bbox":        [x, y, w, h],
                "area":        float(ann["area"]),
                "iscrowd":     int(ann.get("iscrowd", 0)),
            })
            ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": EVAL_CATEGORIES,
    }
    coco_gt.createIndex()
    return coco_gt


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: 推理收集预测框
# ──────────────────────────────────────────────────────────────────────────────

def run_eval(
    csma: CSMA,
    dino: GroundingDinoForObjectDetection,
    processor: Any,
    dataset,
    device: torch.device,
    batch_size: int = 4,
    num_workers: int = 2,
    box_threshold: float = 0.05,
    text_threshold: float = 0.05,
    dataset_mode: str = "flir_v1",
) -> List[Dict]:
    """
    在整个 val 集上推理，返回 pycocotools 格式的预测列表。

    box/text_threshold 设置较低（0.05），保证召回率，
    由 COCOeval 通过置信度阈值扫描计算 AP 曲线。
    """
    collate_fn = _get_collate(dataset_mode)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    enc = processor.tokenizer(EVAL_TEXT_PROMPT, return_tensors="pt")
    input_ids_base = enc["input_ids"].to(device)
    attention_mask_base = enc["attention_mask"].to(device)

    # 预构建 image_path → image_id 索引
    path_to_id: Dict[str, int] = {
        os.path.join(dataset._root, img["file_name"]): img["id"]
        for img in dataset._images
    }

    csma.eval()
    dino.eval()
    predictions: List[Dict] = []

    total = len(dataset)
    processed = 0
    with torch.no_grad():
        for batch in loader:
            ir_pv = batch["pixel_values"].to(device)
            pm    = batch["pixel_mask"].to(device)
            bsz   = ir_pv.shape[0]

            pseudo_rgb = csma(ir_pv)
            # target_sizes 必须用原始图像尺寸（GT 坐标系），而非处理后张量尺寸。
            # pixel_mask=1 表示有效像素，通过 mask 的行/列 sum 反推原始 H/W。
            orig_h = pm[:, :, 0].sum(dim=1).long()   # [B]
            orig_w = pm[:, 0, :].sum(dim=1).long()   # [B]
            target_sizes = torch.stack([orig_h, orig_w], dim=1).to(device)  # [B,2]

            outputs = dino(
                pixel_values=pseudo_rgb,
                pixel_mask=pm,
                input_ids=input_ids_base.expand(bsz, -1),
                attention_mask=attention_mask_base.expand(bsz, -1),
            )

            results_list = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=input_ids_base,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )

            img_paths = batch["image_paths"]
            for res, img_path in zip(results_list, img_paths):
                img_id = path_to_id.get(img_path, 0)
                boxes  = res["boxes"].cpu().numpy()
                scores = res["scores"].cpu().numpy()
                labels = res["labels"]

                for box, score, label in zip(boxes, scores, labels):
                    cat_id = _label_to_eval_cat(label)
                    if cat_id is None:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    predictions.append({
                        "image_id":    img_id,
                        "category_id": cat_id,
                        "bbox":        [x1, y1, x2 - x1, y2 - y1],
                        "score":       float(score),
                    })

            processed += bsz
            print(f"\r  推理进度: {processed}/{total}", end="", flush=True)

    print()
    return predictions


def _label_to_eval_cat(label: str) -> Optional[int]:
    label = label.strip().lower()
    if label.startswith("person"):
        return 1
    if label.startswith("car"):
        return 2
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: 计算 mAP
# ──────────────────────────────────────────────────────────────────────────────

def compute_map(coco_gt: COCO, predictions: List[Dict]) -> Dict[str, Any]:
    """
    使用 pycocotools COCOeval 计算 mAP@0.5 和 mAP@0.5:0.95。
    """
    if not predictions:
        print("  [警告] 无任何预测框，mAP=0")
        return {
            "map_50": 0.0, "map_50_95": 0.0,
            "ap_person": 0.0, "ap_car": 0.0,
            "n_preds": 0, "n_gt": len(coco_gt.anns),
        }

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    map_50_95 = float(evaluator.stats[0])
    map_50    = float(evaluator.stats[1])

    def _per_class_ap(cat_id: int) -> float:
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.catIds = [cat_id]
        ev.params.iouThrs = np.array([0.5])
        ev.evaluate()
        ev.accumulate()
        ev.summarize()   # 必须调用 summarize() 才能填充 ev.stats
        return max(float(ev.stats[0]), 0.0)

    return {
        "map_50":    map_50,
        "map_50_95": map_50_95,
        "ap_person": _per_class_ap(1),
        "ap_car":    _per_class_ap(2),
        "n_preds":   len(predictions),
        "n_gt":      len(coco_gt.anns),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5: CLI 入口
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSMA 通用评估：计算 mAP@0.5（支持 flir_v1 / flir_v2）"
    )
    parser.add_argument("--ckpt",         type=str, required=True,
                        help="CSMA 权重路径")
    parser.add_argument("--dataset",      type=str, default="flir_v1",
                        choices=["flir_v1", "flir_v2"],
                        help="数据集类型（默认 flir_v1）")
    parser.add_argument("--data-root",    type=str, default="FLIR_License/val",
                        help="val split 目录")
    parser.add_argument("--out-json",     type=str,
                        default="outputs_csma/logs/eval_result.json",
                        help="评估结果输出 JSON 路径")
    parser.add_argument("--batch-size",   type=int, default=4)
    parser.add_argument("--num-workers",  type=int, default=2)
    parser.add_argument("--box-threshold",  type=float, default=0.05)
    parser.add_argument("--text-threshold", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_csma] 设备:    {device}")
    print(f"[eval_csma] 数据集:  {args.dataset}  {args.data_root}")
    print(f"[eval_csma] 权重:    {args.ckpt}")

    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)

    # 与训练保持一致：限制 processor 图像尺寸为 cfg.img_size，
    # 否则默认 shortest_edge=800 会导致预测框坐标与 GT（原始尺寸）不匹配，AP=0
    cfg = CSMAConfig()
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
        ip = processor.image_processor
        try:
            cur_se = ip.size.shortest_edge or 0
        except AttributeError:
            cur_se = ip.size.get("shortest_edge", 0) or 0
        if cur_se > cfg.img_size:
            try:
                ip.size.shortest_edge = cfg.img_size
                ip.size.longest_edge  = cfg.img_size * 2
            except AttributeError:
                ip.size = {"shortest_edge": cfg.img_size, "longest_edge": cfg.img_size * 2}
            print(f"[eval_csma] processor image size 限制到 shortest_edge={cfg.img_size}")

    dino = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False

    csma = CSMA(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    csma.load_state_dict(state)
    csma.eval()
    print("[eval_csma] CSMA 权重已加载")

    dataset, valid_ids = _load_dataset(args.dataset, args.data_root, processor)
    print(f"[eval_csma] val 集大小: {len(dataset)} 张")

    coco_gt = _build_gt_coco(dataset, valid_ids)
    print(f"[eval_csma] GT 标注总数: {len(coco_gt.anns)}")

    print("[eval_csma] 开始推理...")
    predictions = run_eval(
        csma=csma, dino=dino, processor=processor,
        dataset=dataset, device=device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        box_threshold=args.box_threshold, text_threshold=args.text_threshold,
        dataset_mode=args.dataset,
    )
    print(f"[eval_csma] 共生成 {len(predictions)} 个预测框")

    print("[eval_csma] 计算 mAP...")
    results = compute_map(coco_gt, predictions)
    results["ckpt"]      = args.ckpt
    results["dataset"]   = args.dataset
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
    print(f"[eval_csma] 结果已保存: {args.out_json}")


if __name__ == "__main__":
    main()
