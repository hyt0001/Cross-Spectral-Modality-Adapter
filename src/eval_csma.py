"""
CSMA 通用检测评估模块（支持 flir_v1 / flir_v2 / llvip）。

在 val/test 集上计算 mAP@0.5（COCO 标准），输出 per-class AP + mAP。
使用 pycocotools 进行严格 IoU 匹配，保证与 COCO benchmark 结果可比。

两种运行模式：
  --no-csma   纯 Grounding DINO 基线（直接用红外图像推理，不经 CSMA）
  默认        DINO + CSMA（红外 → CSMA → 伪 RGB → DINO）

CLI 用法：
    # FLIR v1 + CSMA
    conda run -n RGBtest python -m src.eval_csma \\
        --ckpt outputs_csma/ckpt/csma_last.pt \\
        --dataset flir_v1 \\
        --data-root FLIR_License/val \\
        --out-json outputs_csma/logs/eval_last.json

    # LLVIP + CSMA（数据根目录 = 含 infrared/、visible/ 的目录）
    conda run -n RGBtest python -m src.eval_csma \\
        --ckpt outputs_csma_v3/ckpt/csma_last.pt \\
        --dataset llvip \\
        --data-root LLVIP/LLVIP \\
        --ann-file LLVIP/annotations/val.json \\
        --out-json outputs_csma_v3/logs/eval_llvip_csma.json

    # LLVIP 纯 DINO 基线（无需 checkpoint）
    conda run -n RGBtest python -m src.eval_csma \\
        --no-csma \\
        --dataset llvip \\
        --data-root LLVIP/LLVIP \\
        --ann-file LLVIP/annotations/val.json \\
        --out-json outputs_csma_v3/logs/eval_llvip_baseline.json

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
from src.dino_vision_bridge import forward_dino_eval_with_feature_adapter
from src.feature_adapter import FeatureAdapter


# ── 常量 ──────────────────────────────────────────────────────────────────────
# 默认 prompt（flir_v1 / flir_v2 使用）；llvip/m3fd 建议用 CLI --text-prompt 覆盖
DEFAULT_TEXT_PROMPT: str = "person. car."
EVAL_CATEGORIES: List[Dict] = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "car"},
]
# 各数据集 raw cat_id → eval cat_id（person=1, car=2）
# flir_v1/v2: person=1→1, car=3→2
# llvip:      person=0→1（仅行人，无 car GT）
# m3fd:       People=5→1, Car=2→2, Bus=1→3, Motorcycle=4→4, Truck=6→5, Lamp=3→6
DATASET_TO_EVAL_CAT: Dict[str, Dict[int, int]] = {
    "flir_v1": {1: 1, 3: 2},
    "flir_v2": {1: 1, 3: 2},
    "llvip":   {0: 1},
    "m3fd":    {5: 1, 2: 2, 1: 3, 4: 4, 6: 5, 3: 6},
}
# 各数据集 eval 用的 COCO categories 列表（m3fd 有 6 类）
DATASET_EVAL_CATEGORIES: Dict[str, List[Dict]] = {
    "flir_v1": EVAL_CATEGORIES,
    "flir_v2": EVAL_CATEGORIES,
    "llvip":   EVAL_CATEGORIES,
    "m3fd": [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "car"},
        {"id": 3, "name": "bus"},
        {"id": 4, "name": "motorcycle"},
        {"id": 5, "name": "truck"},
        {"id": 6, "name": "lamp"},
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: 数据集加载（三种模式）
# ──────────────────────────────────────────────────────────────────────────────

def _load_dataset(
    dataset_mode: str,
    data_root: str,
    processor: Any,
    text_prompt: str,
    ann_file: Optional[str] = None,
    split: str = "all",
    canonical_size: Optional[tuple[int, int]] = None,
):
    """
    根据 dataset_mode 加载对应 val/test 数据集。

    Args:
        dataset_mode: "flir_v1" / "flir_v2" / "llvip"
        data_root:    数据集根/split 目录
        processor:    AutoProcessor
        text_prompt:  检测 prompt，如 "person. car." 或 "person."（llvip 推荐）
        ann_file:     可选，显式指定 COCO 标注 JSON（llvip 跨目录时使用）

    Returns:
        (dataset, valid_cat_ids)
    """
    if dataset_mode == "flir_v1":
        from src.dataset_flir_v1 import FlirV1PairedDataset, build_flir_v1_category_map
        cat_map, valid_ids = build_flir_v1_category_map(text_prompt)
        dataset = FlirV1PairedDataset(
            root=data_root,
            processor=processor,
            text_prompt=text_prompt,
            category_map=cat_map,
            valid_cat_ids=valid_ids,
        )
        return dataset, valid_ids

    elif dataset_mode == "flir_v2":
        from src.dataset_flir_v2 import FlirADASV2Dataset, build_flir_v2_category_map
        cat_map, valid_ids = build_flir_v2_category_map(text_prompt)
        dataset = FlirADASV2Dataset(
            root=data_root,
            processor=processor,
            text_prompt=text_prompt,
            category_map=cat_map,
            valid_cat_ids=valid_ids,
        )
        return dataset, valid_ids

    elif dataset_mode == "llvip":
        from src.dataset_llvip import LLVIPPairedDataset, build_llvip_category_map
        cat_map, valid_ids = build_llvip_category_map(text_prompt)
        dataset = LLVIPPairedDataset(
            root=data_root,
            processor=processor,
            text_prompt=text_prompt,
            split="test",
            ann_file=ann_file,
            category_map=cat_map,
            valid_cat_ids=valid_ids,
        )
        return dataset, valid_ids

    elif dataset_mode == "m3fd":
        from src.dataset_m3fd import M3FDPairedDataset, build_m3fd_category_map
        cat_map, valid_ids = build_m3fd_category_map(text_prompt)
        dataset = M3FDPairedDataset(
            root=data_root,
            processor=processor,
            text_prompt=text_prompt,
            ann_file=ann_file,
            category_map=cat_map,
            valid_cat_ids=valid_ids,
            split=split,
            canonical_size=canonical_size,
        )
        return dataset, valid_ids

    else:
        raise ValueError(
            f"不支持的 dataset_mode: {dataset_mode}，"
            f"请使用 flir_v1 / flir_v2 / llvip / m3fd"
        )


def _get_collate(dataset_mode: str):
    """返回对应数据集的 collate 函数。"""
    if dataset_mode == "flir_v1":
        from src.dataset_flir_v1 import collate_flir_v1
        return collate_flir_v1
    elif dataset_mode == "flir_v2":
        from src.dataset_flir_v2 import collate_flir_v2
        return collate_flir_v2
    elif dataset_mode == "llvip":
        from src.dataset_llvip import collate_llvip
        return collate_llvip
    elif dataset_mode == "m3fd":
        from src.dataset_m3fd import collate_m3fd
        return collate_m3fd
    else:
        raise ValueError(f"不支持的 dataset_mode: {dataset_mode}")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: 构建 GT COCO 对象
# ──────────────────────────────────────────────────────────────────────────────

def _build_path_to_id(dataset) -> Dict[str, int]:
    """构建 image 绝对路径 → COCO image_id 索引。"""
    if hasattr(dataset, "_resolve_paths"):
        return {
            dataset._resolve_paths(img["file_name"])[0]: int(img["id"])
            for img in dataset._images
        }
    return {
        os.path.join(dataset._root, img["file_name"]): int(img["id"])
        for img in dataset._images
    }


def _build_gt_coco(
    dataset,
    valid_cat_ids: frozenset,
    dataset_mode: str,
) -> COCO:
    """
    将数据集 GT 标注转换为 pycocotools COCO 对象。

    兼容 FlirV1PairedDataset、FlirADASV2Dataset、LLVIPPairedDataset
    （统一接口：_images, _id_to_anns, _root, _images[i]["file_name"]）。
    """
    images: List[Dict] = []
    annotations: List[Dict] = []
    ann_id = 1

    if dataset_mode == "m3fd" and hasattr(dataset, "_cat_map"):
        from src.dataset_m3fd import build_m3fd_eval_categories
        cat_to_eval = dict(dataset._cat_map)
        eval_cats = build_m3fd_eval_categories(dataset._text_prompt)
    else:
        cat_to_eval = DATASET_TO_EVAL_CAT[dataset_mode]
        eval_cats = DATASET_EVAL_CATEGORIES.get(dataset_mode, EVAL_CATEGORIES)

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
                "category_id": cat_to_eval[cid],
                "bbox":        [x, y, w, h],
                "area":        float(ann["area"]),
                "iscrowd":     int(ann.get("iscrowd", 0)),
            })
            ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": eval_cats,
    }
    coco_gt.createIndex()
    return coco_gt


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: 推理收集预测框
# ──────────────────────────────────────────────────────────────────────────────

def run_eval(
    dino: GroundingDinoForObjectDetection,
    processor: Any,
    dataset,
    device: torch.device,
    text_prompt: str,
    batch_size: int = 4,
    num_workers: int = 2,
    box_threshold: float = 0.05,
    text_threshold: float = 0.05,
    dataset_mode: str = "flir_v1",
    csma: Optional[CSMA] = None,
    feature_adapter: Optional[FeatureAdapter] = None,
    adapter_mode: str = "pixel",
) -> List[Dict]:
    """
    在整个 val/test 集上推理，返回 pycocotools 格式的预测列表。

    Args:
        csma:             像素模式 CSMA；与 feature_adapter 互斥。
        feature_adapter:  特征模式适配器。
        adapter_mode:     ``pixel`` 或 ``feature``；无适配器时为基线（IR 直送 DINO）。
        text_prompt:      检测 prompt，与数据集加载时保持一致。

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

    enc = processor.tokenizer(text_prompt, return_tensors="pt")
    input_ids_base = enc["input_ids"].to(device)
    attention_mask_base = enc["attention_mask"].to(device)

    # 预构建 image_path → image_id 索引
    path_to_id: Dict[str, int] = _build_path_to_id(dataset)

    if csma is not None:
        csma.eval()
    if feature_adapter is not None:
        feature_adapter.eval()
    dino.eval()
    predictions: List[Dict] = []

    total = len(dataset)
    processed = 0
    with torch.no_grad():
        for batch in loader:
            ir_pv = batch["pixel_values"].to(device)
            pm    = batch["pixel_mask"].to(device)
            bsz   = ir_pv.shape[0]

            # Phase 3.1：特征转换
            if adapter_mode == "feature" and feature_adapter is not None:
                outputs = forward_dino_eval_with_feature_adapter(
                    dino,
                    feature_adapter,
                    ir_pv,
                    pm,
                    input_ids_base.expand(bsz, -1),
                    attention_mask_base.expand(bsz, -1),
                )
            elif csma is not None:
                pixel_values_for_dino = csma(ir_pv)
                outputs = dino(
                    pixel_values=pixel_values_for_dino,
                    pixel_mask=pm,
                    input_ids=input_ids_base.expand(bsz, -1),
                    attention_mask=attention_mask_base.expand(bsz, -1),
                )
            else:
                outputs = dino(
                    pixel_values=ir_pv,
                    pixel_mask=pm,
                    input_ids=input_ids_base.expand(bsz, -1),
                    attention_mask=attention_mask_base.expand(bsz, -1),
                )

            # target_sizes 必须用原始图像尺寸（GT 坐标系），而非 resize/pad 后的张量尺寸。
            # pixel_mask 推算的是 resize 后尺寸（非原始），必须从 labels["orig_size"] 获取。
            labels_batch = batch["labels"]
            target_sizes = torch.stack(
                [lbl["orig_size"] for lbl in labels_batch], dim=0
            ).to(device)  # [B, 2]: [H, W] 原始图像尺寸

            results_list = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=input_ids_base.expand(bsz, -1),
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
                    cat_id = _label_to_eval_cat(label, dataset_mode, text_prompt)
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


def _label_to_eval_cat(
    label: str,
    dataset_mode: str = "flir_v1",
    text_prompt: str = DEFAULT_TEXT_PROMPT,
) -> Optional[int]:
    """
    将 DINO 输出的文本标签映射到 eval cat_id。

    m3fd：仅接受 text_prompt 中出现的类别；其余标签返回 None。
    flir_v1 / flir_v2 / llvip：person=1, car=2。
    """
    label = label.strip().lower()
    if dataset_mode == "m3fd":
        from src.dataset_m3fd import build_m3fd_label_to_eval_cat
        label_map = build_m3fd_label_to_eval_cat(text_prompt)
        for prefix, cat_id in sorted(label_map.items(), key=lambda x: -len(x[0])):
            if label.startswith(prefix):
                return cat_id
        return None
    # 默认：flir_v1 / flir_v2 / llvip
    if label.startswith("person"):
        return 1
    if label.startswith("car"):
        return 2
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: 计算 mAP
# ──────────────────────────────────────────────────────────────────────────────

def _coco_stats_to_summary(stats: np.ndarray) -> Dict[str, float]:
    """
    将 COCOeval.summarize() 的 stats 向量转为结构化字典。

    stats 顺序与 pycocotools 默认 bbox 评测一致（12 项）。
    无 GT 的 area 档位返回 -1.0，调用方展示时可标为 N/A。
    """
    keys = (
        "map_50_95",
        "map_50",
        "map_75",
        "ap_small_50_95",
        "ap_medium_50_95",
        "ap_large_50_95",
        "ar_all_50_95_max1",
        "ar_all_50_95_max10",
        "ar_all_50_95_max100",
        "ar_small_50_95",
        "ar_medium_50_95",
        "ar_large_50_95",
    )
    summary: Dict[str, float] = {}
    for key, val in zip(keys, stats[: len(keys)]):
        summary[key] = float(val)
    return summary


def _per_class_coco_metrics(
    coco_gt: COCO,
    coco_dt: COCO,
    cat_id: int,
) -> Dict[str, float]:
    """
    单类别 COCO 指标（IoU=0.50:0.95 主阈值，含 AP@0.5 / AP@0.75 / AR@100）。
    """
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.catIds = [cat_id]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    stats = ev.stats
    return {
        "ap_50_95": max(float(stats[0]), 0.0),
        "ap_50": max(float(stats[1]), 0.0),
        "ap_75": max(float(stats[2]), 0.0) if stats[2] >= 0 else -1.0,
        "ar_max1": float(stats[6]),
        "ar_max10": float(stats[7]),
        "ar_max100": float(stats[8]),
    }


def compute_map(
    coco_gt: COCO,
    predictions: List[Dict],
    dataset_mode: str = "flir_v1",
) -> Dict[str, Any]:
    """
    使用 pycocotools COCOeval 计算 mAP@0.5 和 mAP@0.5:0.95。

    评测类别以 coco_gt.dataset['categories'] 为准（M3FD 两类 prompt 时仅 person+car）。

    注意：对于 LLVIP（仅 person GT），car 类 AP 恒为 0；
    COCOeval 在无 GT 类别时自动从均值中排除该类，map_50 即等于 ap_person。
    """
    eval_cats: List[Dict] = coco_gt.dataset.get("categories", EVAL_CATEGORIES)

    if not predictions:
        print("  [警告] 无任何预测框，mAP=0")
        result: Dict[str, Any] = {
            "map_50": 0.0, "map_50_95": 0.0,
            "n_preds": 0, "n_gt": len(coco_gt.anns),
        }
        for cat in eval_cats:
            result[f"ap_{cat['name']}"] = 0.0
        return result

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    coco_summary = _coco_stats_to_summary(evaluator.stats)
    map_50_95 = coco_summary["map_50_95"]
    map_50 = coco_summary["map_50"]

    result: Dict[str, Any] = {
        "map_50": map_50,
        "map_50_95": map_50_95,
        "map_75": coco_summary["map_75"],
        "coco_summary": coco_summary,
        "n_preds": len(predictions),
        "n_gt": len(coco_gt.anns),
    }

    per_class: Dict[str, Dict[str, float]] = {}
    gt_cat_ids = {ann["category_id"] for ann in coco_gt.anns.values()}
    for cat in eval_cats:
        if cat["id"] not in gt_cat_ids:
            continue
        cat_id = cat["id"]
        cat_name = cat["name"]
        metrics = _per_class_coco_metrics(coco_gt, coco_dt, cat_id)
        per_class[cat_name] = metrics
        result[f"ap_{cat_name}"] = metrics["ap_50"]
    if per_class:
        result["per_class"] = per_class

    if "person" in per_class and "car" in per_class:
        result["person_car_mean"] = (
            per_class["person"]["ap_50"] + per_class["car"]["ap_50"]
        ) / 2.0
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5: CLI 入口
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSMA 通用评估：计算 mAP@0.5（支持 flir_v1 / flir_v2 / llvip）"
    )
    parser.add_argument(
        "--ckpt", type=str, default="",
        help="CSMA 权重路径（--no-csma 时可省略）",
    )
    parser.add_argument(
        "--no-csma", action="store_true",
        help="纯 Grounding DINO 基线模式：跳过 CSMA，直接将 IR 图像送入 DINO",
    )
    parser.add_argument(
        "--dataset", type=str, default="flir_v1",
        choices=["flir_v1", "flir_v2", "llvip", "m3fd"],
        help="数据集类型（默认 flir_v1）",
    )
    parser.add_argument(
        "--data-root", type=str, default="FLIR_License/val",
        help="数据集根/split 目录（llvip 传含 infrared/、visible/ 的目录）",
    )
    parser.add_argument(
        "--ann-file", type=str, default=None,
        help="（可选）显式指定 COCO 标注 JSON，用于 llvip 标注与图像不在同一根目录的情况",
    )
    parser.add_argument(
        "--split", type=str, default="all",
        choices=["all", "train", "val"],
        help="（m3fd）数据集划分：all=全量 / train=前80%% / val=后20%%（与训练 val 早停一致）",
    )
    parser.add_argument(
        "--canonical-size", type=str, default=None,
        help="（m3fd）统一缩放到 WxH，如 1024,768；与训练 val 保持一致时推荐开启",
    )
    parser.add_argument(
        "--text-prompt", type=str, default=None,
        help=(
            "检测 prompt（默认 flir_v1/v2 用 'person. car.'，llvip 用 'person.'）。"
            "若不指定则自动按 dataset 选择默认值。"
        ),
    )
    parser.add_argument(
        "--out-json", type=str,
        default="outputs_csma/logs/eval_result.json",
        help="评估结果输出 JSON 路径",
    )
    parser.add_argument("--batch-size",      type=int,   default=4)
    parser.add_argument("--num-workers",     type=int,   default=2)
    parser.add_argument("--box-threshold",   type=float, default=0.05)
    parser.add_argument("--text-threshold",  type=float, default=0.05)
    parser.add_argument(
        "--adapter-mode", type=str, default="pixel",
        choices=["pixel", "feature"],
        help="适配器模式：pixel=CSMA；feature=FeatureAdapter",
    )
    parser.add_argument(
        "--pseudo-clamp", type=float, default=None,
        help="pseudo 像素 clamp 上限；须与训练一致（OWL 常用 3.0 / Final 2.0）",
    )
    parser.add_argument(
        "--residual-scale", type=float, default=None,
        help="残差缩放 pseudo=IR+scale*delta；须与训练一致（OWL 常用 0.1 / Final 0.05）",
    )
    args = parser.parse_args()

    # 参数校验
    if not args.no_csma and not args.ckpt:
        parser.error("--ckpt 是必填项（除非使用 --no-csma 基线模式）")

    # text_prompt：未指定时按数据集自动选择合理默认值
    # llvip 只有 person GT，使用 "person." 避免 DINO 注意力被 car token 分散
    if args.text_prompt:
        text_prompt = args.text_prompt.strip()
    elif args.dataset == "llvip":
        text_prompt = "person."
    elif args.dataset == "m3fd":
        from src.dataset_m3fd import M3FD_DEFAULT_TEXT_PROMPT
        text_prompt = M3FD_DEFAULT_TEXT_PROMPT
    else:
        text_prompt = DEFAULT_TEXT_PROMPT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_baseline = args.no_csma
    if is_baseline:
        mode_str = "纯 DINO 基线"
    elif args.adapter_mode == "feature":
        mode_str = f"DINO + FeatureAdapter  权重={args.ckpt}"
    else:
        mode_str = f"DINO + CSMA  权重={args.ckpt}"
    print(f"[eval_csma] 设备:    {device}")
    print(f"[eval_csma] 模式:    {mode_str}")
    print(f"[eval_csma] 数据集:  {args.dataset}  {args.data_root}")
    print(f"[eval_csma] Prompt:  {text_prompt!r}")
    if args.ann_file:
        print(f"[eval_csma] 标注文件: {args.ann_file}")

    canonical_size: Optional[tuple[int, int]] = None
    if args.canonical_size:
        parts = [p.strip() for p in args.canonical_size.split(",")]
        if len(parts) != 2:
            parser.error("--canonical-size 格式应为 'W,H'，如 1024,768")
        canonical_size = (int(parts[0]), int(parts[1]))
    if args.dataset == "m3fd" and args.split != "all":
        print(f"[eval_csma] M3FD split: {args.split}")
    if canonical_size is not None:
        print(f"[eval_csma] canonical_size: {canonical_size}")

    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)

    # 与训练保持一致：限制 processor 图像尺寸为 cfg.img_size，
    # 否则默认 shortest_edge=800 会导致预测框坐标与 GT（原始尺寸）不匹配，AP=0
    cfg_overrides: Dict[str, Any] = {}
    if args.pseudo_clamp is not None:
        cfg_overrides["pseudo_clamp"] = args.pseudo_clamp
    if args.residual_scale is not None:
        cfg_overrides["residual_scale"] = args.residual_scale
    cfg = CSMAConfig(**cfg_overrides)
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

    dino = GroundingDinoForObjectDetection.from_pretrained(model_id, local_files_only=True).to(device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False

    # Phase 5.1：按模式加载适配器（基线模式跳过）
    csma: Optional[CSMA] = None
    feature_adapter: Optional[FeatureAdapter] = None
    if not args.no_csma:
        raw = torch.load(args.ckpt, map_location=device, weights_only=True)
        if args.adapter_mode == "feature":
            feature_adapter = FeatureAdapter(cfg).to(device)
            state = raw["feature_adapter"] if isinstance(raw, dict) and "feature_adapter" in raw else raw
            feature_adapter.load_state_dict(state)
            feature_adapter.eval()
            print("[eval_csma] FeatureAdapter 权重已加载")
        else:
            csma = CSMA(cfg).to(device)
            state = raw["csma"] if isinstance(raw, dict) and "csma" in raw else raw
            csma.load_state_dict(state)
            csma.eval()
            print("[eval_csma] CSMA 权重已加载")
    else:
        print("[eval_csma] 基线模式：不加载适配器，IR 图像直接送入 DINO")

    dataset, valid_ids = _load_dataset(
        args.dataset, args.data_root, processor,
        text_prompt=text_prompt,
        ann_file=args.ann_file,
        split=args.split,
        canonical_size=canonical_size,
    )
    print(f"[eval_csma] 数据集大小: {len(dataset)} 张")

    coco_gt = _build_gt_coco(dataset, valid_ids, args.dataset)
    print(f"[eval_csma] GT 标注总数: {len(coco_gt.anns)}")

    print("[eval_csma] 开始推理...")
    predictions = run_eval(
        dino=dino,
        processor=processor,
        dataset=dataset,
        device=device,
        text_prompt=text_prompt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        dataset_mode=args.dataset,
        csma=csma,
        feature_adapter=feature_adapter,
        adapter_mode=args.adapter_mode,
    )
    print(f"[eval_csma] 共生成 {len(predictions)} 个预测框")

    print("[eval_csma] 计算 mAP...")
    results = compute_map(coco_gt, predictions, dataset_mode=args.dataset)
    results["mode"]        = (
        "baseline_dino" if args.no_csma
        else ("feature_adapter" if args.adapter_mode == "feature" else "csma")
    )
    results["ckpt"]        = args.ckpt if not args.no_csma else ""
    results["dataset"]     = args.dataset
    results["data_root"]   = args.data_root
    results["text_prompt"] = text_prompt
    results["split"]       = args.split
    if canonical_size is not None:
        results["canonical_size"] = list(canonical_size)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    eval_cats = [
        c for c in coco_gt.dataset.get("categories", EVAL_CATEGORIES)
    ]
    print()
    print("=" * 50)
    print(f"  模式          : {results['mode']}")
    print(f"  Prompt        : {text_prompt!r}")
    print(f"  mAP@0.5       : {results['map_50']:.4f}")
    if "person_car_mean" in results:
        print(f"  person+car    : {results['person_car_mean']:.4f}")
    print(f"  mAP@0.5:0.95  : {results['map_50_95']:.4f}")
    if "map_75" in results and results["map_75"] >= 0:
        print(f"  AP@0.75       : {results['map_75']:.4f}")
    cs = results.get("coco_summary", {})
    if cs.get("ar_all_50_95_max100", -1) >= 0:
        print(f"  AR@100        : {cs['ar_all_50_95_max100']:.4f}")
    for cat in eval_cats:
        key = f"ap_{cat['name']}"
        if key in results:
            print(f"  AP_{cat['name']:12s}@0.5 : {results[key]:.4f}")
    per_class = results.get("per_class", {})
    for cat in eval_cats:
        pc = per_class.get(cat["name"])
        if pc is None:
            continue
        if pc.get("ap_75", -1) >= 0:
            print(f"  AP_{cat['name']:12s}@0.75: {pc['ap_75']:.4f}")
        if pc.get("ar_max100", -1) >= 0:
            print(f"  AR_{cat['name']:12s}@100: {pc['ar_max100']:.4f}")
    print(f"  预测框 / GT框  : {results['n_preds']} / {results['n_gt']}")
    print("=" * 50)
    print(f"[eval_csma] 结果已保存: {args.out_json}")


if __name__ == "__main__":
    main()
