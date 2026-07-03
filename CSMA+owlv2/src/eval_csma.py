"""OWLv2-CSMA evaluation on FLIR / LLVIP / KAIST / M3FD / NOT-156."""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from typing import Any, Dict, List, Union

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from src.config import CSMAConfig
from src.csma import CSMA
from src.dataset_flir_v1 import FlirV1PairedDataset, build_flir_v1_category_map, collate_flir_v1
from src.dataset_kaist import (
    KAISTPairedDataset,
    KAIST_ANN_CAT_TO_EVAL_CAT,
    KAIST_EVAL_CATEGORIES,
    KAIST_LABEL_TO_EVAL_CAT,
    KAIST_VALID_CAT_IDS,
)
from src.dataset_llvip import (
    LLVIPPairedDataset,
    LLVIP_ANN_CAT_TO_EVAL_CAT,
    LLVIP_EVAL_CATEGORIES,
    LLVIP_LABEL_TO_EVAL_CAT,
    LLVIP_VALID_CAT_IDS,
)
from src.dataset_m3fd import M3FDPairedDataset, build_m3fd_category_map
from src.dataset_not156 import NOT156PairedDataset, build_not156_category_map

FLIR_TO_EVAL_CAT = {1: 1, 3: 2}
FLIR_EVAL_CATEGORIES = [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}]
FLIR_LABEL_TO_EVAL_CAT = {"person": 1, "car": 2}

EvalDataset = Union[
    FlirV1PairedDataset,
    LLVIPPairedDataset,
    KAISTPairedDataset,
    M3FDPairedDataset,
    NOT156PairedDataset,
]


def collate_eval_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "gt_boxes": [b["gt_boxes"] for b in batch],
        "gt_labels": [b["gt_labels"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],
        "orig_sizes": [b.get("orig_size", b.get("orig_sizes")) for b in batch],
    }
    rgb = [b.get("rgb_pixel_values") for b in batch]
    if any(v is not None for v in rgb):
        out["rgb_pixel_values"] = torch.stack([v for v in rgb if v is not None])
        out["rgb_indices"] = [i for i, v in enumerate(rgb) if v is not None]
    return out


def _build_gt_coco(
    dataset: EvalDataset,
    valid_cat_ids: frozenset,
    ann_cat_to_eval_cat: Dict[int, int],
    eval_categories: List[Dict[str, Any]],
) -> COCO:
    images, annotations, ann_id = [], [], 1
    for img_info in dataset._images:
        img_id = int(img_info["id"])
        images.append({
            "id": img_id,
            "width": int(img_info["width"]),
            "height": int(img_info["height"]),
            "file_name": img_info["file_name"],
        })
        for ann in dataset._id_to_anns.get(img_id, []):
            cid = int(ann["category_id"])
            if isinstance(dataset, M3FDPairedDataset):
                if cid not in ann_cat_to_eval_cat:
                    continue
                eval_cid = ann_cat_to_eval_cat[cid]
            else:
                if cid not in valid_cat_ids:
                    continue
                eval_cid = ann_cat_to_eval_cat[cid]
            x, y, w, h = [float(v) for v in ann["bbox"]]
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": eval_cid,
                "bbox": [x, y, w, h],
                "area": float(ann["area"]),
                "iscrowd": int(ann.get("iscrowd", 0)),
            })
            ann_id += 1
    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": annotations, "categories": eval_categories}
    coco_gt.createIndex()
    return coco_gt


def run_eval(
    csma: CSMA,
    owlv2: Owlv2ForObjectDetection,
    processor: Owlv2Processor,
    dataset: EvalDataset,
    device: torch.device,
    text_labels: List[str],
    label_to_eval_cat: Dict[str, int],
    batch_size: int = 4,
    num_workers: int = 2,
    threshold: float = 0.2,
) -> List[Dict[str, Any]]:
    collate_fn = collate_flir_v1 if isinstance(dataset, FlirV1PairedDataset) else collate_eval_batch
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    with torch.no_grad():
        text_enc = processor(text=[text_labels], images=None, return_tensors="pt", padding=True)
    base_input_ids = text_enc["input_ids"].to(device)
    base_attn_mask = text_enc["attention_mask"].to(device)

    root = getattr(dataset, "_root", "")
    path_to_id = {os.path.join(root, img["file_name"]): img["id"] for img in dataset._images}

    csma.eval()
    owlv2.eval()
    predictions: List[Dict[str, Any]] = []
    processed, total = 0, len(dataset)

    with torch.no_grad():
        for batch in loader:
            ir_pv = batch["pixel_values"].to(device)
            B = ir_pv.shape[0]
            pseudo_rgb = csma(ir_pv)
            ids_b = base_input_ids.repeat(B, 1)
            atm_b = base_attn_mask.repeat(B, 1)
            outputs = owlv2(input_ids=ids_b, pixel_values=pseudo_rgb, attention_mask=atm_b)
            target_sizes = torch.tensor(batch["orig_sizes"], device=device)
            results_list = processor.post_process_grounded_object_detection(
                outputs, target_sizes=target_sizes,
                threshold=threshold, text_labels=[text_labels] * B,
            )
            for res, img_path in zip(results_list, batch["image_paths"]):
                img_id = path_to_id.get(img_path, 0)
                for box, score, label in zip(res["boxes"].cpu(), res["scores"].cpu(), res["text_labels"]):
                    cat_id = label_to_eval_cat.get(label.strip().lower())
                    if cat_id is None:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    predictions.append({
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    })
            processed += B
            print(f"\r  推理进度: {processed}/{total}", end="", flush=True)
    print()
    return predictions


def compute_map(coco_gt: COCO, predictions: List[Dict[str, Any]], eval_categories: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not predictions:
        metrics: Dict[str, Any] = {
            "AP": 0.0, "AP50": 0.0, "AP75": 0.0,
            "APS": 0.0, "APM": 0.0, "APL": 0.0,
            "AR1": 0.0, "AR10": 0.0, "AR100": 0.0,
            "ARS": 0.0, "ARM": 0.0, "ARL": 0.0,
            "n_preds": 0, "n_gt": len(coco_gt.anns),
        }
        for cat in eval_categories:
            metrics[f"AP_{cat['name']}"] = 0.0
        return metrics

    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    old = sys.stdout
    sys.stdout = io.StringIO()
    ev.summarize()
    sys.stdout = old
    s = ev.stats

    metrics = {
        "AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]),
        "APS": float(s[3]), "APM": float(s[4]), "APL": float(s[5]),
        "AR1": float(s[6]), "AR10": float(s[7]), "AR100": float(s[8]),
        "ARS": float(s[9]), "ARM": float(s[10]), "ARL": float(s[11]),
        "n_preds": len(predictions), "n_gt": len(coco_gt.anns),
    }
    for cat in eval_categories:
        ev2 = COCOeval(coco_gt, coco_dt, "bbox")
        ev2.params.catIds = [cat["id"]]
        ev2.params.iouThrs = np.array([0.5])
        old = sys.stdout
        sys.stdout = io.StringIO()
        ev2.evaluate()
        ev2.accumulate()
        ev2.summarize()
        sys.stdout = old
        metrics[f"AP_{cat['name']}"] = max(float(ev2.stats[0]), 0.0)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="OWLv2-CSMA 评估")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", choices=["flir_v1", "llvip", "kaist", "m3fd", "not156"], default="flir_v1")
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/val")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--ann-file", type=str, default=None,
                        help="M3FD: COCO 标注相对 data-root 的路径")
    parser.add_argument("--seq-subdir", type=str, default=None,
                        help="NOT-156: 序列根目录，默认 NOT156_train/NOT156_train")
    parser.add_argument("--text-labels", nargs="+", default=None)
    parser.add_argument("--out-json", type=str, default="outputs_teammate/logs/eval_result.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--disable-group-norm", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device} dataset={args.dataset}", flush=True)
    raw = torch.load(args.ckpt, map_location=device, weights_only=True)

    overrides: dict[str, Any] = {}
    if isinstance(raw, dict) and "config_overrides" in raw:
        overrides.update(raw["config_overrides"])
    if args.disable_group_norm:
        overrides["use_group_norm"] = False
    cfg = CSMAConfig.from_overrides(overrides) if overrides else CSMAConfig()
    if args.text_labels is not None:
        cfg.text_labels = args.text_labels
    elif args.dataset in {"llvip", "kaist"}:
        cfg.text_labels = ["person"]
    elif args.dataset in {"m3fd", "not156"}:
        cfg.text_labels = ["person", "car"]

    processor = Owlv2Processor.from_pretrained(cfg.model_id)
    owlv2 = Owlv2ForObjectDetection.from_pretrained(cfg.model_id).to(device)
    owlv2.eval()

    csma = CSMA(cfg).to(device)
    state = raw["csma"] if isinstance(raw, dict) and "csma" in raw else raw
    csma.load_state_dict(state)
    csma.eval()

    if args.dataset == "flir_v1":
        cat_map, valid_ids = build_flir_v1_category_map(cfg.text_labels)
        dataset: EvalDataset = FlirV1PairedDataset(
            root=args.data_root, processor=processor,
            text_labels=cfg.text_labels, category_map=cat_map, valid_cat_ids=valid_ids,
        )
        eval_categories = FLIR_EVAL_CATEGORIES
        ann_cat_to_eval_cat = FLIR_TO_EVAL_CAT
        label_to_eval_cat = FLIR_LABEL_TO_EVAL_CAT
    elif args.dataset == "llvip":
        dataset = LLVIPPairedDataset(
            root=args.data_root, processor=processor,
            split=args.split, text_labels=cfg.text_labels,
        )
        valid_ids = LLVIP_VALID_CAT_IDS
        eval_categories = LLVIP_EVAL_CATEGORIES
        ann_cat_to_eval_cat = LLVIP_ANN_CAT_TO_EVAL_CAT
        label_to_eval_cat = LLVIP_LABEL_TO_EVAL_CAT
    elif args.dataset == "kaist":
        if cfg.text_labels != ["person"]:
            raise ValueError("KAIST evaluation currently supports only text_labels=['person']")
        split = "test-all-20" if args.split == "test" else args.split
        dataset = KAISTPairedDataset(
            root=args.data_root, processor=processor,
            split=split, text_labels=cfg.text_labels,
        )
        valid_ids = KAIST_VALID_CAT_IDS
        eval_categories = KAIST_EVAL_CATEGORIES
        ann_cat_to_eval_cat = KAIST_ANN_CAT_TO_EVAL_CAT
        label_to_eval_cat = KAIST_LABEL_TO_EVAL_CAT
    elif args.dataset == "m3fd":
        ann_file = args.ann_file or "annotations/instances_default.json"
        (
            _cat_map,
            valid_ids,
            eval_categories,
            ann_cat_to_eval_cat,
            label_to_eval_cat,
        ) = build_m3fd_category_map(cfg.text_labels)
        dataset = M3FDPairedDataset(
            root=args.data_root,
            processor=processor,
            text_labels=cfg.text_labels,
            ann_file=ann_file,
        )
    else:
        (
            _cat_map,
            valid_ids,
            eval_categories,
            ann_cat_to_eval_cat,
            label_to_eval_cat,
        ) = build_not156_category_map(cfg.text_labels)
        dataset = NOT156PairedDataset(
            root=args.data_root,
            processor=processor,
            text_labels=cfg.text_labels,
            seq_subdir=args.seq_subdir,
        )

    coco_gt = _build_gt_coco(dataset, valid_ids, ann_cat_to_eval_cat, eval_categories)
    predictions = run_eval(
        csma, owlv2, processor, dataset, device,
        cfg.text_labels, label_to_eval_cat,
        args.batch_size, args.num_workers, args.threshold,
    )
    results = compute_map(coco_gt, predictions, eval_categories)
    results.update({
        "ckpt": args.ckpt,
        "data_root": args.data_root,
        "dataset": args.dataset,
        "text_labels": cfg.text_labels,
    })

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 55)
    print(f"  AP[.5:.95] : {results['AP']:.4f}   AP50 : {results['AP50']:.4f}")
    for cat in eval_categories:
        key = f"AP_{cat['name']}"
        if key in results:
            print(f"  {key:12s}: {results[key]:.4f}")
    print(f"  预测框/GT框 : {results['n_preds']}/{results['n_gt']}")
    print("=" * 55)
    print(f"[eval] 结果已保存: {args.out_json}")


if __name__ == "__main__":
    main()
