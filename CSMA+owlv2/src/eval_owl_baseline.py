"""Native OWLv2 baseline evaluation on IR images (no CSMA)."""
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
from torch.utils.data import DataLoader
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from src.config import CSMAConfig
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
from src.eval_csma import EvalDataset, _build_gt_coco, collate_eval_batch, compute_map

FLIR_TO_EVAL_CAT = {1: 1, 3: 2}
FLIR_EVAL_CATEGORIES = [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}]
FLIR_LABEL_TO_EVAL_CAT = {"person": 1, "car": 2}


def run_baseline_eval(
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

    owlv2.eval()
    predictions: List[Dict[str, Any]] = []
    processed, total = 0, len(dataset)

    with torch.no_grad():
        for batch in loader:
            ir_pv = batch["pixel_values"].to(device)
            B = ir_pv.shape[0]
            ids_b = base_input_ids.repeat(B, 1)
            atm_b = base_attn_mask.repeat(B, 1)
            outputs = owlv2(input_ids=ids_b, pixel_values=ir_pv, attention_mask=atm_b)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="OWLv2 IR baseline 评估")
    parser.add_argument("--dataset", choices=["flir_v1", "llvip", "kaist", "m3fd", "not156"], default="llvip")
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/LLVIP")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--ann-file", type=str, default=None)
    parser.add_argument("--seq-subdir", type=str, default=None)
    parser.add_argument("--text-labels", nargs="+", default=None)
    parser.add_argument("--out-json", type=str, default="outputs_teammate/logs/owl_baseline.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] OWLv2 baseline device={device} dataset={args.dataset}", flush=True)

    cfg = CSMAConfig()
    if args.text_labels is not None:
        cfg.text_labels = args.text_labels
    elif args.dataset in {"llvip", "kaist"}:
        cfg.text_labels = ["person"]
    elif args.dataset in {"m3fd", "not156"}:
        cfg.text_labels = ["person", "car"]

    processor = Owlv2Processor.from_pretrained(cfg.model_id)
    owlv2 = Owlv2ForObjectDetection.from_pretrained(cfg.model_id).to(device)
    owlv2.eval()

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
    predictions = run_baseline_eval(
        owlv2, processor, dataset, device,
        cfg.text_labels, label_to_eval_cat,
        args.batch_size, args.num_workers, args.threshold,
    )
    results = compute_map(coco_gt, predictions, eval_categories)
    results.update({
        "data_root": args.data_root,
        "dataset": args.dataset,
        "text_labels": cfg.text_labels,
        "mode": "owl_baseline_ir",
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
