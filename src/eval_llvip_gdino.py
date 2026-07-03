"""
LLVIP 上 Grounding DINO + CSMA 零样本泛化评估（teacher 匹配）。

与 eval_llvip_detr 不同：CSMA 在 FLIR 上由 GDINO teacher 训练，
本脚本用同一 GDINO 做 downstream，避免 cross-detector 混淆。

坐标：LLVIP GT 在 1280×1024；GDINO 输入为 processor 缩放后的 ~640×512。
post_process 的 target_sizes 必须用 orig_h/orig_w（GT 坐标系），
不能只用 pixel_mask 反推尺寸（eval_csma 在 FLIR 640×512 上碰巧一致）。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from src.config import CSMAConfig
from src.csma import CSMA
from src.eval_csma import compute_map
from src.eval_llvip_detr import (
    LLVIP_GT_JSON,
    LLVIP_IR_DIR,
    LLVIPIRDataset,
    _build_llvip_gt,
    _collate_llvip,
    _load_csma,
)
from src.eval_yolo_csma import _configure_processor

EVAL_TEXT_PROMPT = "person."
# GDINO COCO label 1 = person → LLVIP eval cat 1
GDINO_LABEL_TO_LLVIP: Dict[int, int] = {1: 1}


def run_gdino_llvip_eval(
    csma: CSMA,
    dino: GroundingDinoForObjectDetection,
    processor: Any,
    dataset: LLVIPIRDataset,
    device: torch.device,
    batch_size: int,
    box_threshold: float,
    text_threshold: float,
    input_mode: str,
) -> List[Dict]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_llvip, num_workers=2,
    )
    enc = processor.tokenizer(EVAL_TEXT_PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    csma.eval()
    dino.eval()
    predictions: List[Dict] = []
    total = len(dataset)
    processed = 0

    with torch.no_grad():
        for batch in loader:
            image_ids = batch["image_ids"]
            orig_ws = batch["orig_ws"]
            orig_hs = batch["orig_hs"]
            bsz = len(image_ids)

            if input_mode == "ir_pipeline":
                ir_pv = batch["pixel_values"].to(device)
            else:
                assert input_mode == "pseudo_rgb"
                ir_pv = batch["pixel_values"].to(device)
                ir_pv = csma(ir_pv)

            pm = batch["pixel_masks"].to(device)
            # GT 坐标系：1280×1024，而非 processor 有效区域 640×512
            target_sizes = torch.tensor(
                [[orig_hs[i], orig_ws[i]] for i in range(bsz)],
                dtype=torch.int64,
                device=device,
            )

            outputs = dino(
                pixel_values=ir_pv,
                pixel_mask=pm,
                input_ids=input_ids.expand(bsz, -1),
                attention_mask=attention_mask.expand(bsz, -1),
            )
            results_list = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=input_ids.expand(bsz, -1),
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )

            for res, img_id in zip(results_list, image_ids):
                boxes = res["boxes"].cpu().numpy()
                scores = res["scores"].cpu().numpy()
                labels = res["labels"]
                for box, score, label in zip(boxes, scores, labels):
                    if isinstance(label, str):
                        if label.lower() != "person":
                            continue
                        cat_id = 1
                    else:
                        cat_id = GDINO_LABEL_TO_LLVIP.get(int(label))
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
            print(f"\r  GDINO 推理进度: {processed}/{total}", end="", flush=True)

    print()
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLVIP 上 GDINO + CSMA 零样本评估（teacher 匹配）"
    )
    parser.add_argument("--ir-dir", type=str, default=LLVIP_IR_DIR)
    parser.add_argument("--gt-json", type=str, default=LLVIP_GT_JSON)
    parser.add_argument("--input-mode", type=str, default="pseudo_rgb",
                        choices=["pseudo_rgb", "ir_pipeline"])
    parser.add_argument("--ckpt", type=str, default="outputs_csma/ckpt/best_stage1.pt")
    parser.add_argument("--out-json", type=str,
                        default="outputs_csma/logs/llvip/eval_llvip_gdino_csma.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--box-threshold", type=float, default=0.05)
    parser.add_argument("--text-threshold", type=float, default=0.05)
    args = parser.parse_args()

    if args.input_mode == "pseudo_rgb" and not args.ckpt:
        raise ValueError("pseudo_rgb 需要 --ckpt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CSMAConfig()
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    _configure_processor(processor, cfg.img_size)

    print(f"[eval_llvip_gdino] 设备: {device}")
    print(f"[eval_llvip_gdino] input_mode: {args.input_mode}")

    dino = GroundingDinoForObjectDetection.from_pretrained(cfg.model_id).to(device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False

    csma: Optional[CSMA] = None
    if args.input_mode == "pseudo_rgb":
        print(f"[eval_llvip_gdino] CSMA: {args.ckpt}")
        csma = _load_csma(args.ckpt, cfg, device)
    else:
        csma = CSMA(cfg).to(device)  # dummy, unused

    dataset = LLVIPIRDataset(args.ir_dir, args.gt_json, processor, cfg)
    coco_gt = _build_llvip_gt(dataset)

    predictions = run_gdino_llvip_eval(
        csma=csma,
        dino=dino,
        processor=processor,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        input_mode=args.input_mode,
    )

    results = compute_map(coco_gt, predictions)
    results.update({
        "input_mode": args.input_mode,
        "ckpt": args.ckpt if args.input_mode == "pseudo_rgb" else None,
        "dataset": "LLVIP",
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "teacher_matched": True,
    })

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 55)
    print(f"  input_mode    : {args.input_mode}")
    print(f"  mAP@0.5       : {results['map_50']:.4f}")
    print(f"  AP_person@0.5 : {results['ap_person']:.4f}")
    print(f"  预测框 / GT框  : {results['n_preds']} / {results['n_gt']}")
    print("=" * 55)
    print(f"[eval_llvip_gdino] 结果: {args.out_json}")


if __name__ == "__main__":
    main()
