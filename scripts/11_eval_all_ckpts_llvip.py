"""
批量评测所有 CSMA checkpoint 在 LLVIP test 集上的 mAP@0.5。

优化：DINO 和 dataset 只初始化一次，逐个替换 CSMA 权重，大幅减少重复开销。

用法：
    CUDA_VISIBLE_DEVICES=0 conda run -n RGBtest python scripts/11_eval_all_ckpts_llvip.py
    CUDA_VISIBLE_DEVICES=0 conda run -n RGBtest python scripts/11_eval_all_ckpts_llvip.py \
        --batch-size 8 --out-json outputs_csma_v3/logs/eval_all_ckpts_llvip.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, Any, List

import torch
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from torch.utils.data import DataLoader

# 项目根目录加入 sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from src.config import CSMAConfig
from src.csma import CSMA
from src.dataset_llvip import LLVIPPairedDataset, collate_llvip, build_llvip_category_map
from src.eval_csma import _build_gt_coco, _build_path_to_id, compute_map, run_eval


def _collect_ckpts(root: str) -> List[str]:
    """
    收集指定目录（含子目录）中所有 .pt 文件，按目录名+文件名排序。
    排除 latest.pt（与 epoch_xxxx.pt 重复）。
    """
    pts = sorted(glob.glob(os.path.join(root, "**", "*.pt"), recursive=True))
    pts = [p for p in pts if os.path.basename(p) != "latest.pt"]
    return pts


def main() -> None:
    parser = argparse.ArgumentParser(description="批量评测所有 CSMA checkpoint @ LLVIP test")
    parser.add_argument("--ckpt-roots",   nargs="+",
                        default=["outputs_csma", "outputs_csma_v3"],
                        help="包含 ckpt/*.pt 的根目录列表（支持多个）")
    parser.add_argument("--data-root",    default="LLVIP/LLVIP")
    parser.add_argument("--ann-file",     default="LLVIP/annotations/val.json")
    parser.add_argument("--text-prompt",  default="person.")
    parser.add_argument("--batch-size",   type=int, default=8)
    parser.add_argument("--box-threshold",  type=float, default=0.05)
    parser.add_argument("--text-threshold", type=float, default=0.05)
    parser.add_argument("--out-json",
                        default="outputs_csma_v3/logs/eval_all_ckpts_llvip.json")
    args = parser.parse_args()

    os.chdir(_PROJECT_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[batch_eval] 设备: {device}")

    # ── Phase 1: 收集所有 ckpt ────────────────────────────────────────────────
    all_pts: List[str] = []
    for root in args.ckpt_roots:
        pts = _collect_ckpts(root)
        all_pts.extend(pts)
    print(f"[batch_eval] 共找到 {len(all_pts)} 个 checkpoint:")
    for p in all_pts:
        size_mb = os.path.getsize(p) / 1024 / 1024
        print(f"  {p}  ({size_mb:.1f} MB)")

    # ── Phase 2: 初始化 DINO（只做一次）──────────────────────────────────────
    cfg = CSMAConfig()
    model_id = cfg.model_id
    print(f"\n[batch_eval] 加载 processor & DINO ({model_id})...")
    processor = AutoProcessor.from_pretrained(model_id)
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
    print(f"  processor size: {ip.size}")

    dino = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False
    print("  DINO 已加载并冻结")

    # ── Phase 3: 初始化 LLVIP 数据集（只做一次）──────────────────────────────
    text_prompt = args.text_prompt
    cat_map, valid_ids = build_llvip_category_map(text_prompt)
    dataset = LLVIPPairedDataset(
        root=args.data_root,
        processor=processor,
        text_prompt=text_prompt,
        split="test",
        ann_file=args.ann_file,
        category_map=cat_map,
        valid_cat_ids=valid_ids,
    )
    coco_gt = _build_gt_coco(dataset, valid_ids, "llvip")
    print(f"[batch_eval] 数据集: {len(dataset)} 张  GT 标注: {len(coco_gt.anns)}")

    # ── Phase 4: 逐个 checkpoint 评测 ─────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    total = len(all_pts)

    for idx, ckpt_path in enumerate(all_pts, 1):
        rel_path = os.path.relpath(ckpt_path, _PROJECT_ROOT)
        print(f"\n[{idx:2d}/{total}] checkpoint: {rel_path}")
        t0 = time.time()

        csma = CSMA(cfg).to(device)
        raw = torch.load(ckpt_path, map_location=device, weights_only=True)
        state = raw["csma"] if isinstance(raw, dict) and "csma" in raw else raw
        csma.load_state_dict(state)
        csma.eval()

        preds = run_eval(
            csma=csma,
            dino=dino,
            processor=processor,
            dataset=dataset,
            device=device,
            text_prompt=text_prompt,
            batch_size=args.batch_size,
            num_workers=2,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            dataset_mode="llvip",
        )
        metrics = compute_map(coco_gt, preds)
        elapsed = time.time() - t0

        row: Dict[str, Any] = {
            "ckpt":       rel_path,
            "map_50":     round(metrics["map_50"], 4),
            "map_50_95":  round(metrics["map_50_95"], 4),
            "ap_person":  round(metrics["ap_person"], 4),
            "n_preds":    metrics["n_preds"],
            "elapsed_s":  round(elapsed, 1),
        }
        results.append(row)
        print(
            f"  mAP@0.5={row['map_50']:.4f}  "
            f"AP_person={row['ap_person']:.4f}  "
            f"preds={row['n_preds']}  "
            f"耗时={row['elapsed_s']:.0f}s"
        )

        del csma
        torch.cuda.empty_cache()

    # ── Phase 5: 汇总输出 ─────────────────────────────────────────────────────
    baseline_map = 0.8572  # 纯 DINO baseline（无 CSMA）
    results_sorted = sorted(results, key=lambda r: r["map_50"], reverse=True)

    print("\n" + "=" * 70)
    print(f"  LLVIP test 批量评测结果（baseline DINO = {baseline_map:.4f}）")
    print("=" * 70)
    print(f"  {'checkpoint':<50} {'mAP@0.5':>8} {'vs base':>8} {'AP_per':>8}")
    print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*8}")
    for r in results_sorted:
        delta = r["map_50"] - baseline_map
        flag = "↑" if delta >= 0 else "↓"
        print(
            f"  {r['ckpt']:<50} {r['map_50']:>8.4f} "
            f"{flag}{abs(delta):>6.4f} {r['ap_person']:>8.4f}"
        )
    print("=" * 70)

    best = results_sorted[0]
    print(f"\n  最佳 checkpoint: {best['ckpt']}")
    print(f"  最佳 mAP@0.5:   {best['map_50']:.4f}")

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_map_50": baseline_map,
            "total_checkpoints": len(results),
            "best": best,
            "results": results_sorted,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[batch_eval] 结果已保存: {args.out_json}")


if __name__ == "__main__":
    main()
