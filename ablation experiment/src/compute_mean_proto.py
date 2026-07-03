"""
离线均值原型预计算脚本（Ablation B-2 前置步骤）。

功能：
  遍历训练集全部 RGB 图像，经冻结 DINO backbone 提取多尺度特征，
  对所有样本的所有 patch 维度求均值，得到 mean_proto [proto_dim]，
  保存为 .pt 文件，供 CSMAMeanProto 初始化使用。

运行方式（一次性执行）：
  CUDA_VISIBLE_DEVICES=0 conda run -n RGBtest python -m src.compute_mean_proto \\
      --data-root FLIR_License/train \\
      --out outputs_csma/mean_proto.pt

对应 docs/实验实施细节.md §1.3（B-2 运行顺序 Step 1）。
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from src.config import CSMAConfig
from src.dataset_flir_v1 import FlirV1PairedDataset, build_flir_v1_category_map, collate_flir_v1
from src.train_csma import extract_dino_backbone_features


def compute_mean_proto(
    dino: GroundingDinoForObjectDetection,
    loader: DataLoader,
    device: torch.device,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    遍历全部训练集 RGB 图像，提取 DINO 多尺度特征后对所有 patch 求均值。

    Args:
        dino:           冻结的 GroundingDinoForObjectDetection。
        loader:         训练集 DataLoader（需包含 rgb_pixel_values）。
        device:         运算设备。
        input_ids:      文本编码 [1, T]。
        attention_mask: 文本注意力掩码 [1, T]。

    Returns:
        mean_proto: [proto_dim] float32 CPU tensor，所有样本所有 patch 特征的全局均值。
    """
    # 使用 sum + count 精确累加，避免批次大小不等时 mean-of-means 的偏差
    # （最后一批 batch_size 可能 < 其他批，简单平均会高估其权重）
    total_sum: Optional[torch.Tensor] = None
    total_count: int = 0
    n_skipped = 0
    n_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if "rgb_pixel_values" not in batch:
                n_skipped += 1
                continue
            rgb_pv = batch["rgb_pixel_values"].to(device)   # [B, 3, H, W]

            # extract_dino_backbone_features 返回 [B, L_total, 256]，语义特征不是像素
            feat_rgb = extract_dino_backbone_features(
                dino, rgb_pv, input_ids, attention_mask
            )  # [B, L, D]

            B, L, D = feat_rgb.shape
            # 精确累加：对当前 batch 的 B*L 个 patch 特征向量求和
            batch_sum = feat_rgb.sum(dim=[0, 1]).cpu()      # [D]，精确求和，非均值
            if total_sum is None:
                total_sum = batch_sum
            else:
                total_sum += batch_sum
            total_count += B * L                             # 记录真实 patch 总数
            n_batches += 1

            if (i + 1) % 50 == 0:
                print(f"[compute_mean_proto] 已处理 {i + 1} batch，累计 {total_count:,} patch...")

    if total_sum is None or total_count == 0:
        raise RuntimeError(
            "未收集到任何 RGB 特征，请检查数据集是否包含 rgb_pixel_values。"
        )

    # 全局精确均值：总 patch 特征之和 / 总 patch 数
    mean_proto = total_sum / total_count                     # [D]
    print(
        f"[compute_mean_proto] 共处理 {n_batches} batch，"
        f"跳过 {n_skipped} batch（无 rgb_pixel_values），"
        f"总 patch 数: {total_count:,}，"
        f"mean_proto shape={tuple(mean_proto.shape)}，"
        f"norm={mean_proto.norm().item():.4f}"
    )
    return mean_proto


def main() -> None:
    """均值原型预计算入口。"""
    parser = argparse.ArgumentParser(description="Ablation B-2：离线计算 RGB 特征均值原型")
    parser.add_argument(
        "--data-root", type=str, default="FLIR_License/train",
        help="FLIR v1 训练集根目录（含 thermal_annotations.json + thermal_8_bit/ + RGB/）",
    )
    parser.add_argument(
        "--out", type=str, default="outputs_csma/mean_proto.pt",
        help="输出文件路径（.pt），保存 mean_proto [proto_dim] tensor",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="DataLoader batch size（越大越快，按显存调整）",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader 并行 worker 数",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[compute_mean_proto] 设备: {device}")

    # 使用与主实验相同的 CSMAConfig 默认值获取 model_id 和 text_prompt
    cfg = CSMAConfig()

    # 加载处理器（限制图像尺寸，与训练一致）
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
        ip = processor.image_processor
        try:
            cur_se = ip.size.shortest_edge or 0
        except AttributeError:
            cur_se = ip.size.get("shortest_edge", 0) or 0
        if cur_se > cfg.img_size:
            try:
                ip.size.shortest_edge = cfg.img_size
                ip.size.longest_edge = cfg.img_size * 2
            except AttributeError:
                ip.size = {"shortest_edge": cfg.img_size, "longest_edge": cfg.img_size * 2}

    tokenizer = processor.tokenizer
    encoded = tokenizer(cfg.text_prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # 加载并冻结 DINO
    dino: GroundingDinoForObjectDetection = GroundingDinoForObjectDetection.from_pretrained(
        cfg.model_id
    )
    dino.to(device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    print("[compute_mean_proto] DINO 加载完成，参数已冻结")

    # 构建数据集（仅需 rgb_pixel_values，不需要 labels）
    category_map, valid_cat_ids = build_flir_v1_category_map(cfg.text_prompt)
    dataset = FlirV1PairedDataset(
        root=args.data_root,
        processor=processor,
        text_prompt=cfg.text_prompt,
        category_map=category_map,
        valid_cat_ids=valid_cat_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_flir_v1,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[compute_mean_proto] 数据集大小: {len(dataset)} 样本，{len(loader)} batch")

    # 预计算均值原型
    mean_proto = compute_mean_proto(dino, loader, device, input_ids, attention_mask)

    # 保存
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    torch.save(mean_proto, args.out)
    print(f"[compute_mean_proto] 保存至: {args.out}")


if __name__ == "__main__":
    main()
