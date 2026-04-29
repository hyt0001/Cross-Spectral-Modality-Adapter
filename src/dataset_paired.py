"""
RGB-IR 配对数据集：在 FlirCocoOverfitDataset 基础上同步加载对应 RGB 图像。

对应 docs/TD.md §1.4。
- RGB 图像路径：与 IR 文件名相同，在 rgb_root 目录下查找。
- 若 RGB 不存在，rgb_pixel_values 置 None，训练时跳过 L_align。
- collate_paired 采用"全有或全无"策略：
    batch 中全部样本都有 RGB → 输出 rgb_pixel_values Tensor[B,3,H,W]
    任一样本缺少 RGB      → 该键不出现，训练循环凭 "rgb_pixel_values" in batch 判断
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from src.dataset import FlirCocoOverfitDataset, collate_fn


class FlirPairedDataset(FlirCocoOverfitDataset):
    """
    在 COCO 格式红外数据集基础上，额外加载对应 RGB 配对图像。

    RGB 路径规则：取 IR 图像文件名，在 rgb_root 目录下查找同名文件。
    若 RGB 不存在，则 rgb_pixel_values 为 None，训练时跳过 L_align。
    """

    def __init__(
        self,
        ir_root: str,
        rgb_root: str,
        processor: Any,
        text_prompt: str,
        coco_category_id_to_class_idx: Dict[int, int],
    ) -> None:
        super().__init__(ir_root, processor, text_prompt, coco_category_id_to_class_idx)
        self._rgb_root = os.path.abspath(rgb_root)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = super().__getitem__(index)

        ir_filename = Path(sample["image_path"]).name
        rgb_path = Path(self._rgb_root) / ir_filename

        if rgb_path.exists():
            rgb_img = Image.open(rgb_path).convert("RGB")
            rgb_enc = self._processor.image_processor(
                images=rgb_img,
                return_tensors="pt",
            )
            sample["rgb_pixel_values"] = rgb_enc["pixel_values"][0]  # [3, H, W]
        else:
            sample["rgb_pixel_values"] = None

        return sample


def collate_paired(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    在 collate_fn 基础上额外处理 rgb_pixel_values（可能含 None）。

    "全有或全无"策略：
        - batch 中所有样本均有 rgb_pixel_values → stack 为 Tensor[B,3,H,W] 并写入输出
        - 任一样本缺少 rgb_pixel_values      → 该键不出现在输出中
    """
    base: Dict[str, Any] = collate_fn(batch)

    rgb_list: List[Optional[torch.Tensor]] = [
        b.get("rgb_pixel_values") for b in batch
    ]

    if all(v is not None for v in rgb_list):
        base["rgb_pixel_values"] = torch.stack(rgb_list)  # type: ignore[arg-type]

    return base
