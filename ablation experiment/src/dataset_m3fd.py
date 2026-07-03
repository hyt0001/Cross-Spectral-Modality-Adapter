"""
M3FD 多光谱配对数据集适配器。

数据集目录结构（train / val 相同）：
    {root}/annotations_coco.json  — COCO 格式标注（convert_m3fd_to_coco.py 输出）
    {root}/ir/{image}.png         — 红外图像
    {root}/vi/{image}.png         — 可见光（RGB）图像（同名，配对）

COCO 类别约定（与 convert_m3fd_to_coco.py 保持一致）：
    cat_id=1  → person  → class_idx=0  （prompt 左起第一个）
    cat_id=2  → car     → class_idx=1  （prompt 左起第二个）
    其他类别已在转换时过滤

与 FlirV1PairedDataset 的关键差异：
  - 标注文件：annotations_coco.json（非 thermal_annotations.json）
  - IR 图像目录：ir/（非 thermal_8_bit/）
  - RGB 图像目录：vi/（非 RGB/），同 stem 直接配对
  - cat_id 映射：{1:0, 2:1}（而非 {1:0, 3:1}）
  - 文本 prompt：person. car.（与 FLIR 完全相同）

对应 docs/TD.md §1.4 RGB-IR 配对逻辑。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


# M3FD COCO 标注 category_id → class_idx（对应 "person. car." prompt）
M3FD_CATEGORY_TO_CLASS_IDX: Dict[int, int] = {1: 0, 2: 1}
M3FD_VALID_CAT_IDS: frozenset = frozenset({1, 2})
M3FD_TEXT_PROMPT: str = "person. car."
M3FD_ANN_FILENAME: str = "annotations_coco.json"


def build_m3fd_category_map(
    text_prompt: str,
) -> Tuple[Dict[int, int], frozenset]:
    """
    校验 text_prompt 并返回 (cat_id→class_idx 映射, 有效 cat_id 集合)。

    Args:
        text_prompt: 与训练时相同的文本，如 "person. car."

    Returns:
        (category_map, valid_cat_ids)

    Raises:
        ValueError: 若 prompt 不匹配。
    """
    normalized = text_prompt.strip().lower().rstrip(".")
    segments = [s.strip() for s in normalized.split(".") if s.strip()]
    if segments != ["person", "car"]:
        raise ValueError(
            f"M3FD 适配器当前仅支持 prompt 'person. car.'，"
            f"当前解析为: {segments}"
        )
    return dict(M3FD_CATEGORY_TO_CLASS_IDX), frozenset(M3FD_VALID_CAT_IDS)


class M3FDPairedDataset(Dataset):
    """
    M3FD 红外-可见光配对数据集。

    每个样本返回：
        pixel_values      [3, H, W]  红外图像（ImageNet 归一化）
        pixel_mask        [H, W]     有效像素掩码
        labels            Dict       DINO 格式目标框（cxcywh 归一化）
        rgb_pixel_values  [3, H, W]  对应可见光图像；若缺失则为 None
        image_path        str        红外图像的绝对路径
        rgb_path          str | None 可见光图像的绝对路径（或 None）

    RGB 配对策略：
        IR  : {root}/ir/{stem}.png
        RGB : {root}/vi/{stem}.png
        取 IR 文件名的 stem，在 vi/ 目录下查找同名文件（任意扩展名）。
    """

    def __init__(
        self,
        root: str,
        processor: Any,
        text_prompt: str,
        category_map: Optional[Dict[int, int]] = None,
        valid_cat_ids: Optional[frozenset] = None,
        ann_filename: str = M3FD_ANN_FILENAME,
        canonical_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            root:           M3FD split 目录（含 annotations_coco.json + ir/ + vi/）。
            processor:      AutoProcessor（GroundingDinoImageProcessor）。
            text_prompt:    检测 prompt，如 "person. car."
            category_map:   cat_id → class_idx 映射；None 时使用 {1:0, 2:1}。
            valid_cat_ids:  有效 cat_id 集合；None 时使用 {1, 2}。
            ann_filename:   标注文件名，默认 "annotations_coco.json"。
            canonical_size: 可选 (W, H)；设置后在送 processor 前将图像统一缩放到该尺寸
                            并同步缩放 bbox（训练推荐 (1024, 768)，保证 batch 同形）。
                            None=保持原始分辨率（评测时使用，batch 由 collate 补边对齐）。
        """
        super().__init__()
        self._canonical_size: Optional[Tuple[int, int]] = canonical_size
        self._root = os.path.abspath(root)
        self._processor = processor
        self._text_prompt = text_prompt
        self._cat_map: Dict[int, int] = (
            category_map if category_map is not None
            else dict(M3FD_CATEGORY_TO_CLASS_IDX)
        )
        self._valid_ids: frozenset = (
            valid_cat_ids if valid_cat_ids is not None
            else frozenset(M3FD_VALID_CAT_IDS)
        )

        # 加载 COCO 标注
        ann_path = os.path.join(self._root, ann_filename)
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(
                f"未找到 M3FD 标注文件: {ann_path}\n"
                f"请先运行: python src/convert_m3fd_to_coco.py"
            )
        with open(ann_path, encoding="utf-8") as f:
            coco = json.load(f)

        # 仅保留含有效类别（person / car）标注的图像
        valid_image_ids: set = {
            a["image_id"]
            for a in coco["annotations"]
            if int(a["category_id"]) in self._valid_ids
        }

        self._images: List[Dict[str, Any]] = sorted(
            [img for img in coco["images"] if img["id"] in valid_image_ids],
            key=lambda x: x["id"],
        )

        self._id_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for a in coco["annotations"]:
            cid = int(a["category_id"])
            if cid not in self._valid_ids:
                continue
            self._id_to_anns.setdefault(a["image_id"], []).append(a)

        assert len(self._images) > 0, (
            f"过滤后无有效图像（root={self._root}），"
            f"请检查 {ann_filename} 中 category_id 是否包含 {self._valid_ids}"
        )

        # 预构建 vi/ 目录索引（stem → 完整路径），加速 __getitem__ 查找
        vi_dir = os.path.join(self._root, "vi")
        self._vi_stem_to_path: Dict[str, str] = {}
        if os.path.isdir(vi_dir):
            for fn in os.listdir(vi_dir):
                stem = os.path.splitext(fn)[0]
                self._vi_stem_to_path[stem] = os.path.join(vi_dir, fn)

        paired = sum(
            1 for img in self._images
            if os.path.splitext(img["file_name"])[0] in self._vi_stem_to_path
        )
        print(
            f"[M3FDPairedDataset] root={self._root}  "
            f"有效图像={len(self._images)}  RGB配对={paired}"
        )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_info = self._images[index]
        # file_name 为 ir/ 下的文件名（如 "00001I.png"）
        ir_filename = img_info["file_name"]
        img_path = os.path.join(self._root, "ir", ir_filename)
        stem = os.path.splitext(ir_filename)[0]

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        # canonical_size 设置时统一缩放（训练用）；None 时保持原始分辨率（评测用）
        if self._canonical_size is not None and image.size != self._canonical_size:
            image = image.resize(self._canonical_size, Image.BILINEAR)
            scale_x = self._canonical_size[0] / orig_w
            scale_y = self._canonical_size[1] / orig_h
        else:
            scale_x = 1.0
            scale_y = 1.0

        anns = self._id_to_anns.get(img_info["id"], [])

        coco_anns: List[Dict[str, Any]] = []
        for ann in anns:
            cid = int(ann["category_id"])
            if cid not in self._valid_ids:
                continue
            x, y, w, h = ann["bbox"]
            x, w = x * scale_x, w * scale_x
            y, h = y * scale_y, h * scale_y
            coco_anns.append(
                {
                    "category_id": self._cat_map[cid],
                    "bbox":        [float(x), float(y), float(w), float(h)],
                    "area":        float(w * h),
                    "iscrowd":     int(ann.get("iscrowd", 0)),
                }
            )

        coco_target: Dict[str, Any] = {
            "image_id":    int(img_info["id"]),
            "annotations": coco_anns,
        }

        img_enc = self._processor.image_processor(
            images=image,
            annotations=coco_target,
            return_tensors="pt",
        )

        pixel_values = img_enc["pixel_values"][0]
        pixel_mask   = img_enc["pixel_mask"][0]
        labels_dict  = img_enc["labels"][0]
        labels = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in labels_dict.items()
        }

        # RGB 配对：与 IR 保持相同尺寸（canonical 或原始）
        rgb_path = self._vi_stem_to_path.get(stem)
        rgb_pixel_values: Optional[torch.Tensor] = None
        if rgb_path is not None and os.path.isfile(rgb_path):
            rgb_img = Image.open(rgb_path).convert("RGB")
            target_wh = (
                self._canonical_size
                if self._canonical_size is not None
                else (orig_w, orig_h)
            )
            if rgb_img.size != target_wh:
                rgb_img = rgb_img.resize(target_wh, Image.BILINEAR)
            rgb_enc = self._processor.image_processor(
                images=rgb_img,
                return_tensors="pt",
            )
            rgb_pixel_values = rgb_enc["pixel_values"][0]  # [3, H, W]

        return {
            "pixel_values":     pixel_values,
            "pixel_mask":       pixel_mask,
            "labels":           labels,
            "rgb_pixel_values": rgb_pixel_values,
            "image_path":       img_path,
            "rgb_path":         rgb_path,
        }


def _pad_stack_tensors(
    tensors: List[torch.Tensor],
    pad_value: float,
) -> torch.Tensor:
    """
    将 [C,H,W] 或 [H,W] 张量列表右下补边（padding）到 batch 内最大 H/W 后 stack。

    M3FD 含多种原始分辨率，processor 保宽高比 resize 后单张形状一致
    但 batch 内可能不同，须补边后才能 stack（不拉伸图像，保持长宽比）。

    Args:
        tensors:   待合并的张量列表，ndim 须均为 2 或 3。
        pad_value: 补边填充值（pixel_values 与 pixel_mask 均用 0）。

    Returns:
        [B, C, H, W] 或 [B, H, W] 张量。
    """
    if not tensors:
        raise ValueError("tensors 不能为空")
    ndim = tensors[0].ndim
    if ndim not in (2, 3):
        raise ValueError(f"不支持的 ndim={ndim}，仅支持 2 或 3")

    max_h = max(t.shape[-2] for t in tensors)
    max_w = max(t.shape[-1] for t in tensors)

    padded: List[torch.Tensor] = []
    for t in tensors:
        pad_h = max_h - t.shape[-2]
        pad_w = max_w - t.shape[-1]
        if pad_h or pad_w:
            # F.pad 参数顺序: (left, right, top, bottom)
            t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), value=pad_value)
        padded.append(t)
    return torch.stack(padded, dim=0)


def collate_m3fd(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    M3FD 配对数据集 collate 函数（支持 batch 内混合分辨率）。

    M3FD 含多种原始分辨率，processor 保宽高比 resize 后 batch 内尺寸可能不同。
    本函数将 pixel_values / pixel_mask 补边到 batch 内最大 H/W，同时将
    labels["boxes"]（cxcywh，按原图归一化）重新归一化到补边后的坐标系，
    确保 DINO loss 匹配器看到正确的框坐标。

    Args:
        batch: List[Dict]，M3FDPairedDataset.__getitem__ 的输出

    Returns:
        collated dict，含 pixel_values / pixel_mask / labels /
        image_paths / rgb_paths / rgb_pixel_values（仅当全部有效时）
    """
    # 记录每张图补边前的 (H, W)，用于 box 坐标缩放
    orig_hw: List[Tuple[int, int]] = [
        (b["pixel_values"].shape[-2], b["pixel_values"].shape[-1]) for b in batch
    ]

    pixel_values = _pad_stack_tensors(
        [b["pixel_values"] for b in batch], pad_value=0.0
    )
    pixel_mask = _pad_stack_tensors(
        [b["pixel_mask"] for b in batch], pad_value=0.0
    )
    h_max = pixel_values.shape[-2]
    w_max = pixel_values.shape[-1]

    # boxes 归一化坐标从原尺寸坐标系缩放到补边后坐标系
    labels: List[Dict[str, Any]] = []
    for b, (h_i, w_i) in zip(batch, orig_hw):
        entry: Dict[str, Any] = {}
        for k, v in b["labels"].items():
            if k == "boxes" and isinstance(v, torch.Tensor) and v.numel() > 0:
                boxes = v.clone()
                sx = w_i / w_max
                sy = h_i / h_max
                boxes[:, 0] = boxes[:, 0] * sx   # cx
                boxes[:, 1] = boxes[:, 1] * sy   # cy
                boxes[:, 2] = boxes[:, 2] * sx   # w
                boxes[:, 3] = boxes[:, 3] * sy   # h
                entry[k] = boxes
            else:
                entry[k] = v.clone() if isinstance(v, torch.Tensor) else v
        labels.append(entry)

    result: Dict[str, Any] = {
        "pixel_values": pixel_values,
        "pixel_mask":   pixel_mask,
        "labels":       labels,
        "image_paths":  [b["image_path"] for b in batch],
        "rgb_paths":    [b.get("rgb_path") for b in batch],
    }

    rgb_list: List[Optional[torch.Tensor]] = [b.get("rgb_pixel_values") for b in batch]
    if all(v is not None for v in rgb_list):
        result["rgb_pixel_values"] = _pad_stack_tensors(
            rgb_list,  # type: ignore[arg-type]
            pad_value=0.0,
        )

    return result
