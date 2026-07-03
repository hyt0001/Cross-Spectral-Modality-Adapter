"""
FLIR ADAS v1 配对数据集适配器。

数据集目录结构（train / val 相同）：
    {root}/thermal_annotations.json  — COCO 格式标注（image.file_name 含 thermal_8_bit/ 前缀）
    {root}/thermal_8_bit/FLIR_XXXXX.jpeg  — 8-bit 热红外图像
    {root}/RGB/FLIR_XXXXX.jpg             — 对应可见光图像（同名，扩展名不同）

与 FLIR_ADAS_v2（Roboflow）的关键差异：
  - 标注文件：thermal_annotations.json（非 coco.json）
  - file_name 含子目录前缀：thermal_8_bit/FLIR_XXXXX.jpeg
  - RGB 通过文件名直接配对（FLIR_XXXXX.jpg），配对成功率 > 99%
  - 类别 ID：person=1, car=3（与 flir_v2 prompt 一致）
  - 支持完整 loss_mode=full（L_det + L_align 同时使用）

对应 docs/TD.md §1.4 RGB-IR 配对逻辑，docs/architecture.md §11 文件结构。
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


# FLIR v1 类别 ID 映射（与 thermal_annotations.json 一致）
# person: cat_id=1 → class_idx=0  （prompt 左起第一个）
# car:    cat_id=3 → class_idx=1  （prompt 左起第二个）
FLIR_V1_CATEGORY_TO_CLASS_IDX: Dict[int, int] = {1: 0, 3: 1}
FLIR_V1_VALID_CAT_IDS: frozenset = frozenset({1, 3})
FLIR_V1_TEXT_PROMPT: str = "person. car."


def build_flir_v1_category_map(
    text_prompt: str,
) -> Tuple[Dict[int, int], frozenset]:
    """
    校验 text_prompt 并返回 (cat_id→class_idx 映射, 有效 cat_id 集合)。

    Args:
        text_prompt: 与训练时相同的文本，如 "person. car."

    Returns:
        (category_map, valid_cat_ids)
    """
    normalized = text_prompt.strip().lower().rstrip(".")
    segments = [s.strip() for s in normalized.split(".") if s.strip()]
    if segments != ["person", "car"]:
        raise ValueError(
            f"FLIR v1 当前仅支持 prompt 'person. car.'，"
            f"当前解析为: {segments}"
        )
    return dict(FLIR_V1_CATEGORY_TO_CLASS_IDX), frozenset(FLIR_V1_VALID_CAT_IDS)


class FlirV1PairedDataset(Dataset):
    """
    FLIR ADAS v1 RGB-IR 配对数据集。

    每个样本返回：
        pixel_values      [3, H, W]  热红外图像（ImageNet 归一化）
        pixel_mask        [H, W]     有效像素掩码
        labels            Dict       DINO 格式目标框（cxcywh 归一化）
        rgb_pixel_values  [3, H, W]  对应 RGB 图像（ImageNet 归一化）；
                                     若 RGB 文件不存在则为 None
        image_path        str        热红外图像的绝对路径
        rgb_path          str | None RGB 图像的绝对路径（或 None）

    RGB 配对策略：
        IR  : {root}/thermal_8_bit/FLIR_XXXXX.jpeg
        RGB : {root}/RGB/FLIR_XXXXX.jpg
        取 IR 文件名的 stem（FLIR_XXXXX），在 RGB/ 目录下查找同名 .jpg。
    """

    def __init__(
        self,
        root: str,
        processor: Any,
        text_prompt: str,
        category_map: Optional[Dict[int, int]] = None,
        valid_cat_ids: Optional[frozenset] = None,
        ir_aug: Optional[Callable[[Image.Image], Image.Image]] = None,
    ) -> None:
        """
        Args:
            root:          FLIR v1 split 目录（如 FLIR_License/train）。
            processor:     AutoProcessor（GroundingDinoImageProcessor）。
            text_prompt:   检测 prompt，如 "person. car."
            category_map:  cat_id → class_idx 映射；None 时使用默认 {1:0, 3:1}。
            valid_cat_ids: 有效 cat_id 集合；None 时使用 {1, 3}。
            ir_aug:        可选的 IR 增强 callable（PIL→PIL），训练时传入以改善跨域泛化。
        """
        super().__init__()
        self._root = os.path.abspath(root)
        self._processor = processor
        self._text_prompt = text_prompt
        self._ir_aug = ir_aug
        self._cat_map: Dict[int, int] = (
            category_map if category_map is not None
            else dict(FLIR_V1_CATEGORY_TO_CLASS_IDX)
        )
        self._valid_ids: frozenset = (
            valid_cat_ids if valid_cat_ids is not None
            else frozenset(FLIR_V1_VALID_CAT_IDS)
        )

        ann_path = os.path.join(self._root, "thermal_annotations.json")
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"未找到标注文件: {ann_path}")

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
            iid = a["image_id"]
            self._id_to_anns.setdefault(iid, []).append(a)

        assert len(self._images) > 0, (
            f"过滤后无有效图像（root={self._root}），"
            f"请检查 thermal_annotations.json 中 category_id 是否包含 {self._valid_ids}"
        )

        # 预构建 RGB 目录索引（stem → 完整路径），加速 __getitem__ 查找
        rgb_dir = os.path.join(self._root, "RGB")
        self._rgb_stem_to_path: Dict[str, str] = {}
        if os.path.isdir(rgb_dir):
            for fn in os.listdir(rgb_dir):
                stem = os.path.splitext(fn)[0]
                self._rgb_stem_to_path[stem] = os.path.join(rgb_dir, fn)

        paired = sum(
            1 for img in self._images
            if os.path.splitext(os.path.basename(img["file_name"]))[0]
            in self._rgb_stem_to_path
        )
        print(
            f"[FlirV1PairedDataset] root={self._root}  "
            f"有效图像={len(self._images)}  RGB配对={paired}"
        )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_info = self._images[index]
        # file_name 形如 thermal_8_bit/FLIR_00001.jpeg
        img_path = os.path.join(self._root, img_info["file_name"])
        stem = os.path.splitext(os.path.basename(img_info["file_name"]))[0]

        image = Image.open(img_path).convert("RGB")
        if self._ir_aug is not None:
            image = self._ir_aug(image)
        anns = self._id_to_anns.get(img_info["id"], [])

        coco_anns: List[Dict[str, Any]] = []
        for ann in anns:
            cid = int(ann["category_id"])
            if cid not in self._valid_ids:
                continue
            coco_anns.append(
                {
                    "category_id": self._cat_map[cid],
                    "bbox": [float(x) for x in ann["bbox"]],
                    "area": float(ann["area"]),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                }
            )

        coco_target: Dict[str, Any] = {
            "image_id": int(img_info["id"]),
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

        # RGB 配对：resize 到与 IR 相同的原始尺寸，确保 processor 输出形状一致
        rgb_path = self._rgb_stem_to_path.get(stem)
        rgb_pixel_values: Optional[torch.Tensor] = None
        if rgb_path is not None and os.path.isfile(rgb_path):
            rgb_img = Image.open(rgb_path).convert("RGB")
            # IR 原始尺寸（保证两路图像经 processor resize+pad 后形状相同）
            ir_w, ir_h = int(img_info["width"]), int(img_info["height"])
            if rgb_img.size != (ir_w, ir_h):
                rgb_img = rgb_img.resize((ir_w, ir_h), Image.BILINEAR)
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


def collate_flir_v1(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    FLIR v1 配对数据集 collate 函数。

    - pixel_values / pixel_mask / labels：与 collate_fn 相同逻辑
    - rgb_pixel_values：全有则 stack，任一缺失则不写入（训练循环依据键是否存在决定是否计算 L_align）
    - image_paths / rgb_paths：保留为列表（供评估反查 image_id）

    Args:
        batch: List[Dict]，FlirV1PairedDataset.__getitem__ 的输出

    Returns:
        collated dict，key 说明见上
    """
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_mask   = torch.stack([b["pixel_mask"]   for b in batch])
    labels       = [b["labels"] for b in batch]

    result: Dict[str, Any] = {
        "pixel_values": pixel_values,
        "pixel_mask":   pixel_mask,
        "labels":       labels,
        "image_paths":  [b["image_path"] for b in batch],
        "rgb_paths":    [b.get("rgb_path") for b in batch],
    }

    rgb_list: List[Optional[torch.Tensor]] = [b.get("rgb_pixel_values") for b in batch]
    if all(v is not None for v in rgb_list):
        result["rgb_pixel_values"] = torch.stack(rgb_list)  # type: ignore[arg-type]

    return result
