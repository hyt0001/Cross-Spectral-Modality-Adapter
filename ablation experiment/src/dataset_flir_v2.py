"""
FLIR_ADAS_v2 数据集适配器。

与旧版 FlirCocoOverfitDataset 的差异：
  - 标注文件：coco.json（非 _annotations.coco.json）
  - 图像路径：{root}/{file_name}，file_name 已含 data/ 前缀
  - 类别 ID：person=1, car=3（旧版 person=2, car=1）
  - iscrowd：boolean（旧版为 int）
  - 无 RGB-IR 配对能力（thermal/rgb video ID 不匹配），默认 det_only 模式

对应 docs/architecture.md §11 文件结构。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


# FLIR_ADAS_v2 Roboflow 版本类别 ID 映射
# 仅保留 person 与 car，与 text_prompt "person. car." 一致
# person: cat_id=1 → class_idx=0（prompt 左起第一个）
# car:    cat_id=3 → class_idx=1（prompt 左起第二个）
FLIR_V2_CATEGORY_TO_CLASS_IDX: Dict[int, int] = {1: 0, 3: 1}
FLIR_V2_VALID_CAT_IDS: frozenset = frozenset({1, 3})
FLIR_V2_TEXT_PROMPT: str = "person. car."


def build_flir_v2_category_map(
    text_prompt: str,
) -> Tuple[Dict[int, int], frozenset]:
    """
    校验 text_prompt 并返回 (cat_id→class_idx 映射, 有效 cat_id 集合)。

    当前仅支持 "person. car." prompt，对应 FLIR_ADAS_v2 cat_id 1 和 3。

    Args:
        text_prompt: 与训练时相同的文本，如 "person. car."

    Returns:
        (category_map, valid_cat_ids)
    """
    normalized = text_prompt.strip().lower().rstrip(".")
    segments = [s.strip() for s in normalized.split(".") if s.strip()]
    if segments != ["person", "car"]:
        raise ValueError(
            f"FLIR_ADAS_v2 当前仅支持 prompt 'person. car.'，"
            f"当前解析为: {segments}"
        )
    return dict(FLIR_V2_CATEGORY_TO_CLASS_IDX), frozenset(FLIR_V2_VALID_CAT_IDS)


class FlirADASV2Dataset(Dataset):
    """
    FLIR_ADAS_v2 热红外数据集（Roboflow COCO 格式）。

    目录结构（以 thermal_train 为例）：
        {root}/coco.json          — 标注文件（含 images / annotations / categories）
        {root}/data/*.jpg         — 热红外图像（file_name 含 data/ 前缀）

    类别映射（固定）：
        person (cat_id=1) → class_idx=0
        car    (cat_id=3) → class_idx=1

    不含 RGB 配对逻辑；训练时使用 loss_mode="det_only"。
    """

    def __init__(
        self,
        root: str,
        processor: Any,
        text_prompt: str,
        category_map: Optional[Dict[int, int]] = None,
        valid_cat_ids: Optional[frozenset] = None,
    ) -> None:
        """
        Args:
            root:          FLIR_ADAS_v2 split 目录（如 FLIR_ADAS_v2/images_thermal_train）。
            processor:     AutoProcessor（GroundingDinoImageProcessor）。
            text_prompt:   检测 prompt，如 "person. car."
            category_map:  cat_id → class_idx 映射；None 时使用默认 {1:0, 3:1}。
            valid_cat_ids: 有效 cat_id 集合；None 时使用 {1, 3}。
        """
        super().__init__()
        self._root = os.path.abspath(root)
        self._processor = processor
        self._text_prompt = text_prompt
        self._cat_map: Dict[int, int] = (
            category_map if category_map is not None else dict(FLIR_V2_CATEGORY_TO_CLASS_IDX)
        )
        self._valid_ids: frozenset = (
            valid_cat_ids if valid_cat_ids is not None else frozenset(FLIR_V2_VALID_CAT_IDS)
        )

        ann_path = os.path.join(self._root, "coco.json")
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"未找到标注文件: {ann_path}")

        with open(ann_path, encoding="utf-8") as f:
            coco = json.load(f)

        # 仅保留含有效类别标注的图像（过滤无人/车标注的帧）
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
            if int(a["category_id"]) not in self._valid_ids:
                continue
            iid = a["image_id"]
            self._id_to_anns.setdefault(iid, []).append(a)

        assert len(self._images) > 0, (
            f"过滤后无有效图像（root={self._root}），"
            f"请检查 coco.json 中 category_id 是否包含 {self._valid_ids}"
        )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_info = self._images[index]
        # file_name 已含 data/ 前缀，直接拼接 root
        img_path = os.path.join(self._root, img_info["file_name"])
        img_w = int(img_info["width"])
        img_h = int(img_info["height"])

        image = Image.open(img_path).convert("RGB")
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
                    "iscrowd": int(ann.get("iscrowd", 0)),  # bool → int
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
        pixel_mask = img_enc["pixel_mask"][0]
        labels_dict = img_enc["labels"][0]
        labels = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in labels_dict.items()
        }

        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": labels,
            "image_path": img_path,
            "orig_size": torch.tensor([img_h, img_w], dtype=torch.int64),
        }


def collate_flir_v2(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    FlirADASV2Dataset 的 collate 函数（与 collate_fn 逻辑相同，独立命名避免混淆）。

    labels 保持为 List[Dict]，pixel_values / pixel_mask stack 为 Tensor。
    """
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    pixel_mask = torch.stack([b["pixel_mask"] for b in batch], dim=0)

    labels: List[Dict[str, Any]] = []
    for b in batch:
        entry: Dict[str, Any] = {}
        for k, v in b["labels"].items():
            entry[k] = v.clone() if isinstance(v, torch.Tensor) else v
        labels.append(entry)

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "labels": labels,
        "image_paths": [b["image_path"] for b in batch],
        "orig_sizes": torch.stack([b["orig_size"] for b in batch], dim=0),
    }
