"""
COCO 格式小样本数据集：`train/_annotations.coco.json` + `train/*.jpg`。

仅保留 category_id ∈ {1: car, 2: person}，忽略 Roboflow dummy 父类 id=0。

通过 `GroundingDinoImageProcessor` 传入 COCO annotations，使 bbox 与 resize/pad 后的 `pixel_values` 对齐；
`class_labels` 为 prompt 中从左到右的类别序号 0（person）、1（car）。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset

# COCO: 1=car, 2=person；与 prompt ``person. car.`` 中从左到右顺序：0=person，1=car
COCO_CATEGORY_TO_CLASS_IDX: Dict[int, int] = {2: 0, 1: 1}


def build_coco_category_to_class_index(text_prompt: str) -> Dict[int, int]:
    """
    校验 prompt 与固定映射一致（MVP 仅 person / car）。

    Args:
        text_prompt: 与训练时完全相同的文本，如 ``person. car.``
    """
    normalized = text_prompt.strip().lower().rstrip(".")
    segments = [s.strip() for s in normalized.split(".") if s.strip()]
    if segments != ["person", "car"]:
        raise ValueError(
            f"MVP 要求 prompt 为 'person. car.' 形式（两类顺序：先 person 后 car），当前解析为: {segments}"
        )
    return dict(COCO_CATEGORY_TO_CLASS_IDX)


class FlirCocoOverfitDataset(Dataset):
    """
    使用 ``processor.image_processor(images=, annotations=)`` 生成与模型输入对齐的 ``labels``。
    """

    def __init__(
        self,
        root: str,
        processor: Any,
        text_prompt: str,
        coco_category_id_to_class_idx: Dict[int, int],
    ) -> None:
        super().__init__()
        self._root = os.path.abspath(root)
        self._processor = processor
        self._text_prompt = text_prompt
        self._cat_map = coco_category_id_to_class_idx

        ann_path = os.path.join(self._root, "_annotations.coco.json")
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"未找到标注文件: {ann_path}")

        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self._images: List[Dict[str, Any]] = sorted(coco["images"], key=lambda x: x["id"])
        id_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for a in coco["annotations"]:
            iid = a["image_id"]
            id_to_anns.setdefault(iid, []).append(a)

        self._id_to_anns = id_to_anns
        self._valid_cat_ids = {1, 2}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_info = self._images[index]
        file_name = img_info["file_name"]
        img_w = int(img_info["width"])
        img_h = int(img_info["height"])
        img_path = os.path.join(self._root, file_name)

        image = Image.open(img_path).convert("RGB")
        anns = self._id_to_anns.get(img_info["id"], [])

        coco_anns: List[Dict[str, Any]] = []
        for ann in anns:
            cid = int(ann["category_id"])
            if cid not in self._valid_cat_ids:
                continue
            class_idx = self._cat_map[cid]
            coco_anns.append(
                {
                    "category_id": class_idx,
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


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """将 batch 堆叠；``labels`` 为 ``list[dict]`` 供 HF 模型使用。"""
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    pixel_mask = torch.stack([b["pixel_mask"] for b in batch], dim=0)
    labels: List[Dict[str, Any]] = []
    for b in batch:
        lab = b["labels"]
        entry: Dict[str, Any] = {}
        for k, v in lab.items():
            if isinstance(v, torch.Tensor):
                entry[k] = v.clone()
            else:
                entry[k] = v
        labels.append(entry)

    image_paths = [b["image_path"] for b in batch]
    orig_sizes = torch.stack([b["orig_size"] for b in batch], dim=0)
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "labels": labels,
        "image_paths": image_paths,
        "orig_sizes": orig_sizes,
    }
