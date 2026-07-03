from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

FLIR_V1_CAT_TO_CLASS_IDX = {1: 0, 3: 1}
FLIR_V1_VALID_CAT_IDS = frozenset({1, 3})


def build_flir_v1_category_map(text_labels: List[str]) -> Tuple[Dict[int, int], frozenset]:
    if [t.strip().lower() for t in text_labels] != ["person", "car"]:
        raise ValueError("FLIR v1 only supports ['person', 'car']")
    return dict(FLIR_V1_CAT_TO_CLASS_IDX), FLIR_V1_VALID_CAT_IDS


class FlirV1PairedDataset(Dataset):
    def __init__(
        self, root: str, processor: Any, text_labels: Optional[List[str]] = None,
        category_map: Optional[Dict[int, int]] = None, valid_cat_ids: Optional[frozenset] = None,
        require_rgb: bool = True,
    ) -> None:
        super().__init__()
        self._root = os.path.abspath(root)
        self._processor = processor
        self._text_labels = text_labels or ["person", "car"]
        self._cat_map = category_map or dict(FLIR_V1_CAT_TO_CLASS_IDX)
        self._valid_ids = valid_cat_ids or FLIR_V1_VALID_CAT_IDS

        with open(os.path.join(self._root, "thermal_annotations.json"), encoding="utf-8") as f:
            coco = json.load(f)
        valid_ids = {a["image_id"] for a in coco["annotations"] if int(a["category_id"]) in self._valid_ids}
        self._images = sorted([img for img in coco["images"] if img["id"] in valid_ids], key=lambda x: x["id"])
        self._id_to_anns: Dict[int, List[Dict]] = {}
        for ann in coco["annotations"]:
            cid = int(ann["category_id"])
            if cid in self._valid_ids:
                self._id_to_anns.setdefault(ann["image_id"], []).append(ann)

        rgb_dir = os.path.join(self._root, "RGB")
        self._rgb_stem_to_path = {}
        if os.path.isdir(rgb_dir):
            for fn in os.listdir(rgb_dir):
                self._rgb_stem_to_path[os.path.splitext(fn)[0]] = os.path.join(rgb_dir, fn)

        if require_rgb:
            before = len(self._images)
            self._images = [
                img for img in self._images
                if os.path.splitext(os.path.basename(img["file_name"]))[0] in self._rgb_stem_to_path
            ]
            dropped = before - len(self._images)
            if dropped:
                print(f"[FlirV1PairedDataset] require_rgb=True: 过滤掉 {dropped} 张无 RGB 配对的图")
        print(f"[FlirV1PairedDataset] root={self._root} images={len(self._images)}")

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_info = self._images[index]
        img_path = os.path.join(self._root, img_info["file_name"])
        stem = os.path.splitext(os.path.basename(img_info["file_name"]))[0]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        pixel_values = self._processor.image_processor(images=image, return_tensors="pt")["pixel_values"][0]

        boxes, labels = [], []
        for ann in self._id_to_anns.get(img_info["id"], []):
            x, y, w, h = [float(v) for v in ann["bbox"]]
            cx, cy, nw, nh = (x + w / 2) / orig_w, (y + h / 2) / orig_h, w / orig_w, h / orig_h
            if nw > 0 and nh > 0:
                boxes.append([min(max(cx, 0), 1), min(max(cy, 0), 1), min(max(nw, 0), 1), min(max(nh, 0), 1)])
                labels.append(self._cat_map[int(ann["category_id"])])

        gt_boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        gt_labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        rgb_pixel_values = None
        rgb_path = self._rgb_stem_to_path.get(stem)
        if rgb_path and os.path.isfile(rgb_path):
            rgb_img = Image.open(rgb_path).convert("RGB").resize((orig_w, orig_h), Image.BILINEAR)
            rgb_pixel_values = self._processor.image_processor(images=rgb_img, return_tensors="pt")["pixel_values"][0]

        return {
            "pixel_values": pixel_values, "gt_boxes": gt_boxes, "gt_labels": gt_labels,
            "rgb_pixel_values": rgb_pixel_values, "image_id": int(img_info["id"]),
            "image_path": img_path, "orig_size": (orig_h, orig_w),
        }


def collate_flir_v1(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "gt_boxes": [b["gt_boxes"] for b in batch],
        "gt_labels": [b["gt_labels"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],
        "orig_sizes": [b["orig_size"] for b in batch],
    }
    # 按样本级别收集 RGB：只把有 RGB 的样本堆叠，并记录其在 batch 内的索引
    rgb_indices, rgb_tensors = [], []
    for i, b in enumerate(batch):
        v = b.get("rgb_pixel_values")
        if v is not None:
            rgb_indices.append(i)
            rgb_tensors.append(v)
    if rgb_tensors:
        result["rgb_pixel_values"] = torch.stack(rgb_tensors)
        result["rgb_indices"] = rgb_indices  # batch 内哪些位置有 RGB
    return result
