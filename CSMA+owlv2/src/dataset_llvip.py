from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

LLVIP_EVAL_CATEGORIES = [{"id": 1, "name": "person"}]
LLVIP_LABEL_TO_EVAL_CAT = {"person": 1}
LLVIP_ANN_CAT_TO_EVAL_CAT = {1: 1}
LLVIP_VALID_CAT_IDS = frozenset({1})


class LLVIPPairedDataset(Dataset):
    """LLVIP paired IR/RGB dataset for evaluation.

    Evaluation uses infrared images as model input and VOC XML boxes as GT.
    Visible images are paired by filename and kept in samples for future use,
    but they are not required for inference-only validation.
    """

    def __init__(
        self,
        root: str,
        processor: Any,
        split: str = "test",
        text_labels: Optional[List[str]] = None,
        require_visible: bool = False,
    ) -> None:
        super().__init__()
        if split not in {"train", "test"}:
            raise ValueError("LLVIP split must be 'train' or 'test'")
        self._root = os.path.abspath(root)
        self._processor = processor
        self._split = split
        self._text_labels = text_labels or ["person"]

        self._ir_dir = os.path.join(self._root, "infrared", split)
        self._visible_dir = os.path.join(self._root, "visible", split)
        self._ann_dir = os.path.join(self._root, "Annotations")
        for path in [self._ir_dir, self._ann_dir]:
            if not os.path.isdir(path):
                raise FileNotFoundError(path)

        visible_stems = set()
        if os.path.isdir(self._visible_dir):
            visible_stems = {
                os.path.splitext(fn)[0]
                for fn in os.listdir(self._visible_dir)
                if fn.lower().endswith((".jpg", ".jpeg", ".png"))
            }

        samples: List[Dict[str, Any]] = []
        for fn in sorted(os.listdir(self._ir_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fn)[0]
            ann_path = os.path.join(self._ann_dir, f"{stem}.xml")
            if not os.path.isfile(ann_path):
                continue
            if require_visible and stem not in visible_stems:
                continue
            width, height, anns = self._parse_annotation(ann_path)
            if not anns:
                continue
            samples.append({
                "id": int(stem) if stem.isdigit() else len(samples) + 1,
                "file_name": os.path.join("infrared", split, fn),
                "visible_file_name": os.path.join("visible", split, fn)
                if stem in visible_stems else None,
                "width": width,
                "height": height,
                "anns": anns,
            })

        self._images = samples
        self._id_to_anns: Dict[int, List[Dict[str, Any]]] = {
            int(img["id"]): img["anns"] for img in self._images
        }
        print(
            f"[LLVIPPairedDataset] root={self._root} split={split} "
            f"images={len(self._images)}"
        )

    @staticmethod
    def _parse_annotation(path: str) -> tuple[int, int, List[Dict[str, Any]]]:
        root = ET.parse(path).getroot()
        size = root.find("size")
        if size is None:
            raise ValueError(f"Missing <size> in {path}")
        width = int(float(size.findtext("width", "0")))
        height = int(float(size.findtext("height", "0")))

        anns: List[Dict[str, Any]] = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            if name != "person":
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            xmin = float(bnd.findtext("xmin", "0"))
            ymin = float(bnd.findtext("ymin", "0"))
            xmax = float(bnd.findtext("xmax", "0"))
            ymax = float(bnd.findtext("ymax", "0"))
            xmin = max(0.0, min(xmin, float(width)))
            ymin = max(0.0, min(ymin, float(height)))
            xmax = max(0.0, min(xmax, float(width)))
            ymax = max(0.0, min(ymax, float(height)))
            bw, bh = xmax - xmin, ymax - ymin
            if bw <= 0 or bh <= 0:
                continue
            anns.append({
                "category_id": 1,
                "bbox": [xmin, ymin, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
        return width, height, anns

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_info = self._images[index]
        img_path = os.path.join(self._root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        pixel_values = self._processor.image_processor(images=image, return_tensors="pt")["pixel_values"][0]

        boxes, labels = [], []
        for ann in self._id_to_anns.get(int(img_info["id"]), []):
            x, y, w, h = [float(v) for v in ann["bbox"]]
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw, nh = w / orig_w, h / orig_h
            if nw > 0 and nh > 0:
                boxes.append([
                    min(max(cx, 0), 1),
                    min(max(cy, 0), 1),
                    min(max(nw, 0), 1),
                    min(max(nh, 0), 1),
                ])
                labels.append(0)

        gt_boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        gt_labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        visible_pixel_values = None
        visible_file_name = img_info.get("visible_file_name")
        if visible_file_name:
            visible_path = os.path.join(self._root, visible_file_name)
            if os.path.isfile(visible_path):
                visible = Image.open(visible_path).convert("RGB").resize((orig_w, orig_h), Image.BILINEAR)
                visible_pixel_values = self._processor.image_processor(
                    images=visible, return_tensors="pt")["pixel_values"][0]

        return {
            "pixel_values": pixel_values,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "rgb_pixel_values": visible_pixel_values,
            "image_id": int(img_info["id"]),
            "image_path": img_path,
            "orig_size": (orig_h, orig_w),
        }
