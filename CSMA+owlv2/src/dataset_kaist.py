from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

KAIST_EVAL_CATEGORIES = [{"id": 1, "name": "person"}]
KAIST_LABEL_TO_EVAL_CAT = {"person": 1}
KAIST_ANN_CAT_TO_EVAL_CAT = {1: 1}
KAIST_VALID_CAT_IDS = frozenset({1})


class KAISTPairedDataset(Dataset):
    """KAIST multispectral pedestrian dataset for evaluation.

    The official imageSets text files contain entries like
    ``set06/V000/I00019``. Images live under ``images/<entry>/{lwir,visible}``
    and annotations are VOC-like XML files with ``x/y/w/h`` boxes.
    """

    def __init__(
        self,
        root: str,
        processor: Any,
        split: str = "test-all-20",
        text_labels: Optional[List[str]] = None,
        require_visible: bool = False,
        include_empty: bool = True,
        annotation_dir: str = "annotations-xml-new-sanitized",
    ) -> None:
        super().__init__()
        self._root = os.path.abspath(root)
        self._processor = processor
        self._split = split
        self._text_labels = text_labels or ["person"]
        self._ann_root = os.path.join(self._root, annotation_dir)

        split_path = split
        if not os.path.isabs(split_path):
            if split_path.endswith(".txt"):
                split_path = os.path.join(self._root, "imageSets", split_path)
            else:
                split_path = os.path.join(self._root, "imageSets", f"{split_path}.txt")
        if not os.path.isfile(split_path):
            raise FileNotFoundError(split_path)
        if not os.path.isdir(self._ann_root):
            raise FileNotFoundError(self._ann_root)

        entries: List[str] = []
        with open(split_path, encoding="utf-8") as f:
            for line in f:
                item = line.strip()
                if item:
                    entries.append(item[:-4] if item.lower().endswith(".jpg") else item)

        samples: List[Dict[str, Any]] = []
        dropped_missing = 0
        dropped_empty = 0
        for entry in entries:
            lwir_path = os.path.join(self._root, "images", entry, "lwir")
            visible_path = os.path.join(self._root, "images", entry, "visible")
            # imageSets entries point to frame stems; actual files are in modality dirs.
            if not os.path.isfile(lwir_path + ".jpg"):
                lwir_path = os.path.join(self._root, "images", os.path.dirname(entry), "lwir", os.path.basename(entry))
            if not os.path.isfile(visible_path + ".jpg"):
                visible_path = os.path.join(self._root, "images", os.path.dirname(entry), "visible", os.path.basename(entry))

            lwir_file = lwir_path + ".jpg"
            visible_file = visible_path + ".jpg"
            ann_file = os.path.join(self._ann_root, entry + ".xml")
            if not os.path.isfile(lwir_file) or not os.path.isfile(ann_file):
                dropped_missing += 1
                continue
            if require_visible and not os.path.isfile(visible_file):
                dropped_missing += 1
                continue

            width, height, anns = self._parse_annotation(ann_file)
            if not anns and not include_empty:
                dropped_empty += 1
                continue
            samples.append({
                "id": len(samples) + 1,
                "entry": entry,
                "file_name": os.path.relpath(lwir_file, self._root),
                "visible_file_name": os.path.relpath(visible_file, self._root)
                if os.path.isfile(visible_file) else None,
                "width": width,
                "height": height,
                "anns": anns,
            })

        self._images = samples
        self._id_to_anns: Dict[int, List[Dict[str, Any]]] = {
            int(img["id"]): img["anns"] for img in self._images
        }
        print(
            f"[KAISTPairedDataset] root={self._root} split={split} images={len(self._images)} "
            f"dropped_missing={dropped_missing} dropped_empty={dropped_empty}"
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
            if bnd.find("xmin") is not None:
                xmin = float(bnd.findtext("xmin", "0"))
                ymin = float(bnd.findtext("ymin", "0"))
                xmax = float(bnd.findtext("xmax", "0"))
                ymax = float(bnd.findtext("ymax", "0"))
                bw, bh = xmax - xmin, ymax - ymin
            else:
                xmin = float(bnd.findtext("x", "0"))
                ymin = float(bnd.findtext("y", "0"))
                bw = float(bnd.findtext("w", "0"))
                bh = float(bnd.findtext("h", "0"))
                xmax = xmin + bw
                ymax = ymin + bh
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
