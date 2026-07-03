from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

# M3FD raw COCO category_id -> name (from HF release)
M3FD_RAW_CAT_ID_TO_NAME = {
    1: "bus",
    2: "car",
    3: "lamp",
    4: "motorcycle",
    5: "people",
    6: "truck",
}

M3FD_EVAL_CATEGORIES_PERSON_CAR = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "car"},
]
M3FD_LABEL_TO_EVAL_CAT_PERSON_CAR = {"person": 1, "car": 2}
M3FD_ANN_CAT_TO_EVAL_CAT_PERSON_CAR = {5: 1, 2: 2}
M3FD_VALID_CAT_IDS_PERSON_CAR = frozenset({2, 5})

M3FD_EVAL_CATEGORIES_ALL = [
    {"id": 1, "name": "bus"},
    {"id": 2, "name": "car"},
    {"id": 3, "name": "lamp"},
    {"id": 4, "name": "motorcycle"},
    {"id": 5, "name": "person"},
    {"id": 6, "name": "truck"},
]
M3FD_LABEL_TO_EVAL_CAT_ALL = {c["name"]: c["id"] for c in M3FD_EVAL_CATEGORIES_ALL}
M3FD_ANN_CAT_TO_EVAL_CAT_ALL = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
M3FD_VALID_CAT_IDS_ALL = frozenset({1, 2, 3, 4, 5, 6})


def build_m3fd_category_map(text_labels: List[str]) -> Tuple[Dict[int, int], frozenset, List[Dict[str, Any]], Dict[int, int], Dict[str, int]]:
    labels = [t.strip().lower() for t in text_labels]
    if labels == ["person", "car"]:
        raw_to_class = {5: 0, 2: 1}
        return (
            raw_to_class,
            M3FD_VALID_CAT_IDS_PERSON_CAR,
            M3FD_EVAL_CATEGORIES_PERSON_CAR,
            M3FD_ANN_CAT_TO_EVAL_CAT_PERSON_CAR,
            M3FD_LABEL_TO_EVAL_CAT_PERSON_CAR,
        )
    if labels == ["person"]:
        raw_to_class = {5: 0}
        return (
            raw_to_class,
            frozenset({5}),
            [{"id": 1, "name": "person"}],
            {5: 1},
            {"person": 1},
        )
    if labels == ["bus", "car", "lamp", "motorcycle", "person", "truck"]:
        raw_to_class = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
        return (
            raw_to_class,
            M3FD_VALID_CAT_IDS_ALL,
            M3FD_EVAL_CATEGORIES_ALL,
            M3FD_ANN_CAT_TO_EVAL_CAT_ALL,
            M3FD_LABEL_TO_EVAL_CAT_ALL,
        )
    raise ValueError(
        "M3FD supports text_labels=['person','car'], ['person'], or "
        "['bus','car','lamp','motorcycle','person','truck']"
    )


class M3FDPairedDataset(Dataset):
    """M3FD IR detection dataset for external evaluation.

    Expects:
      {root}/ir/*.png
      {root}/annotations/instances_default.json  (or --ann-file override)
    """

    def __init__(
        self,
        root: str,
        processor: Any,
        text_labels: Optional[List[str]] = None,
        ann_file: str = "annotations/instances_default.json",
        ir_subdir: str = "ir",
        include_empty: bool = True,
    ) -> None:
        super().__init__()
        self._root = os.path.abspath(root)
        self._processor = processor
        self._text_labels = text_labels or ["person", "car"]
        self._ir_dir = os.path.join(self._root, ir_subdir)

        ann_path = ann_file if os.path.isabs(ann_file) else os.path.join(self._root, ann_file)
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(ann_path)
        if not os.path.isdir(self._ir_dir):
            raise FileNotFoundError(self._ir_dir)

        (
            self._cat_map,
            self._valid_raw_cat_ids,
            self.eval_categories,
            self.ann_cat_to_eval_cat,
            self.label_to_eval_cat,
        ) = build_m3fd_category_map(self._text_labels)

        with open(ann_path, encoding="utf-8") as f:
            coco = json.load(f)

        id_to_img = {int(img["id"]): img for img in coco["images"]}
        self._images: List[Dict[str, Any]] = []
        self._id_to_anns: Dict[int, List[Dict[str, Any]]] = {}

        for img in sorted(coco["images"], key=lambda x: int(x["id"])):
            img_id = int(img["id"])
            file_name = img["file_name"]
            rel_path = file_name if file_name.startswith(ir_subdir + os.sep) else os.path.join(ir_subdir, file_name)
            abs_path = os.path.join(self._root, rel_path)
            if not os.path.isfile(abs_path):
                continue
            anns = []
            for ann in coco.get("annotations", []):
                if int(ann["image_id"]) != img_id:
                    continue
                raw_cid = int(ann["category_id"])
                if raw_cid not in self._valid_raw_cat_ids:
                    continue
                x, y, w, h = [float(v) for v in ann["bbox"]]
                if w <= 0 or h <= 0:
                    continue
                anns.append({
                    "category_id": raw_cid,
                    "bbox": [x, y, w, h],
                    "area": float(ann.get("area", w * h)),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                })
            if not include_empty and not anns:
                continue
            self._images.append({
                "id": img_id,
                "file_name": rel_path,
                "width": int(img.get("width", 0) or 0),
                "height": int(img.get("height", 0) or 0),
            })
            self._id_to_anns[img_id] = anns

        print(
            f"[M3FDPairedDataset] root={self._root} images={len(self._images)} "
            f"text_labels={self._text_labels} ann={ann_path}"
        )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_info = self._images[index]
        img_path = os.path.join(self._root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        if not img_info["width"] or not img_info["height"]:
            img_info["width"], img_info["height"] = orig_w, orig_h

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
                labels.append(self._cat_map[int(ann["category_id"])])

        gt_boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        gt_labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "rgb_pixel_values": None,
            "image_id": int(img_info["id"]),
            "image_path": img_path,
            "orig_size": (orig_h, orig_w),
        }
