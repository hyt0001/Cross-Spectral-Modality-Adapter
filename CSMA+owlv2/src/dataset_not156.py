from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

NOT156_EVAL_CATEGORIES_PERSON_CAR = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "car"},
]
NOT156_LABEL_TO_EVAL_CAT_PERSON_CAR = {"person": 1, "car": 2}
NOT156_ANN_CAT_TO_EVAL_CAT_PERSON_CAR = {1: 1, 2: 2}
NOT156_VALID_CAT_IDS_PERSON_CAR = frozenset({1, 2})

NOT156_EVAL_CATEGORIES_PERSON = [{"id": 1, "name": "person"}]
NOT156_LABEL_TO_EVAL_CAT_PERSON = {"person": 1}
NOT156_ANN_CAT_TO_EVAL_CAT_PERSON = {1: 1}
NOT156_VALID_CAT_IDS_PERSON = frozenset({1})

# Map sequence folder names to evaluation categories. NOT-156 is a tracking
# benchmark with one target object per sequence; we infer class from the name.
NOT156_PERSON_PREFIXES = (
    "person",
    "runner",
    "walker",
    "rider",
    "kid",
    "face",
    "head",
    "twoman",
    "fourman",
    "threeman",
    "hood",
    "headwithhood",
    "exercise",
    "jump",
    "stretch",
    "weightlifting",
    "enterroom",
    "takeoffclothes",
    "drinkwater",
    "throwtrash",
    "walkparallel",
    "walkers",
    "onstep",
    "ongrass",
    "treadmill",
    "shot",
    "library",
    "corridor",
    "courtyard",
    "undertree",
    "soccer",
    "ebike",
)
NOT156_CAR_PREFIXES = ("car", "truck", "toycar")


def infer_not156_sequence_category(seq_name: str) -> Optional[str]:
    low = seq_name.strip().lower()
    for prefix in NOT156_CAR_PREFIXES:
        if low.startswith(prefix):
            return "car"
    for prefix in NOT156_PERSON_PREFIXES:
        if low.startswith(prefix):
            return "person"
    return None


def build_not156_category_map(
    text_labels: List[str],
) -> Tuple[Dict[int, int], frozenset, List[Dict[str, Any]], Dict[int, int], Dict[str, int]]:
    labels = [t.strip().lower() for t in text_labels]
    if labels == ["person", "car"]:
        return (
            {1: 0, 2: 1},
            NOT156_VALID_CAT_IDS_PERSON_CAR,
            NOT156_EVAL_CATEGORIES_PERSON_CAR,
            NOT156_ANN_CAT_TO_EVAL_CAT_PERSON_CAR,
            NOT156_LABEL_TO_EVAL_CAT_PERSON_CAR,
        )
    if labels == ["person"]:
        return (
            {1: 0},
            NOT156_VALID_CAT_IDS_PERSON,
            NOT156_EVAL_CATEGORIES_PERSON,
            NOT156_ANN_CAT_TO_EVAL_CAT_PERSON,
            NOT156_LABEL_TO_EVAL_CAT_PERSON,
        )
    raise ValueError("NOT-156 supports text_labels=['person', 'car'] or ['person']")


def _parse_bbox_line(line: str) -> Optional[List[float]]:
    line = line.strip()
    if not line or line[0].isalpha():
        return None
    parts = re.split(r"[\s,]+", line)
    if len(parts) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in parts[:4]]
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


class NOT156PairedDataset(Dataset):
    """NOT-156 IR detection dataset for external evaluation.

    NOT-156 is originally a night tracking benchmark. Each sequence tracks one
    target object; we convert frame-level tracking boxes into detection GT.

    Expected layout (partial release also works):
      {root}/{seq_subdir}/{sequence}/channel2/*.jpg   # thermal IR frames
      {root}/{seq_subdir}/{sequence}/groundtruth_rect.txt
    """

    def __init__(
        self,
        root: str,
        processor: Any,
        text_labels: Optional[List[str]] = None,
        seq_subdir: str = "NOT156_train/NOT156_train",
        ir_subdir: str = "channel2",
        include_empty: bool = False,
    ) -> None:
        super().__init__()
        self._root = os.path.abspath(root)
        self._processor = processor
        self._text_labels = text_labels or ["person", "car"]
        self._seq_root = os.path.join(self._root, seq_subdir)
        self._ir_subdir = ir_subdir

        if not os.path.isdir(self._seq_root):
            raise FileNotFoundError(self._seq_root)

        (
            self._cat_map,
            self._valid_cat_ids,
            self.eval_categories,
            self.ann_cat_to_eval_cat,
            self.label_to_eval_cat,
        ) = build_not156_category_map(self._text_labels)

        allowed_names = {c["name"] for c in self.eval_categories}
        self._allowed_seq_cats = allowed_names

        self._images: List[Dict[str, Any]] = []
        self._id_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        dropped_seq = 0
        dropped_empty = 0
        dropped_unmapped = 0
        img_id = 1

        for seq_name in sorted(os.listdir(self._seq_root)):
            seq_dir = os.path.join(self._seq_root, seq_name)
            if not os.path.isdir(seq_dir):
                continue

            seq_cat = infer_not156_sequence_category(seq_name)
            if seq_cat not in self._allowed_seq_cats:
                dropped_unmapped += 1
                continue

            gt_path = os.path.join(seq_dir, "groundtruth_rect.txt")
            ir_dir = os.path.join(seq_dir, ir_subdir)
            if not os.path.isfile(gt_path) or not os.path.isdir(ir_dir):
                dropped_seq += 1
                continue

            with open(gt_path, encoding="utf-8") as f:
                gt_lines = [ln for ln in f if ln.strip()]

            frame_files = sorted(
                name
                for name in os.listdir(ir_dir)
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            )
            if not frame_files:
                dropped_seq += 1
                continue

            n_frames = min(len(gt_lines), len(frame_files))
            ann_cat_id = 1 if seq_cat == "person" else 2

            for frame_idx in range(n_frames):
                bbox = _parse_bbox_line(gt_lines[frame_idx])
                if bbox is None:
                    if include_empty:
                        bbox = None
                    else:
                        dropped_empty += 1
                        continue

                rel_path = os.path.join(seq_subdir, seq_name, ir_subdir, frame_files[frame_idx])
                abs_path = os.path.join(self._root, rel_path)
                if not os.path.isfile(abs_path):
                    dropped_seq += 1
                    continue

                anns: List[Dict[str, Any]] = []
                if bbox is not None:
                    x, y, w, h = bbox
                    anns.append({
                        "category_id": ann_cat_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    })

                if not include_empty and not anns:
                    dropped_empty += 1
                    continue

                self._images.append({
                    "id": img_id,
                    "file_name": rel_path,
                    "width": 0,
                    "height": 0,
                    "sequence": seq_name,
                    "frame_index": frame_idx + 1,
                })
                self._id_to_anns[img_id] = anns
                img_id += 1

        print(
            f"[NOT156PairedDataset] root={self._root} seq_root={self._seq_root} "
            f"images={len(self._images)} text_labels={self._text_labels} "
            f"dropped_unmapped_seq={dropped_unmapped} dropped_incomplete_seq={dropped_seq} "
            f"dropped_empty={dropped_empty}"
        )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_info = self._images[index]
        img_path = os.path.join(self._root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        img_info["width"] = orig_w
        img_info["height"] = orig_h

        pixel_values = self._processor.image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"][0]

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
