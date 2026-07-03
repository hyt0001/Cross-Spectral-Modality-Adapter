"""M3FD 跨数据集评估共用数据加载与 GT 构建。"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from src.config import CSMAConfig

M3FD_IR_DIR = "/root/autodl-tmp/M3FD/ir"
M3FD_GT_JSON = "/root/autodl-tmp/M3FD/m3fd_test_person_car.json"


def pad_for_csma(
    pixel_values: torch.Tensor,
    pixel_mask: torch.Tensor,
    multiple: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad processor output so CSMA skip connections align (W/H multiple of 8)."""
    _, _, h, w = pixel_values.shape
    nh = ((h + multiple - 1) // multiple) * multiple
    nw = ((w + multiple - 1) // multiple) * multiple
    if nh == h and nw == w:
        return pixel_values, pixel_mask

    pv_pad = torch.zeros(
        pixel_values.shape[0], pixel_values.shape[1], nh, nw,
        dtype=pixel_values.dtype, device=pixel_values.device,
    )
    pv_pad[:, :, :h, :w] = pixel_values
    pm_pad = torch.zeros(nh, nw, dtype=pixel_mask.dtype, device=pixel_mask.device)
    pm_pad[:h, :w] = pixel_mask
    return pv_pad, pm_pad


class M3FDIRDataset(Dataset):
    """M3FD 红外 test 集（person+car GT）。"""

    def __init__(
        self,
        ir_dir: str,
        gt_json: str,
        gdino_processor: Any,
        cfg: CSMAConfig,
    ) -> None:
        self._ir_dir = ir_dir
        self._cfg = cfg
        self._gdino_proc = gdino_processor

        with open(gt_json, encoding="utf-8") as f:
            coco_data = json.load(f)

        ir_files = set(os.listdir(ir_dir))
        self._images = [
            img for img in coco_data["images"]
            if img["file_name"] in ir_files
        ]
        self._images.sort(key=lambda x: x["id"])

        img_ids = {img["id"] for img in self._images}
        self._annotations = [
            ann for ann in coco_data["annotations"]
            if ann["image_id"] in img_ids
        ]

        print(f"[M3FDIRDataset] images={len(self._images)}  annotations={len(self._annotations)}")

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self._images[idx]
        ir_path = os.path.join(self._ir_dir, info["file_name"])
        pil_ir = Image.open(ir_path).convert("RGB")
        orig_w, orig_h = pil_ir.size

        enc = self._gdino_proc(images=pil_ir, return_tensors="pt")
        pixel_values = enc["pixel_values"][0]
        pixel_mask = enc["pixel_mask"][0]

        return {
            "image_id": info["id"],
            "file_name": info["file_name"],
            "orig_w": orig_w,
            "orig_h": orig_h,
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
        }


def _collate_m3fd(batch: List[Dict]) -> Dict[str, Any]:
    pixel_values_list = [b["pixel_values"] for b in batch]
    pixel_masks_list = [b["pixel_mask"] for b in batch]
    max_h = max(pv.shape[1] for pv in pixel_values_list)
    max_w = max(pv.shape[2] for pv in pixel_values_list)

    padded_pv: List[torch.Tensor] = []
    padded_pm: List[torch.Tensor] = []
    for pv, pm in zip(pixel_values_list, pixel_masks_list):
        _, h, w = pv.shape
        pv_pad = torch.zeros(pv.shape[0], max_h, max_w, dtype=pv.dtype)
        pv_pad[:, :h, :w] = pv
        pm_pad = torch.zeros(max_h, max_w, dtype=pm.dtype)
        pm_pad[:h, :w] = pm
        padded_pv.append(pv_pad)
        padded_pm.append(pm_pad)

    return {
        "image_ids": [b["image_id"] for b in batch],
        "file_names": [b["file_name"] for b in batch],
        "orig_ws": [b["orig_w"] for b in batch],
        "orig_hs": [b["orig_h"] for b in batch],
        "pixel_values": torch.stack(padded_pv),
        "pixel_masks": torch.stack(padded_pm),
    }


def _build_m3fd_gt(dataset: M3FDIRDataset) -> COCO:
    coco_dict: Dict[str, Any] = {
        "images": dataset._images,
        "annotations": dataset._annotations,
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "car"},
        ],
    }
    coco = COCO()
    coco.dataset = coco_dict
    coco.createIndex()
    return coco
