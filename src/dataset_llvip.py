"""
LLVIP 配对数据集适配器（跨数据集泛化测试）。

数据集目录结构（解压后）：
    {root}/infrared/train|test/*.jpg   — 红外图像
    {root}/visible/train|test/*.jpg    — 对应可见光图像（同名配对）
    {root}/annotations/val.json        — COCO 格式测试集标注（Zenodo CAFF-DINO）
    {root}/annotations/train.json      — COCO 格式训练集标注（可选）

与 FLIR v1 的关键差异：
  - 仅 person 类别（category_id=0，COCO 转换后）
  - file_name 为纯文件名（如 230286.jpg），无子目录前缀
  - 红外/可见光分属 infrared/ 与 visible/ 子目录，按 split 组织

对应 docs/TD.md §3.1 跨数据集泛化测试。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

# LLVIP COCO 标注中 person 的 category_id（voc2coco 从 0 起编）
LLVIP_CATEGORY_TO_CLASS_IDX: Dict[int, int] = {0: 0}
LLVIP_VALID_CAT_IDS: frozenset = frozenset({0})
LLVIP_TEXT_PROMPT: str = "person."


def build_llvip_category_map(
    text_prompt: str,
) -> Tuple[Dict[int, int], frozenset]:
    """
    校验 text_prompt 并返回 (cat_id→class_idx 映射, 有效 cat_id 集合)。

    Args:
        text_prompt: 检测 prompt，如 "person." 或 "person. car."（仅 person 有 GT）

    Returns:
        (category_map, valid_cat_ids)
    """
    normalized = text_prompt.strip().lower().rstrip(".")
    segments = [s.strip() for s in normalized.split(".") if s.strip()]
    if "person" not in segments:
        raise ValueError(
            f"LLVIP 至少需要 prompt 含 'person'，当前解析为: {segments}"
        )
    return dict(LLVIP_CATEGORY_TO_CLASS_IDX), frozenset(LLVIP_VALID_CAT_IDS)


class LLVIPPairedDataset(Dataset):
    """
    LLVIP RGB-IR 配对数据集（测试/验证 split）。

    每个样本返回：
        pixel_values      [3, H, W]  红外图像（ImageNet 归一化）
        pixel_mask        [H, W]     有效像素掩码
        labels            Dict       DINO 格式目标框（cxcywh 归一化）
        rgb_pixel_values  [3, H, W]  对应可见光图像；缺失时为 None
        image_path        str        红外图像绝对路径
        rgb_path          str | None 可见光图像绝对路径

    RGB 配对策略：
        IR  : {root}/infrared/{split}/{stem}.jpg
        RGB : {root}/visible/{split}/{stem}.jpg
    """

    def __init__(
        self,
        root: str,
        processor: Any,
        text_prompt: str,
        split: str = "test",
        ann_file: Optional[str] = None,
        category_map: Optional[Dict[int, int]] = None,
        valid_cat_ids: Optional[frozenset] = None,
    ) -> None:
        """
        Args:
            root:          LLVIP 根目录（含 infrared/、visible/、annotations/）。
            processor:     AutoProcessor（GroundingDinoImageProcessor）。
            text_prompt:   检测 prompt，如 "person."。
            split:         数据划分，"test" 或 "train"（对应 infrared/{split}/）。
            ann_file:      COCO 标注 JSON 路径；None 时默认 annotations/val.json（test）
                           或 annotations/train.json（train）。
            category_map:  cat_id → class_idx；None 时使用 {0: 0}。
            valid_cat_ids: 有效 cat_id 集合；None 时使用 {0}。
        """
        super().__init__()
        self._root = os.path.abspath(root)
        self._split = split
        self._processor = processor
        self._text_prompt = text_prompt
        self._cat_map: Dict[int, int] = (
            category_map if category_map is not None
            else dict(LLVIP_CATEGORY_TO_CLASS_IDX)
        )
        self._valid_ids: frozenset = (
            valid_cat_ids if valid_cat_ids is not None
            else frozenset(LLVIP_VALID_CAT_IDS)
        )

        if ann_file is None:
            ann_name = "val.json" if split == "test" else "train.json"
            ann_file = os.path.join(self._root, "annotations", ann_name)
        ann_path = os.path.abspath(ann_file)
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"未找到标注文件: {ann_path}")

        with open(ann_path, encoding="utf-8") as f:
            coco = json.load(f)

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
            f"过滤后无有效图像（root={self._root}, split={split}），"
            f"请检查标注中 category_id 是否包含 {self._valid_ids}"
        )

        self._ir_dir = os.path.join(self._root, "infrared", split)
        self._rgb_dir = os.path.join(self._root, "visible", split)

        assert os.path.isdir(self._ir_dir), (
            f"红外图像目录不存在: {self._ir_dir}\n"
            f"  请确认 root 参数指向含 infrared/{split}/ 子目录的 LLVIP 解压根目录"
        )

        # 过滤掉磁盘上实际缺失的 IR 文件，并统计 RGB 配对数
        filtered: List[Dict[str, Any]] = []
        paired = 0
        skipped = 0
        for img in self._images:
            stem = os.path.splitext(os.path.basename(img["file_name"]))[0]
            ir_p = os.path.join(self._ir_dir, f"{stem}.jpg")
            if not os.path.isfile(ir_p):
                skipped += 1
                continue
            filtered.append(img)
            rgb_p = os.path.join(self._rgb_dir, f"{stem}.jpg")
            if os.path.isfile(rgb_p):
                paired += 1

        if skipped > 0:
            print(
                f"[LLVIPPairedDataset] 警告: {skipped} 张图像因 IR 文件缺失被跳过"
            )

        self._images = filtered

        assert len(self._images) > 0, (
            f"过滤后无有效图像（root={self._root}, split={split}），"
            f"请检查 {self._ir_dir} 目录是否包含图像文件"
        )

        print(
            f"[LLVIPPairedDataset] root={self._root} split={split}  "
            f"有效图像={len(self._images)}  RGB配对={paired}"
        )

    def __len__(self) -> int:
        return len(self._images)

    def _resolve_paths(self, file_name: str) -> Tuple[str, Optional[str]]:
        """根据 COCO file_name 解析红外与可见光绝对路径。"""
        stem = os.path.splitext(os.path.basename(file_name))[0]
        ir_path = os.path.join(self._ir_dir, f"{stem}.jpg")
        rgb_path = os.path.join(self._rgb_dir, f"{stem}.jpg")
        if not os.path.isfile(ir_path):
            raise FileNotFoundError(f"红外图像不存在: {ir_path}")
        if not os.path.isfile(rgb_path):
            rgb_path = None
        return ir_path, rgb_path

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_info = self._images[index]
        img_path, rgb_path = self._resolve_paths(img_info["file_name"])

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

        rgb_pixel_values: Optional[torch.Tensor] = None
        if rgb_path is not None and os.path.isfile(rgb_path):
            rgb_img = Image.open(rgb_path).convert("RGB")
            ir_w, ir_h = int(img_info["width"]), int(img_info["height"])
            if rgb_img.size != (ir_w, ir_h):
                rgb_img = rgb_img.resize((ir_w, ir_h), Image.BILINEAR)
            rgb_enc = self._processor.image_processor(
                images=rgb_img,
                return_tensors="pt",
            )
            rgb_pixel_values = rgb_enc["pixel_values"][0]

        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": labels,
            "rgb_pixel_values": rgb_pixel_values,
            "image_path": img_path,
            "rgb_path": rgb_path,
        }


def collate_llvip(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    LLVIP 配对数据集 collate 函数（与 collate_flir_v1 接口一致）。

    Args:
        batch: FlirV1PairedDataset / LLVIPPairedDataset 的 __getitem__ 输出列表

    Returns:
        collated dict
    """
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_mask = torch.stack([b["pixel_mask"] for b in batch])
    labels = [b["labels"] for b in batch]

    result: Dict[str, Any] = {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "labels": labels,
        "image_paths": [b["image_path"] for b in batch],
        "rgb_paths": [b.get("rgb_path") for b in batch],
    }

    rgb_list: List[Optional[torch.Tensor]] = [b.get("rgb_pixel_values") for b in batch]
    if all(v is not None for v in rgb_list):
        result["rgb_pixel_values"] = torch.stack(rgb_list)  # type: ignore[arg-type]

    return result
