"""
LLVIP 可见光-红外配对数据集适配器（原生 Pascal VOC XML 标注）。

LLVIP 目录结构：
    {root}/Annotations/{stem}.xml     — Pascal VOC XML 标注（全量 15488，不分 split）
    {root}/infrared/train/{stem}.jpg  — 红外训练图像（12025）
    {root}/infrared/test/{stem}.jpg   — 红外测试图像（3463）
    {root}/visible/train/{stem}.jpg   — 可见光训练图像（同名，同 split）
    {root}/visible/test/{stem}.jpg    — 可见光测试图像

无需预先转换标注文件：__init__ 扫描 infrared/{split}/ 确定文件集合，
__getitem__ 实时解析对应 XML，直接构造 COCO target 传给 processor。

LLVIP 特性：
  - 仅有 person 一类（cat_id=1 → class_idx=0）
  - IR 与 visible 图像同名（{stem}.jpg），直接配对
  - 标注共享（同一 XML 同时对应 IR 和 visible）
  - bbox 格式：xmin/ymin/xmax/ymax（像素坐标）→ 转为 COCO [x,y,w,h]
  - 固定分辨率 1280×1024

COCO 类别约定：
    cat_id=1 → person → class_idx=0  （prompt "person." 或 "person. car." 均可）

对应 docs/TD.md §1.4 RGB-IR 配对逻辑。
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


# LLVIP 仅 person 类
LLVIP_CATEGORY_TO_CLASS_IDX: Dict[int, int] = {1: 0}
LLVIP_VALID_CAT_IDS: frozenset = frozenset({1})
LLVIP_CLASS_NAME_TO_CAT_ID: Dict[str, int] = {"person": 1}
LLVIP_TEXT_PROMPT: str = "person."


def build_llvip_category_map(
    text_prompt: str,
) -> Tuple[Dict[int, int], frozenset]:
    """
    校验 text_prompt 并返回 (cat_id→class_idx 映射, 有效 cat_id 集合)。

    LLVIP 要求 prompt 首词为 "person"；
    "person." 和 "person. car." 均合法（car 无 GT，不影响训练）。

    Args:
        text_prompt: 检测 prompt，如 "person." 或 "person. car."

    Returns:
        (category_map, valid_cat_ids)

    Raises:
        ValueError: 若 prompt 首词不是 "person"。
    """
    normalized = text_prompt.strip().lower().rstrip(".")
    segments = [s.strip() for s in normalized.split(".") if s.strip()]
    if not segments or segments[0] != "person":
        raise ValueError(
            f"LLVIP 适配器要求 prompt 首词为 'person'，当前解析为: {segments}"
        )
    return dict(LLVIP_CATEGORY_TO_CLASS_IDX), frozenset(LLVIP_VALID_CAT_IDS)


def _parse_voc_xml(
    xml_path: str,
    img_w: int,
    img_h: int,
) -> List[Dict[str, Any]]:
    """
    解析单个 Pascal VOC XML，返回 COCO 格式 annotation 列表。

    bbox 转换：[xmin,ymin,xmax,ymax] → [xmin,ymin,w,h]（像素，含边界裁剪）

    Args:
        xml_path: XML 文件路径。
        img_w:    图像宽度（用于边界裁剪）。
        img_h:    图像高度（用于边界裁剪）。

    Returns:
        List of {"category_id": int, "bbox": [x,y,w,h], "area": float, "iscrowd": int}
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []

    coco_anns: List[Dict[str, Any]] = []
    for obj in tree.getroot().findall("object"):
        cls_name = (obj.findtext("name") or "").strip().lower()
        cat_id = LLVIP_CLASS_NAME_TO_CAT_ID.get(cls_name)
        if cat_id is None:
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = max(0,    int(float(bndbox.findtext("xmin", "0"))))
        ymin = max(0,    int(float(bndbox.findtext("ymin", "0"))))
        xmax = min(img_w, int(float(bndbox.findtext("xmax", "0"))))
        ymax = min(img_h, int(float(bndbox.findtext("ymax", "0"))))
        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 0 or bh <= 0:
            continue

        coco_anns.append(
            {
                "category_id": LLVIP_CATEGORY_TO_CLASS_IDX[cat_id],  # class_idx
                "bbox":        [float(xmin), float(ymin), float(bw), float(bh)],
                "area":        float(bw * bh),
                "iscrowd":     0,
            }
        )
    return coco_anns


class LLVIPPairedDataset(Dataset):
    """
    LLVIP 红外-可见光配对数据集（直接读取原生 Pascal VOC XML）。

    每个样本返回：
        pixel_values      [3, H, W]  红外图像（ImageNet 归一化）
        pixel_mask        [H, W]     有效像素掩码
        labels            Dict       DINO 格式目标框（cxcywh 归一化，仅 person）
        rgb_pixel_values  [3, H, W]  对应可见光图像；若缺失则为 None
        image_path        str        红外图像的绝对路径
        rgb_path          str | None 可见光图像的绝对路径（或 None）

    RGB 配对策略：
        IR  : {root}/infrared/{split}/{stem}.jpg
        RGB : {root}/visible/{split}/{stem}.jpg   （同名直接配对）
    """

    def __init__(
        self,
        root: str,
        split: str,
        processor: Any,
        text_prompt: str,
        category_map: Optional[Dict[int, int]] = None,
        valid_cat_ids: Optional[frozenset] = None,
    ) -> None:
        """
        Args:
            root:          LLVIP 根目录（含 Annotations/ infrared/ visible/）。
            split:         "train" 或 "test"。
            processor:     AutoProcessor（GroundingDinoImageProcessor）。
            text_prompt:   检测 prompt，如 "person." 或 "person. car."
            category_map:  cat_id → class_idx；None 时使用 {1:0}（此参数保留供接口一致性）。
            valid_cat_ids: 有效 cat_id 集合；None 时使用 {1}（此参数保留供接口一致性）。
        """
        super().__init__()
        assert split in ("train", "test"), f"split 必须为 'train' 或 'test'，got: {split}"
        self._root      = os.path.abspath(root)
        self._split     = split
        self._processor = processor

        self._ann_dir     = os.path.join(self._root, "Annotations")
        self._ir_dir      = os.path.join(self._root, "infrared", split)
        self._vis_dir     = os.path.join(self._root, "visible",  split)

        if not os.path.isdir(self._ann_dir):
            raise FileNotFoundError(f"Annotations 目录不存在: {self._ann_dir}")
        if not os.path.isdir(self._ir_dir):
            raise FileNotFoundError(f"IR 图像目录不存在: {self._ir_dir}")

        # 枚举该 split 的 IR 图像列表，并检查对应 XML 是否存在
        valid_exts = {".jpg", ".jpeg", ".png"}
        all_imgs = sorted(
            p for p in Path(self._ir_dir).iterdir()
            if p.suffix.lower() in valid_exts
        )
        self._samples: List[Path] = []
        missing_xml = 0
        for p in all_imgs:
            xml_path = os.path.join(self._ann_dir, p.stem + ".xml")
            if os.path.isfile(xml_path):
                self._samples.append(p)
            else:
                missing_xml += 1

        assert len(self._samples) > 0, (
            f"在 {self._ir_dir} 中未找到任何带 XML 标注的图像（root={self._root}）"
        )

        # 预构建 visible/{split}/ 目录索引（stem → 路径）
        self._vis_stem_to_path: Dict[str, str] = {}
        if os.path.isdir(self._vis_dir):
            for fn in os.listdir(self._vis_dir):
                stem = os.path.splitext(fn)[0]
                self._vis_stem_to_path[stem] = os.path.join(self._vis_dir, fn)

        paired = sum(1 for p in self._samples if p.stem in self._vis_stem_to_path)
        print(
            f"[LLVIPPairedDataset] root={self._root}  split={split}  "
            f"有效图像={len(self._samples)}  RGB配对={paired}  缺失XML={missing_xml}"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ir_path = self._samples[index]
        stem    = ir_path.stem

        image = Image.open(str(ir_path)).convert("RGB")
        img_w, img_h = image.size

        # 实时解析 VOC XML → COCO annotations
        xml_path = os.path.join(self._ann_dir, stem + ".xml")
        coco_anns = _parse_voc_xml(xml_path, img_w, img_h)

        coco_target: Dict[str, Any] = {
            "image_id":    index,
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

        # visible 配对（同名直接查找）
        rgb_path = self._vis_stem_to_path.get(stem)
        rgb_pixel_values: Optional[torch.Tensor] = None
        if rgb_path is not None and os.path.isfile(rgb_path):
            rgb_img = Image.open(rgb_path).convert("RGB")
            if rgb_img.size != (img_w, img_h):
                rgb_img = rgb_img.resize((img_w, img_h), Image.BILINEAR)
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
            "image_path":       str(ir_path),
            "rgb_path":         rgb_path,
        }


def collate_llvip(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    LLVIP 配对数据集 collate 函数。

    - pixel_values / pixel_mask / labels：标准堆叠
    - rgb_pixel_values：全有则 stack，任一缺失则不写入
    - image_paths / rgb_paths：保留为列表

    Args:
        batch: List[Dict]，LLVIPPairedDataset.__getitem__ 的输出

    Returns:
        collated dict
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
