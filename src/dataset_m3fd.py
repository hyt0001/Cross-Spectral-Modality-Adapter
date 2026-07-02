"""
M3FD 配对数据集适配器（跨数据集消融实验）。

数据集目录结构（解压后）：
    {root}/ir/*.png          — 红外图像（或 {root}/ir/ir/*.png，自动探测）
    {root}/vi/*.png          — 可见光图像（或 {root}/vi/vi/*.png，自动探测）
    {root}/annotations/val.json  — COCO 格式标注

数据集信息：
    - 来源：CVPR 2022 Oral，TarDAL 论文，大连理工大学
    - 图像对数：4200 对，分辨率 1024×768
    - 类别（6 类）：Bus(1) / Car(2) / Lamp(3) / Motorcycle(4) / People(5) / Truck(6)
    - 场景：白天 / 阴天 / 夜晚 / 挑战（含迷彩、烟雾、森林等 10 种子场景）
    - 全部 4200 张用于评测（无官方 train/val 划分，取全量作 test）

与 LLVIP 的关键差异：
    - 6 个类别（LLVIP 仅 person）
    - file_name 为 00000.png 形式，无子目录前缀
    - IR/VI 分属 ir/ 与 vi/ 子目录（zip 解压后可能有 ir/ir/ 双层，自动处理）

Eval 类别 ID 映射（与 DINO 文本标签对应）：
    M3FD cat_id  →  eval cat_id  →  DINO label
    5 (People)  →      1        →  "person"
    2 (Car)     →      2        →  "car"
    1 (Bus)     →      3        →  "bus"
    4 (Motorcycle) →   4        →  "motorcycle"
    6 (Truck)   →      5        →  "truck"
    3 (Lamp)    →      6        →  "lamp"

对应 docs/TD.md §3.x M3FD 跨数据集消融实验。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

# ── 类别映射 ─────────────────────────────────────────────────────────────────
# M3FD COCO 标注中的 category_id（原始 JSON 内 1-based）
M3FD_DEFAULT_TEXT_PROMPT: str = "person. car."
M3FD_FULL_TEXT_PROMPT: str = "person. car. bus. motorcycle. truck. lamp."

# prompt 词 → M3FD 原始 category_id
M3FD_LABEL_TO_RAW_CAT_ID: Dict[str, int] = {
    "person": 5,
    "people": 5,
    "car": 2,
    "bus": 1,
    "motorcycle": 4,
    "motor": 4,
    "truck": 6,
    "lamp": 3,
}

# 六类全量映射（向后兼容引用）
M3FD_CATEGORY_TO_EVAL_ID: Dict[int, int] = {
    5: 1, 2: 2, 1: 3, 4: 4, 6: 5, 3: 6,
}
M3FD_CATEGORY_TO_CLASS_IDX: Dict[int, int] = {
    5: 0, 2: 1, 1: 2, 4: 3, 6: 4, 3: 5,
}
M3FD_VALID_CAT_IDS: frozenset = frozenset(M3FD_CATEGORY_TO_EVAL_ID.keys())
M3FD_LABEL_TO_EVAL_CAT: Dict[str, int] = {
    "person": 1, "people": 1, "car": 2, "bus": 3,
    "motorcycle": 4, "motor": 4, "truck": 5, "lamp": 6,
}
M3FD_EVAL_CATEGORIES: List[Dict[str, Any]] = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "car"},
    {"id": 3, "name": "bus"},
    {"id": 4, "name": "motorcycle"},
    {"id": 5, "name": "truck"},
    {"id": 6, "name": "lamp"},
]


def parse_m3fd_prompt_segments(text_prompt: str) -> List[str]:
    """
    解析 text_prompt 为有序类别词列表（小写、去尾点）。

    Args:
        text_prompt: 如 ``"person. car."``。

    Returns:
        有序 segment 列表，如 ``["person", "car"]``。
    """
    normalized = text_prompt.strip().lower()
    segments = [s.strip().rstrip(".") for s in normalized.split(".") if s.strip()]
    if not segments:
        raise ValueError(f"M3FD text_prompt 为空: {text_prompt!r}")
    if not any(s in ("person", "people") for s in segments):
        raise ValueError(
            f"M3FD text_prompt 至少需要包含 'person' 或 'people'，当前: {segments}"
        )
    return segments


def _canonical_eval_name(segment: str) -> str:
    """将 prompt segment 规范为 COCO eval 类别名。"""
    if segment in ("person", "people"):
        return "person"
    return segment


def build_m3fd_eval_categories(text_prompt: str) -> List[Dict[str, Any]]:
    """
    按 prompt 顺序生成 COCO eval categories（仅含 prompt 中的类）。

    Args:
        text_prompt: 检测 prompt。

    Returns:
        ``[{"id": 1, "name": "person"}, ...]``，id 为 1-based。
    """
    segments = parse_m3fd_prompt_segments(text_prompt)
    cats: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        if seg not in M3FD_LABEL_TO_RAW_CAT_ID:
            raise ValueError(f"M3FD 不支持的 prompt 类别: {seg!r}")
        cats.append({"id": idx + 1, "name": _canonical_eval_name(seg)})
    return cats


def build_m3fd_label_to_eval_cat(text_prompt: str) -> Dict[str, int]:
    """
    构建 DINO 输出标签前缀 → eval cat_id 映射（仅 prompt 中的类）。

    Args:
        text_prompt: 检测 prompt。

    Returns:
        如 ``{"person": 1, "people": 1, "car": 2}``。
    """
    segments = parse_m3fd_prompt_segments(text_prompt)
    mapping: Dict[str, int] = {}
    for idx, seg in enumerate(segments):
        eval_id = idx + 1
        name = _canonical_eval_name(seg)
        mapping[name] = eval_id
        if name == "person":
            mapping["people"] = eval_id
        if seg == "motorcycle":
            mapping["motor"] = eval_id
    return mapping


def build_m3fd_category_map(
    text_prompt: str,
) -> Tuple[Dict[int, int], frozenset]:
    """
    按 text_prompt 返回 (M3FD原始cat_id→eval_cat_id, 有效原始cat_id集合)。

    仅保留 prompt 中出现的类别；bus 等未出现在 prompt 中的 GT/预测均被忽略。

    Args:
        text_prompt: 检测 prompt，推荐 ``"person. car."``。

    Returns:
        category_map: M3FD 原始 cat_id → 1-based eval_cat_id。
        valid_cat_ids: 参与训练/评测的 M3FD 原始 cat_id 集合。
    """
    segments = parse_m3fd_prompt_segments(text_prompt)
    cat_map: Dict[int, int] = {}
    valid: set[int] = set()
    for idx, seg in enumerate(segments):
        raw_id = M3FD_LABEL_TO_RAW_CAT_ID.get(seg)
        if raw_id is None:
            raise ValueError(f"M3FD 不支持的 prompt 类别: {seg!r}")
        cat_map[raw_id] = idx + 1
        valid.add(raw_id)
    return cat_map, frozenset(valid)


def build_m3fd_category_map_for_training(
    text_prompt: str,
) -> Tuple[Dict[int, int], frozenset]:
    """
    按 text_prompt 返回 (M3FD原始cat_id→0-based class_idx, 有效cat_id集合)。

    class_idx 与 DINO tokenizer 中词语顺序一致（0=第一个词，1=第二个词…）。

    Args:
        text_prompt: 检测 prompt，推荐 ``"person. car."``。

    Returns:
        category_map: M3FD 原始 cat_id → 0-based class_idx。
        valid_cat_ids: 参与 L_det 的 GT 类别集合（未在 prompt 中的框被丢弃）。
    """
    segments = parse_m3fd_prompt_segments(text_prompt)
    cat_map: Dict[int, int] = {}
    valid: set[int] = set()
    for idx, seg in enumerate(segments):
        raw_id = M3FD_LABEL_TO_RAW_CAT_ID.get(seg)
        if raw_id is None:
            raise ValueError(f"M3FD 不支持的 prompt 类别: {seg!r}")
        cat_map[raw_id] = idx
        valid.add(raw_id)
    return cat_map, frozenset(valid)


def _probe_image_dir(root: str, subname: str) -> str:
    """
    自动探测图像子目录路径。
    优先尝试 {root}/{subname}/ 直接有图像；
    若该目录下只有一个同名子目录（如 ir/ir/），则返回内层路径。

    Args:
        root:    数据集根目录
        subname: "ir" 或 "vi"

    Returns:
        实际包含图像文件的目录绝对路径

    Raises:
        FileNotFoundError: 两层均不存在
    """
    outer = os.path.join(root, subname)
    if not os.path.isdir(outer):
        raise FileNotFoundError(f"图像目录不存在: {outer}")
    # 检查外层是否直接有图像
    has_images = any(
        f.lower().endswith((".png", ".jpg", ".bmp", ".tiff"))
        for f in os.listdir(outer)
    )
    if has_images:
        return outer
    # 尝试内层（如 ir/ir/）
    inner = os.path.join(outer, subname)
    if os.path.isdir(inner):
        return inner
    # 尝试外层下的第一个子目录
    subdirs = [
        d for d in os.listdir(outer)
        if os.path.isdir(os.path.join(outer, d))
    ]
    if len(subdirs) == 1:
        return os.path.join(outer, subdirs[0])
    raise FileNotFoundError(
        f"无法在 {outer} 下找到图像文件，子目录: {subdirs}"
    )


class M3FDPairedDataset(Dataset):
    """
    M3FD RGB-IR 配对数据集（全量用于测试/消融）。

    每个样本返回：
        pixel_values      [3, H, W]  红外图像（ImageNet 归一化）
        pixel_mask        [H, W]     有效像素掩码
        labels            Dict       DINO 格式目标框（cxcywh 归一化）
        rgb_pixel_values  [3, H, W]  对应可见光图像；缺失时为 None
        image_path        str        红外图像绝对路径
        rgb_path          str | None 可见光图像绝对路径

    路径解析（自动探测双层目录）：
        IR  : {root}/ir[/ir]/{stem}.png
        RGB : {root}/vi[/vi]/{stem}.png
    """

    def __init__(
        self,
        root: str,
        processor: Any,
        text_prompt: str,
        ann_file: Optional[str] = None,
        category_map: Optional[Dict[int, int]] = None,
        valid_cat_ids: Optional[frozenset] = None,
        split: str = "all",
        canonical_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            root:           M3FD 根目录（含 ir/、vi/、annotations/）。
            processor:      AutoProcessor（GroundingDinoImageProcessor）。
            text_prompt:    检测 prompt，如 "person. car. bus. motorcycle. truck. lamp."。
            ann_file:       COCO 标注 JSON 路径；None 时默认 {root}/annotations/val.json。
            category_map:   M3FD cat_id → eval_cat_id；None 时使用默认 6 类全量映射。
            valid_cat_ids:  有效 cat_id 集合；None 时使用所有 6 类。
            split:          数据集划分："all"=全量（默认）、"train"=前 80%、"val"=后 20%。
                            M3FD 无官方 train/val 划分，按 image_id 排序后切分。
            canonical_size: 可选 (W, H)；设置后在 processor 前将所有图像 resize 到该尺寸，
                            确保 batch 内张量同形（无需 padding），推荐训练时设为 (1024, 768)。
                            None=保持原始尺寸（默认，eval 时使用）。
        """
        if split not in ("all", "train", "val"):
            raise ValueError(f"split 须为 'all'/'train'/'val'，当前: {split!r}")
        self._canonical_size: Optional[Tuple[int, int]] = canonical_size
        super().__init__()
        self._root = os.path.abspath(root)
        self._processor = processor
        self._text_prompt = text_prompt
        self._cat_map: Dict[int, int] = (
            category_map if category_map is not None
            else dict(M3FD_CATEGORY_TO_EVAL_ID)
        )
        self._valid_ids: frozenset = (
            valid_cat_ids if valid_cat_ids is not None
            else frozenset(M3FD_VALID_CAT_IDS)
        )

        # 标注文件
        if ann_file is None:
            ann_file = os.path.join(self._root, "annotations", "val.json")
        ann_path = os.path.abspath(ann_file)
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"未找到 M3FD 标注文件: {ann_path}")

        with open(ann_path, encoding="utf-8") as f:
            coco = json.load(f)

        # 仅保留含有效类别标注的图像（按 id 排序，保证 split 结果确定性）
        valid_image_ids: set = {
            a["image_id"]
            for a in coco["annotations"]
            if int(a["category_id"]) in self._valid_ids
        }
        all_images: List[Dict[str, Any]] = sorted(
            [img for img in coco["images"] if img["id"] in valid_image_ids],
            key=lambda x: x["id"],
        )
        # 按 split 做 80/20 划分
        n_total = len(all_images)
        n_train = int(n_total * 0.8)
        if split == "train":
            self._images = all_images[:n_train]
        elif split == "val":
            self._images = all_images[n_train:]
        else:
            self._images = all_images
        self._id_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for a in coco["annotations"]:
            cid = int(a["category_id"])
            if cid not in self._valid_ids:
                continue
            self._id_to_anns.setdefault(a["image_id"], []).append(a)

        assert len(self._images) > 0, (
            f"M3FD 过滤后无有效图像（root={self._root}），"
            f"请检查标注中 category_id 是否包含 {self._valid_ids}"
        )

        # 自动探测图像目录
        self._ir_dir = _probe_image_dir(self._root, "ir")
        self._rgb_dir: Optional[str]
        try:
            self._rgb_dir = _probe_image_dir(self._root, "vi")
        except FileNotFoundError:
            self._rgb_dir = None

        # 过滤磁盘缺失的 IR 文件
        filtered: List[Dict[str, Any]] = []
        paired = 0
        skipped = 0
        for img in self._images:
            ir_p = self._find_image(self._ir_dir, img["file_name"])
            if ir_p is None:
                skipped += 1
                continue
            filtered.append(img)
            if self._rgb_dir is not None:
                rgb_p = self._find_image(self._rgb_dir, img["file_name"])
                if rgb_p is not None:
                    paired += 1

        if skipped > 0:
            print(f"[M3FDPairedDataset] 警告: {skipped} 张图像因 IR 文件缺失被跳过")

        self._images = filtered
        assert len(self._images) > 0, (
            f"M3FD 过滤后无有效图像（ir_dir={self._ir_dir}）"
        )
        print(
            f"[M3FDPairedDataset] root={self._root}  split={split}  "
            f"有效图像={len(self._images)}  RGB配对={paired}  "
            f"ir_dir={self._ir_dir}"
        )

    @staticmethod
    def _find_image(directory: str, file_name: str) -> Optional[str]:
        """
        在 directory 下查找与 file_name 同 stem 的图像文件（兼容多种扩展名）。

        Returns:
            绝对路径；不存在时返回 None
        """
        stem = os.path.splitext(os.path.basename(file_name))[0]
        for ext in (".png", ".jpg", ".bmp", ".tiff"):
            p = os.path.join(directory, f"{stem}{ext}")
            if os.path.isfile(p):
                return p
        return None

    def __len__(self) -> int:
        return len(self._images)

    def _resolve_paths(self, file_name: str) -> Tuple[str, Optional[str]]:
        """根据 COCO file_name 解析红外与可见光绝对路径。"""
        ir_path = self._find_image(self._ir_dir, file_name)
        if ir_path is None:
            raise FileNotFoundError(
                f"红外图像不存在: stem={os.path.splitext(file_name)[0]} in {self._ir_dir}"
            )
        rgb_path: Optional[str] = None
        if self._rgb_dir is not None:
            rgb_path = self._find_image(self._rgb_dir, file_name)
        return ir_path, rgb_path

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_info = self._images[index]
        img_path, rgb_path = self._resolve_paths(img_info["file_name"])

        image = Image.open(img_path).convert("RGB")
        # canonical_size: 训练时统一 resize，消除 batch padding 导致的坐标系不一致
        if self._canonical_size is not None and image.size != self._canonical_size:
            image = image.resize(self._canonical_size, Image.BILINEAR)
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
        if rgb_path is not None:
            rgb_img = Image.open(rgb_path).convert("RGB")
            # 当设置了 canonical_size 时，RGB 也必须 resize 到相同尺寸，
            # 否则 processor 输出形状不同，feat_rgb/feat_ir 维度不匹配
            target_wh = (
                self._canonical_size
                if self._canonical_size is not None
                else (int(img_info["width"]), int(img_info["height"]))
            )
            if rgb_img.size != target_wh:
                rgb_img = rgb_img.resize(target_wh, Image.BILINEAR)
            rgb_enc = self._processor.image_processor(
                images=rgb_img,
                return_tensors="pt",
            )
            rgb_pixel_values = rgb_enc["pixel_values"][0]

        return {
            "pixel_values":     pixel_values,
            "pixel_mask":       pixel_mask,
            "labels":           labels,
            "rgb_pixel_values": rgb_pixel_values,
            "image_path":       img_path,
            "rgb_path":         rgb_path,
        }


def _pad_stack_tensors(
    tensors: List[torch.Tensor],
    pad_value: float,
) -> torch.Tensor:
    """
    将 [C,H,W] 或 [H,W] 张量列表 pad 到 batch 内最大 H/W 后 stack。

    M3FD 含多种原始分辨率，GroundingDinoImageProcessor 保宽高比 resize 后
    单张形状一致但 batch 内可能不同，须右下 padding 后才能 stack。

    Args:
        tensors:   待合并的张量列表，ndim 须均为 2 或 3。
        pad_value: padding 填充值；pixel_mask 用 0，pixel_values 用 0。

    Returns:
        [B, C, H, W] 或 [B, H, W] 张量。
    """
    if not tensors:
        raise ValueError("tensors 不能为空")
    ndim = tensors[0].ndim
    if ndim not in (2, 3):
        raise ValueError(f"不支持的 ndim={ndim}，仅支持 2 或 3")

    max_h = max(t.shape[-2] for t in tensors)
    max_w = max(t.shape[-1] for t in tensors)

    padded: List[torch.Tensor] = []
    for t in tensors:
        pad_h = max_h - t.shape[-2]
        pad_w = max_w - t.shape[-1]
        if pad_h or pad_w:
            # F.pad: (left, right, top, bottom)
            t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), value=pad_value)
        padded.append(t)
    return torch.stack(padded, dim=0)


def collate_m3fd(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    M3FD 配对数据集 collate 函数（与 collate_llvip 接口一致）。

    M3FD 含多种原始分辨率，processor 保宽高比 resize 后 batch 内尺寸可能不同。
    本函数将 pixel_values/pixel_mask pad 到 batch 内最大 H/W，同时将 labels["boxes"]
    （cxcywh，原始尺寸归一化）重新归一化到 padded 尺寸，确保 DINO loss 匹配器
    在 padded 坐标系中看到合法的 box 坐标。

    Args:
        batch: M3FDPairedDataset.__getitem__ 输出列表

    Returns:
        collated dict，含 pixel_values / pixel_mask / labels /
        image_paths / rgb_paths / rgb_pixel_values（仅当全部有效时）
    """
    # 记录每张图 pad 前的 (H, W)，用于后续 box 坐标缩放
    orig_hw: List[tuple] = [
        (b["pixel_values"].shape[-2], b["pixel_values"].shape[-1]) for b in batch
    ]

    pixel_values = _pad_stack_tensors(
        [b["pixel_values"] for b in batch], pad_value=0.0
    )
    pixel_mask = _pad_stack_tensors(
        [b["pixel_mask"] for b in batch], pad_value=0.0
    )
    H_max = pixel_values.shape[-2]
    W_max = pixel_values.shape[-1]

    # 将 boxes 缩放到 padded 坐标系（仅当 batch 内存在不同尺寸时才有必要）
    labels: List[Dict[str, Any]] = []
    for b, (H_i, W_i) in zip(batch, orig_hw):
        entry: Dict[str, Any] = {}
        for k, v in b["labels"].items():
            if k == "boxes" and isinstance(v, torch.Tensor) and v.numel() > 0:
                # cxcywh 归一化，x 轴缩放因子 = W_i/W_max，y 轴 = H_i/H_max
                boxes = v.clone()
                sx = W_i / W_max
                sy = H_i / H_max
                boxes[:, 0] = boxes[:, 0] * sx   # cx
                boxes[:, 1] = boxes[:, 1] * sy   # cy
                boxes[:, 2] = boxes[:, 2] * sx   # w
                boxes[:, 3] = boxes[:, 3] * sy   # h
                entry[k] = boxes
            else:
                entry[k] = v.clone() if isinstance(v, torch.Tensor) else v
        labels.append(entry)

    result: Dict[str, Any] = {
        "pixel_values": pixel_values,
        "pixel_mask":   pixel_mask,
        "labels":       labels,
        "image_paths":  [b["image_path"] for b in batch],
        "rgb_paths":    [b.get("rgb_path") for b in batch],
    }

    rgb_list: List[Optional[torch.Tensor]] = [b.get("rgb_pixel_values") for b in batch]
    if all(v is not None for v in rgb_list):
        result["rgb_pixel_values"] = _pad_stack_tensors(
            rgb_list,  # type: ignore[arg-type]
            pad_value=0.0,
        )

    return result
