"""
M3FD 数据集 YOLO 标注 → COCO JSON 转换脚本。

YOLO 格式（每行）：
    class_id  cx  cy  w  h      （均为归一化 0-1，相对图像宽高）

M3FD 默认类别（可被 --classes-txt 覆盖）：
    0: People  → COCO cat_id=1 (person)
    1: Car     → COCO cat_id=2 (car)
    2: Bus     → 默认跳过（--keep-bus 后映射到 car）
    3: Motorcycle → 默认跳过
    4: Lamp    → 默认跳过
    5: Truck   → 默认跳过（--keep-truck 后映射到 car）

COCO 输出 bbox 格式：[x_min, y_min, width, height]（像素，左上角原点）

用法示例：
    # 基础（只保留 People + Car）
    python src/convert_m3fd_to_coco.py \\
        --img-dir  M3FD/ir \\
        --label-dir M3FD/labels \\
        --output   M3FD/annotations_coco.json

    # 保留 Bus / Truck（归并到 car）
    python src/convert_m3fd_to_coco.py \\
        --img-dir  M3FD/ir \\
        --label-dir M3FD/labels \\
        --output   M3FD/annotations_coco.json \\
        --keep-bus --keep-truck

    # 自动 train/val 拆分（8:2）
    python src/convert_m3fd_to_coco.py \\
        --img-dir  M3FD/ir \\
        --label-dir M3FD/labels \\
        --output   M3FD/train_coco.json \\
        --val-output M3FD/val_coco.json \\
        --train-ratio 0.8 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image


# M3FD 原始类别 ID → 名称（读取 classes.txt 时可覆盖）
M3FD_DEFAULT_CLASSES: Dict[int, str] = {
    0: "People",
    1: "Car",
    2: "Bus",
    3: "Motorcycle",
    4: "Lamp",
    5: "Truck",
}

# CSMA "person. car." prompt 对应的 COCO 类别定义
# cat_id=1 → person, cat_id=2 → car
CSMA_CATEGORIES: List[Dict] = [
    {"id": 1, "name": "person", "supercategory": "person"},
    {"id": 2, "name": "car",    "supercategory": "vehicle"},
]

# M3FD class_id → COCO category_id（默认映射；Bus/Truck 通过 flag 加入）
_BASE_YOLO_TO_CAT: Dict[int, int] = {
    0: 1,  # People → person
    1: 2,  # Car    → car
}


def _load_classes(classes_txt: Optional[str]) -> Dict[int, str]:
    """
    从 classes.txt 加载类别名称映射。

    Args:
        classes_txt: classes.txt 文件路径；None 则使用 M3FD 默认映射。

    Returns:
        {class_id: class_name}
    """
    if classes_txt is None or not os.path.isfile(classes_txt):
        return dict(M3FD_DEFAULT_CLASSES)
    with open(classes_txt, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return {i: name for i, name in enumerate(lines)}


def _get_image_size(img_path: str) -> Tuple[int, int]:
    """
    返回图像 (width, height)。

    Args:
        img_path: 图像路径。

    Returns:
        (width, height)
    """
    with Image.open(img_path) as im:
        return im.width, im.height


def _build_yolo_to_cat(keep_bus: bool, keep_truck: bool) -> Dict[int, int]:
    """
    构建 YOLO class_id → COCO category_id 映射。

    Args:
        keep_bus:   若 True，Bus(2) 归并到 car(cat_id=2)。
        keep_truck: 若 True，Truck(5) 归并到 car(cat_id=2)。

    Returns:
        {yolo_class_id: coco_category_id}
    """
    mapping = dict(_BASE_YOLO_TO_CAT)
    if keep_bus:
        mapping[2] = 2   # Bus → car
    if keep_truck:
        mapping[5] = 2   # Truck → car
    return mapping


def _parse_label_file(
    label_path: str,
    img_w: int,
    img_h: int,
    yolo_to_cat: Dict[int, int],
    image_id: int,
    ann_id_start: int,
) -> Tuple[List[Dict], int]:
    """
    解析单个 YOLO .txt 标注文件，返回 COCO annotation 列表。

    YOLO 格式：每行 `class_id  cx  cy  w  h`（归一化）
    COCO bbox：[x_min, y_min, width, height]（像素）

    Args:
        label_path:    YOLO .txt 文件路径。
        img_w:         图像宽度（像素）。
        img_h:         图像高度（像素）。
        yolo_to_cat:   YOLO class_id → COCO category_id 映射。
        image_id:      对应的 COCO image_id。
        ann_id_start:  从此值开始分配 annotation id。

    Returns:
        (annotations, next_ann_id)
    """
    annotations: List[Dict] = []
    ann_id = ann_id_start

    if not os.path.isfile(label_path):
        return annotations, ann_id

    with open(label_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            yolo_cls = int(parts[0])
            if yolo_cls not in yolo_to_cat:
                continue  # 跳过不在映射中的类别

            cx_n, cy_n, w_n, h_n = (float(p) for p in parts[1:5])

            # 归一化 → 像素坐标
            x_min = (cx_n - w_n / 2) * img_w
            y_min = (cy_n - h_n / 2) * img_h
            box_w = w_n * img_w
            box_h = h_n * img_h

            # 边界裁剪（防止负值或越界）
            x_min = max(0.0, x_min)
            y_min = max(0.0, y_min)
            box_w = min(box_w, img_w - x_min)
            box_h = min(box_h, img_h - y_min)

            if box_w <= 0 or box_h <= 0:
                continue

            annotations.append(
                {
                    "id":          ann_id,
                    "image_id":    image_id,
                    "category_id": yolo_to_cat[yolo_cls],
                    "bbox":        [
                        round(x_min, 2),
                        round(y_min, 2),
                        round(box_w, 2),
                        round(box_h, 2),
                    ],
                    "area":    round(box_w * box_h, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return annotations, ann_id


def _build_coco(
    img_dir: str,
    label_dir: str,
    yolo_to_cat: Dict[int, int],
    image_ids: Optional[Set[str]] = None,
    img_suffix: str = "",
) -> Dict:
    """
    扫描 img_dir，匹配 label_dir 中同名 .txt 文件，构建 COCO 格式 dict。

    Args:
        img_dir:      图像目录（用于读取尺寸）。
        label_dir:    YOLO 标注目录（.txt 文件，stem 与图像一致）。
        yolo_to_cat:  YOLO class_id → COCO category_id 映射。
        image_ids:    若非 None，只处理 stem 在集合内的图像（用于 train/val 拆分）。
        img_suffix:   图像文件名后缀（如 "I" → 匹配 00001I.png），空串则不过滤。

    Returns:
        COCO dict（images / annotations / categories）
    """
    img_dir_path = Path(img_dir)
    label_dir_path = Path(label_dir)

    # 收集所有图像文件
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    img_files = sorted(
        p for p in img_dir_path.iterdir()
        if p.suffix.lower() in valid_exts
        and (image_ids is None or p.stem in image_ids)
    )

    images: List[Dict] = []
    annotations: List[Dict] = []
    img_id = 1
    ann_id = 1
    skipped_no_ann = 0

    for img_path in img_files:
        try:
            w, h = _get_image_size(str(img_path))
        except Exception as e:
            print(f"[警告] 读取图像失败: {img_path}  ({e})，跳过")
            continue

        label_path = label_dir_path / (img_path.stem + ".txt")
        anns, ann_id = _parse_label_file(
            str(label_path), w, h, yolo_to_cat, img_id, ann_id
        )

        if not anns:
            skipped_no_ann += 1

        images.append(
            {
                "id":        img_id,
                "file_name": img_path.name,
                "width":     w,
                "height":    h,
            }
        )
        annotations.extend(anns)
        img_id += 1

    print(
        f"  处理图像: {len(images)}  有效标注框: {len(annotations)}"
        f"  空标注图像: {skipped_no_ann}"
    )

    return {
        "info": {
            "description": "M3FD converted from YOLO to COCO format",
            "version": "1.0",
            "csma_classes": "person, car",
        },
        "licenses": [],
        "categories": CSMA_CATEGORIES,
        "images": images,
        "annotations": annotations,
    }


def _save_coco(coco: Dict, output: str) -> None:
    """
    将 COCO dict 写入 JSON 文件。

    Args:
        coco:   COCO 格式字典。
        output: 输出文件路径。
    """
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    print(f"  已写入: {output}  ({os.path.getsize(output) // 1024} KB)")


def main() -> None:
    """主入口：解析参数并执行 YOLO → COCO 转换。"""
    parser = argparse.ArgumentParser(
        description="M3FD YOLO 标注 → COCO JSON 转换",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--img-dir", required=True,
        help="IR 图像目录（用于读取图像尺寸，stem 与 label 文件对应）"
    )
    parser.add_argument(
        "--label-dir", required=True,
        help="YOLO .txt 标注目录（每个文件 stem 与图像一致）"
    )
    parser.add_argument(
        "--output", required=True,
        help="输出 COCO JSON 路径（train split）"
    )
    parser.add_argument(
        "--val-output", default=None,
        help="val split COCO JSON 路径；不传则不拆分"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="train 比例（仅在指定 --val-output 时生效）"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（用于 train/val 拆分）"
    )
    parser.add_argument(
        "--classes-txt", default=None,
        help="classes.txt 路径（可选）；默认使用 M3FD 6 类"
    )
    parser.add_argument(
        "--keep-bus", action="store_true",
        help="将 Bus(class_id=2) 归并到 car"
    )
    parser.add_argument(
        "--keep-truck", action="store_true",
        help="将 Truck(class_id=5) 归并到 car"
    )
    args = parser.parse_args()

    # Phase 1：加载类别映射
    class_names = _load_classes(args.classes_txt)
    yolo_to_cat = _build_yolo_to_cat(args.keep_bus, args.keep_truck)
    print(f"[convert_m3fd] 类别文件: {args.classes_txt or '默认 M3FD 6 类'}")
    print(f"[convert_m3fd] 原始类别: {class_names}")
    print(f"[convert_m3fd] YOLO→COCO 映射: {yolo_to_cat}")

    # Phase 2：收集图像 stem 列表（用于 train/val 拆分）
    img_dir = Path(args.img_dir)
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    all_stems = sorted(
        p.stem for p in img_dir.iterdir()
        if p.suffix.lower() in valid_exts
    )
    print(f"[convert_m3fd] 图像总数: {len(all_stems)}")

    if not all_stems:
        raise RuntimeError(f"未在 {args.img_dir} 中找到任何图像文件")

    # Phase 3：train/val 拆分（或全量）
    if args.val_output:
        rng = random.Random(args.seed)
        shuffled = list(all_stems)
        rng.shuffle(shuffled)
        split_at = int(len(shuffled) * args.train_ratio)
        train_stems: Optional[Set[str]] = set(shuffled[:split_at])
        val_stems: Optional[Set[str]]   = set(shuffled[split_at:])
        print(
            f"[convert_m3fd] train/val 拆分: "
            f"train={len(train_stems)}  val={len(val_stems)}  "
            f"ratio={args.train_ratio}  seed={args.seed}"
        )
    else:
        train_stems = None  # None 表示全量
        val_stems = None

    # Phase 4：构建并保存 COCO
    print(f"\n[convert_m3fd] 生成 train COCO...")
    train_coco = _build_coco(args.img_dir, args.label_dir, yolo_to_cat, train_stems)
    _save_coco(train_coco, args.output)

    if args.val_output and val_stems:
        print(f"\n[convert_m3fd] 生成 val COCO...")
        val_coco = _build_coco(args.img_dir, args.label_dir, yolo_to_cat, val_stems)
        _save_coco(val_coco, args.val_output)

    print("\n[convert_m3fd] 完成。COCO 类别定义:")
    for cat in CSMA_CATEGORIES:
        print(f"  cat_id={cat['id']}  name={cat['name']}")
    print(f"\n提示: 使用 --dataset m3fd --data-root <M3FD目录> 加载")


if __name__ == "__main__":
    main()
