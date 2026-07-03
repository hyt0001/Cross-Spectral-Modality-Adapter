"""
LLVIP 数据集 Pascal VOC XML 标注 → COCO JSON 转换脚本。

LLVIP 标注格式（Pascal VOC XML）：
    <annotation>
      <filename>010001.jpg</filename>
      <size><width>1280</width><height>1024</height></size>
      <object>
        <name>person</name>
        <bndbox><xmin>287</xmin><ymin>428</ymin><xmax>351</xmax><ymax>662</ymax></bndbox>
      </object>
      ...
    </annotation>

LLVIP 目录结构（输入）：
    {llvip_root}/Annotations/{stem}.xml    — 全部 15488 个标注（不分 split）
    {llvip_root}/infrared/train/{stem}.jpg — 红外训练图像（12025）
    {llvip_root}/infrared/test/{stem}.jpg  — 红外测试图像（3463）

COCO 类别约定（CSMA 兼容）：
    cat_id=1 → person → class_idx=0
    （LLVIP 仅有 person 一类；训练时建议使用 prompt "person." 或 "person. car."）

输出：两个 COCO JSON 文件（train / test），路径由 --train-output / --test-output 指定。

用法：
    python src/convert_llvip_to_coco.py \\
        --llvip-root LLVIP \\
        --train-output LLVIP/train_coco.json \\
        --test-output  LLVIP/test_coco.json
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# LLVIP 只有 person 一类
LLVIP_CATEGORIES: List[Dict] = [
    {"id": 1, "name": "person", "supercategory": "person"},
]

# 若要与 "person. car." prompt 配合，car 无 GT 但 loss 仍可计算
LLVIP_CLASS_NAME_TO_CAT_ID: Dict[str, int] = {
    "person": 1,
}


def _parse_voc_xml(
    xml_path: str,
) -> Tuple[str, int, int, List[Dict]]:
    """
    解析单个 Pascal VOC XML 标注文件。

    Args:
        xml_path: XML 文件路径。

    Returns:
        (filename, width, height, bboxes)
        bboxes: List[{"class_name": str, "xmin": int, "ymin": int,
                       "xmax": int, "ymax": int, "difficult": int}]

    Raises:
        ValueError: 若 XML 缺少必要字段。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_elem = root.find("filename")
    if filename_elem is None or not filename_elem.text:
        raise ValueError(f"XML 缺少 <filename>: {xml_path}")
    filename: str = filename_elem.text.strip()

    size_elem = root.find("size")
    if size_elem is None:
        raise ValueError(f"XML 缺少 <size>: {xml_path}")
    width  = int(size_elem.findtext("width",  default="0"))
    height = int(size_elem.findtext("height", default="0"))

    bboxes: List[Dict] = []
    for obj in root.findall("object"):
        cls_name = (obj.findtext("name") or "").strip().lower()
        difficult = int(obj.findtext("difficult") or "0")
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = int(float(bndbox.findtext("xmin", default="0")))
        ymin = int(float(bndbox.findtext("ymin", default="0")))
        xmax = int(float(bndbox.findtext("xmax", default="0")))
        ymax = int(float(bndbox.findtext("ymax", default="0")))
        bboxes.append(
            {
                "class_name": cls_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "difficult": difficult,
            }
        )

    return filename, width, height, bboxes


def _build_coco_for_split(
    llvip_root: str,
    split: str,
    skip_difficult: bool,
) -> Dict:
    """
    为指定 split（train / test）构建 COCO 格式 dict。

    策略：
      1. 枚举 {llvip_root}/infrared/{split}/ 下的所有图像作为该 split 的文件集合
      2. 在 {llvip_root}/Annotations/ 查找同名 XML
      3. 解析 VOC XML，转换为 COCO bbox [x_min, y_min, w, h]

    Args:
        llvip_root:      LLVIP 根目录路径。
        split:           "train" 或 "test"。
        skip_difficult:  若 True，跳过 difficult=1 的目标框。

    Returns:
        COCO dict（images / annotations / categories）
    """
    ir_split_dir  = os.path.join(llvip_root, "infrared", split)
    ann_dir       = os.path.join(llvip_root, "Annotations")

    if not os.path.isdir(ir_split_dir):
        raise FileNotFoundError(f"IR split 目录不存在: {ir_split_dir}")
    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(f"Annotations 目录不存在: {ann_dir}")

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = sorted(
        p for p in Path(ir_split_dir).iterdir()
        if p.suffix.lower() in valid_exts
    )

    images: List[Dict] = []
    annotations: List[Dict] = []
    img_id  = 1
    ann_id  = 1
    missing_xml = 0
    skipped_cls = 0

    for img_path in img_files:
        stem    = img_path.stem
        xml_path = os.path.join(ann_dir, stem + ".xml")

        if not os.path.isfile(xml_path):
            missing_xml += 1
            continue

        try:
            filename, w, h, bboxes = _parse_voc_xml(xml_path)
        except (ValueError, ET.ParseError) as e:
            print(f"[警告] 解析失败: {xml_path}  ({e})，跳过")
            continue

        # 若 XML 内 width/height 为 0，尝试从文件名约定获取（LLVIP 固定 1280×1024）
        if w == 0 or h == 0:
            w, h = 1280, 1024

        coco_anns: List[Dict] = []
        for box in bboxes:
            if skip_difficult and box["difficult"] == 1:
                continue
            cat_id = LLVIP_CLASS_NAME_TO_CAT_ID.get(box["class_name"])
            if cat_id is None:
                skipped_cls += 1
                continue

            xmin = max(0, box["xmin"])
            ymin = max(0, box["ymin"])
            xmax = min(w, box["xmax"])
            ymax = min(h, box["ymax"])
            bw   = xmax - xmin
            bh   = ymax - ymin
            if bw <= 0 or bh <= 0:
                continue

            coco_anns.append(
                {
                    "id":          ann_id,
                    "image_id":    img_id,
                    "category_id": cat_id,
                    "bbox":        [xmin, ymin, bw, bh],
                    "area":        bw * bh,
                    "iscrowd":     0,
                }
            )
            ann_id += 1

        images.append(
            {
                "id":        img_id,
                "file_name": img_path.name,   # 仅文件名，不含子目录
                "width":     w,
                "height":    h,
            }
        )
        annotations.extend(coco_anns)
        img_id += 1

    print(
        f"  [{split}] 图像={len(images)}  标注框={len(annotations)}"
        f"  缺失XML={missing_xml}  未知类别框={skipped_cls}"
    )

    return {
        "info": {
            "description": "LLVIP converted from Pascal VOC to COCO format",
            "split":       split,
            "version":     "1.0",
        },
        "licenses":    [],
        "categories":  LLVIP_CATEGORIES,
        "images":      images,
        "annotations": annotations,
    }


def main() -> None:
    """主入口：解析参数并执行 VOC XML → COCO 转换。"""
    parser = argparse.ArgumentParser(
        description="LLVIP Pascal VOC XML 标注 → COCO JSON 转换",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--llvip-root", required=True,
        help="LLVIP 数据集根目录（含 Annotations/ infrared/ visible/）"
    )
    parser.add_argument(
        "--train-output", default=None,
        help="train split COCO JSON 输出路径；不传则不生成 train"
    )
    parser.add_argument(
        "--test-output", default=None,
        help="test split COCO JSON 输出路径；不传则不生成 test"
    )
    parser.add_argument(
        "--skip-difficult", action="store_true",
        help="跳过 <difficult>1</difficult> 的目标框（默认保留）"
    )
    args = parser.parse_args()

    if args.train_output is None and args.test_output is None:
        parser.error("至少需要指定 --train-output 或 --test-output 之一")

    print(f"[convert_llvip] LLVIP 根目录: {args.llvip_root}")
    print(f"[convert_llvip] skip_difficult={args.skip_difficult}")

    for split, output in [("train", args.train_output), ("test", args.test_output)]:
        if output is None:
            continue
        print(f"\n[convert_llvip] 生成 {split} COCO...")
        coco = _build_coco_for_split(args.llvip_root, split, args.skip_difficult)
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False, indent=2)
        size_kb = os.path.getsize(output) // 1024
        print(f"  已写入: {output}  ({size_kb} KB)")

    print("\n[convert_llvip] 完成。COCO 类别定义:")
    for cat in LLVIP_CATEGORIES:
        print(f"  cat_id={cat['id']}  name={cat['name']}")
    print("\n提示: 训练时建议 prompt 使用 \"person.\" 或 \"person. car.\"（仅 person 有 GT）")
    print("      --dataset llvip --data-root LLVIP --llvip-split train")


if __name__ == "__main__":
    main()
