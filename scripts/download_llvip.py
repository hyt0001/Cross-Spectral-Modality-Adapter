"""
LLVIP 数据集下载与校验工具。

数据来源（按优先级）：
  1. HuggingFace 镜像（UserNae3/LLVIP，约 4GB）
  2. Zenodo COCO 标注（CAFF-DINO，train.json + val.json）

使用方式：
  # 下载图像 + 标注（默认）
  conda run -n RGBtest python scripts/download_llvip.py

  # 仅校验现有数据
  conda run -n RGBtest python scripts/download_llvip.py --check-only

  # 指定目标目录
  conda run -n RGBtest python scripts/download_llvip.py --dataset-root LLVIP
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

# HuggingFace 镜像（国内/Google Drive 不可达时的备选）
HF_LLVIP_ZIP = (
    "https://hf-mirror.com/datasets/UserNae3/LLVIP/resolve/main/LLVIP.zip"
)
ZENODO_COCO_ZIP = (
    "https://zenodo.org/api/records/13907794/files/LLVIP_coco.zip/content"
)

EXPECTED_SPLITS = ("train", "test")


def _count_images(directory: Path) -> int:
    """统计目录下 jpg/jpeg 图像数量。"""
    if not directory.is_dir():
        return 0
    return sum(
        1 for f in directory.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg")
    )


def check_dataset(root: Path) -> dict:
    """
    诊断 LLVIP 目录完整性。

    Returns:
        诊断信息字典
    """
    result = {
        "root": str(root),
        "ir_train": 0,
        "ir_test": 0,
        "rgb_train": 0,
        "rgb_test": 0,
        "has_val_ann": False,
        "has_train_ann": False,
        "ok": False,
    }
    for split in EXPECTED_SPLITS:
        ir_dir = root / "infrared" / split
        rgb_dir = root / "visible" / split
        result[f"ir_{split}"] = _count_images(ir_dir)
        result[f"rgb_{split}"] = _count_images(rgb_dir)

    val_ann = root / "annotations" / "val.json"
    train_ann = root / "annotations" / "train.json"
    result["has_val_ann"] = val_ann.is_file()
    result["has_train_ann"] = train_ann.is_file()

    result["ok"] = (
        result["ir_test"] > 0
        and result["rgb_test"] > 0
        and result["has_val_ann"]
    )
    return result


def _wget(url: str, dest: Path) -> None:
    """使用 wget 下载文件（支持断点续传）。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget", "-c", "--timeout=60", "--tries=5",
        "-O", str(dest), url,
    ]
    print(f"[download_llvip] 下载: {url}")
    print(f"[download_llvip] 目标: {dest}")
    subprocess.run(cmd, check=True)


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """解压 zip 到目标目录。"""
    print(f"[download_llvip] 解压: {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def _normalize_layout(root: Path) -> None:
    """
    将 HuggingFace 压缩包解压后的目录整理为项目标准布局。

    标准布局：
        {root}/infrared/train|test/*.jpg
        {root}/visible/train|test/*.jpg
        {root}/annotations/val.json
    """
    # HuggingFace 包内通常为 LLVIP/infrared/... 嵌套
    nested = root / "LLVIP"
    if nested.is_dir() and (nested / "infrared").is_dir():
        for name in ("infrared", "visible", "Annotations"):
            src = nested / name
            if not src.is_dir():
                continue
            dst_name = name.lower() if name == "Annotations" else name
            dst = root / dst_name
            if dst.exists() and dst != src:
                continue
            if not dst.exists():
                shutil.move(str(src), str(dst))

    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)


def download_annotations(root: Path) -> None:
    """下载并解压 Zenodo COCO 标注。"""
    ann_dir = root / "annotations"
    if (ann_dir / "val.json").is_file() and (ann_dir / "train.json").is_file():
        print("[download_llvip] COCO 标注已存在，跳过")
        return

    zip_path = root / "_LLVIP_coco.zip"
    _wget(ZENODO_COCO_ZIP, zip_path)
    _extract_zip(zip_path, ann_dir)
    zip_path.unlink(missing_ok=True)
    print("[download_llvip] COCO 标注已就绪")


def download_images(root: Path) -> None:
    """从 HuggingFace 下载并解压 LLVIP 图像。"""
    ir_test = root / "infrared" / "test"
    if _count_images(ir_test) > 1000:
        print(f"[download_llvip] 图像已存在（test={_count_images(ir_test)}），跳过")
        return

    zip_path = root / "LLVIP.zip"
    if not zip_path.is_file() or zip_path.stat().st_size < 1_000_000_000:
        _wget(HF_LLVIP_ZIP, zip_path)

    _extract_zip(zip_path, root)
    _normalize_layout(root)
    print("[download_llvip] 图像解压完成")


def main(argv: Optional[list[str]] = None) -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="LLVIP 数据集下载与校验")
    parser.add_argument(
        "--dataset-root", type=str, default="LLVIP",
        help="数据集根目录（默认 LLVIP/）",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="仅诊断，不下载",
    )
    parser.add_argument(
        "--annotations-only", action="store_true",
        help="仅下载 COCO 标注",
    )
    args = parser.parse_args(argv)

    root = Path(args.dataset_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.check_only:
        info = check_dataset(root)
        print("=== LLVIP 数据诊断 ===")
        for k, v in info.items():
            print(f"  {k}: {v}")
        sys.exit(0 if info["ok"] else 1)

    if not args.annotations_only:
        download_images(root)
    download_annotations(root)

    info = check_dataset(root)
    print("=== 下载完成，数据诊断 ===")
    for k, v in info.items():
        print(f"  {k}: {v}")

    if not info["ok"]:
        print("[download_llvip] 警告: 数据不完整，请检查网络或手动放置文件")
        sys.exit(1)


if __name__ == "__main__":
    main()
