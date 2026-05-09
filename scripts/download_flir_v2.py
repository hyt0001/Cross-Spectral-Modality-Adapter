"""
FLIR_ADAS_v2 数据集诊断与下载工具。

使用方式：
  # 仅诊断现有数据状态（无需 API Key）
  conda run -n RGBtest python scripts/download_flir_v2.py --check-only

  # 下载热红外图像 + 完整 coco.json（需 Roboflow API Key）
  conda run -n RGBtest python scripts/download_flir_v2.py \\
    --api-key YOUR_ROBOFLOW_API_KEY \\
    --dataset-root FLIR_ADAS_v2

背景：FLIR_ADAS_v2 通过 Roboflow 导出，各 JSON 文件被限制在 512KB，
导致 coco.json / index.json 均截断。同时热红外图像（images_thermal_train/data/）
未被下载，需通过 Roboflow API 补全。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────────────────

FLIR_V2_SPLITS = [
    ("images_thermal_train", "train", "thermal"),
    ("images_thermal_val",   "val",   "thermal"),
    ("images_rgb_train",     "train", "rgb"),
    ("images_rgb_val",       "val",   "rgb"),
]

# Roboflow API 端点模板
RF_EXPORT_URL = "https://api.roboflow.com/{workspace}/{project}/{version}/coco"


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: 数据诊断
# ──────────────────────────────────────────────────────────────────────────────

def _count_files(directory: str, exts: tuple[str, ...]) -> int:
    """统计目录下指定扩展名文件数（递归）。"""
    d = Path(directory)
    if not d.is_dir():
        return 0
    return sum(1 for f in d.rglob("*") if f.suffix.lower() in exts)


def _check_json(json_path: str) -> dict:
    """
    检查 JSON 文件是否完整可解析。

    Returns:
        {"exists": bool, "size_kb": float, "parseable": bool, "has_images": bool,
         "has_categories": bool, "n_annotations": int, "truncated": bool}
    """
    result = {
        "exists": False, "size_kb": 0.0, "parseable": False,
        "has_images": False, "has_categories": False, "n_annotations": 0,
        "truncated": False,
    }
    p = Path(json_path)
    if not p.exists():
        return result
    result["exists"] = True
    result["size_kb"] = p.stat().st_size / 1024
    # 文件大小恰好 = 512KB → 极可能截断
    if abs(p.stat().st_size - 524288) < 10:
        result["truncated"] = True
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        result["parseable"] = True
        result["has_images"] = "images" in data
        result["has_categories"] = "categories" in data
        result["n_annotations"] = len(data.get("annotations", []))
    except json.JSONDecodeError:
        pass
    return result


def check_data_status(dataset_root: str) -> None:
    """
    全面诊断 FLIR_ADAS_v2 数据集状态，打印报告。

    Args:
        dataset_root: FLIR_ADAS_v2 根目录（如 "FLIR_ADAS_v2"）。
    """
    root = Path(dataset_root)
    print("=" * 60)
    print(f"FLIR_ADAS_v2 数据集诊断报告")
    print(f"根目录: {root.resolve()}")
    print("=" * 60)

    all_ok = True
    for folder, split, modality in FLIR_V2_SPLITS:
        split_dir = root / folder
        print(f"\n[{folder}]")

        # 图像文件
        data_dir = split_dir / "data"
        n_jpg = _count_files(str(data_dir), (".jpg", ".jpeg", ".png"))
        n_tiff = _count_files(str(data_dir), (".tiff", ".tif"))
        # analyticsData 只含元数据占位，不算实际图像
        analytics_tiffs = _count_files(str(split_dir / "analyticsData"), (".tiff", ".tif"))

        if modality == "thermal":
            status = "✅" if n_jpg + n_tiff > 0 else "❌ 热红外图像缺失！"
            if n_jpg + n_tiff == 0:
                all_ok = False
        else:
            status = "✅" if n_jpg > 0 else "❌ RGB 图像缺失！"
            if n_jpg == 0:
                all_ok = False

        print(f"  data/ 图像数:        JPEG={n_jpg}  TIFF={n_tiff}  {status}")
        if analytics_tiffs > 0:
            print(f"  analyticsData/ TIFF: {analytics_tiffs} (0-byte 占位符，非真实图像)")

        # coco.json 状态
        coco_path = split_dir / "coco.json"
        cj = _check_json(str(coco_path))
        if not cj["exists"]:
            print(f"  coco.json:          ❌ 不存在")
            all_ok = False
        elif cj["truncated"]:
            print(
                f"  coco.json:          ⚠ 截断（{cj['size_kb']:.0f} KB = 512KB 精确，"
                f"parseable={cj['parseable']}，has_images={cj['has_images']}，"
                f"has_categories={cj['has_categories']}）"
            )
            all_ok = False
        elif cj["parseable"] and cj["has_images"] and cj["has_categories"]:
            print(
                f"  coco.json:          ✅ 完整（{cj['size_kb']:.0f} KB，"
                f"annotations={cj['n_annotations']}）"
            )
        else:
            print(f"  coco.json:          ⚠ 存在但不完整")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("✅ 数据集完整，可直接运行训练。")
    else:
        print("❌ 数据集不完整，请按以下步骤修复：")
        print()
        print("  方法 1（推荐）— 使用 Roboflow API Key 重新下载：")
        print("    conda run -n RGBtest python scripts/download_flir_v2.py \\")
        print("      --api-key YOUR_API_KEY --dataset-root FLIR_ADAS_v2")
        print()
        print("  方法 2 — 从 FLIR 官网重新下载 FLIR_ADAS_v2：")
        print("    https://www.flir.com/oem/adas/adas-dataset-form/")
        print()
        print("  方法 3 — 使用已有 train/ 目录（11张IR，仅供 smoke test）：")
        print("    CUDA_VISIBLE_DEVICES=0 conda run -n RGBtest python -m src.train_csma \\")
        print("      --dataset legacy --data-root train --epochs 2 --batch-size 2")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Roboflow 下载
# ──────────────────────────────────────────────────────────────────────────────

def _read_roboflow_project_info(index_path: str) -> Optional[dict]:
    """
    从 index.json 提取 Roboflow 数据集信息（datasetId, datasetName）。

    index.json 同样被截断，但头部信息在前几百字节内，通常可用。
    """
    p = Path(index_path)
    if not p.exists():
        return None
    try:
        # 只读前 2KB 足够获取 datasetId / datasetName
        with open(p, encoding="utf-8") as f:
            raw = f.read(2048)
        # 找 datasetId
        import re
        did_m = re.search(r'"datasetId"\s*:\s*"([^"]+)"', raw)
        dname_m = re.search(r'"datasetName"\s*:\s*"([^"]+)"', raw)
        if did_m and dname_m:
            return {
                "dataset_id": did_m.group(1),
                "dataset_name": dname_m.group(1),
            }
    except Exception:
        pass
    return None


def download_thermal_via_roboflow(
    api_key: str,
    dataset_root: str,
    workspace: str = "flir-dataengineering",
) -> None:
    """
    通过 Roboflow Python SDK 下载 FLIR_ADAS_v2 热红外数据集。

    需要安装 roboflow SDK：
        conda run -n RGBtest pip install roboflow

    Args:
        api_key:      Roboflow API Key
        dataset_root: FLIR_ADAS_v2 根目录
        workspace:    Roboflow workspace slug（FLIR 官方 workspace）
    """
    try:
        from roboflow import Roboflow  # type: ignore[import-not-found]
    except ImportError:
        print("请先安装 roboflow SDK：")
        print("  conda run -n RGBtest pip install roboflow")
        sys.exit(1)

    root = Path(dataset_root)
    # 读取 thermal train 的项目信息
    thermal_index = root / "images_thermal_train" / "index.json"
    info = _read_roboflow_project_info(str(thermal_index))
    if info:
        print(f"[Roboflow] 检测到数据集: {info['dataset_name']} (id={info['dataset_id']})")

    rf = Roboflow(api_key=api_key)
    # FLIR ADAS v2 thermal 数据集的 Roboflow slug
    # 用户可能需要根据自己项目的实际 workspace/project 修改
    thermal_project_slugs = [
        ("flir-dataengineering", "flir-adas-v2-thermal"),
        ("flir-dataengineering", "ces-images-thermal-img"),
    ]

    downloaded = False
    for ws, proj in thermal_project_slugs:
        try:
            print(f"[Roboflow] 尝试下载: workspace={ws}, project={proj}")
            project = rf.workspace(ws).project(proj)
            version = project.version(1)
            dataset = version.download("coco", location=str(root / "images_thermal_train_new"))
            print(f"[Roboflow] 下载完成: {dataset.location}")
            downloaded = True
            break
        except Exception as e:
            print(f"[Roboflow] 尝试失败: {e}")
            continue

    if not downloaded:
        print()
        print("自动下载失败。请手动从 Roboflow 获取数据集：")
        print("  1. 登录 https://app.roboflow.com")
        print("  2. 搜索 'FLIR ADAS v2 thermal' 或访问 FLIR 的 workspace")
        print("  3. 导出为 COCO JSON 格式")
        print("  4. 将 thermal 图像解压至 FLIR_ADAS_v2/images_thermal_train/data/")
        print("  5. 将完整 coco.json 放置在 FLIR_ADAS_v2/images_thermal_train/coco.json")


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FLIR_ADAS_v2 数据集诊断与下载工具"
    )
    parser.add_argument(
        "--dataset-root",
        default="FLIR_ADAS_v2",
        help="FLIR_ADAS_v2 根目录（默认: FLIR_ADAS_v2）",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅诊断，不下载",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Roboflow API Key（用于下载热红外图像）",
    )
    parser.add_argument(
        "--workspace",
        default="flir-dataengineering",
        help="Roboflow workspace slug",
    )
    args = parser.parse_args()

    check_data_status(args.dataset_root)

    if args.check_only:
        return

    if args.api_key:
        print("\n[下载] 开始通过 Roboflow API 下载热红外图像...")
        download_thermal_via_roboflow(
            api_key=args.api_key,
            dataset_root=args.dataset_root,
            workspace=args.workspace,
        )
    else:
        print("\n[提示] 未提供 --api-key，跳过下载步骤。")
        print("  若要下载，请运行：")
        print("  conda run -n RGBtest python scripts/download_flir_v2.py \\")
        print("    --api-key YOUR_API_KEY --dataset-root FLIR_ADAS_v2")


if __name__ == "__main__":
    main()
