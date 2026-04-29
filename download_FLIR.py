"""
从 FLIR 数据中只取出 val 集：
  - Annotated_thermal_8_bit/  （约 1000+ 张 8-bit 热图）
  - thermal_annotations.json   （若在 val 根目录；官方有时为 index.json，请自行核对）

两种来源（二选一）：
  1) 本地 zip：--zip  path/to/file.zip
  2) Kaggle：  --kaggle  使用 kagglehub 下载完整数据集到缓存，再从缓存复制上述子集到 --out

Kaggle 需已配置 API（环境变量 KAGGLE_USERNAME / KAGGLE_KEY，或 ~/.kaggle/kaggle.json），并安装： pip install kagglehub
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile


def kaggle_hub_dataset_cache_dir(dataset_slug: str) -> str:
    """kagglehub 按 slug 存放缓存的目录，例如 ~/.cache/kagglehub/datasets/deepnewbie/flir-thermal-images-dataset"""
    slug = dataset_slug.replace("\\", "/").strip("/")
    parts = ["~", ".cache", "kagglehub", "datasets"] + slug.split("/")
    return os.path.expanduser(os.path.join(*parts))

# --- 路径判定（兼容 zip 根目录为 val/ 或 FLIR_ADAS_1_3/val/） ---


def is_target_zip_member(member: str) -> bool:
    name = member.replace("\\", "/").strip()
    if "/val/Annotated_thermal_8_bit/" in name or name.startswith("val/Annotated_thermal_8_bit/"):
        return True
    n = name.rstrip("/")
    if n.endswith("thermal_annotations.json"):
        return n.endswith("/val/thermal_annotations.json") or n.endswith("val/thermal_annotations.json")
    return False


def find_flir_root_with_val(data_path: str) -> str | None:
    """在下载目录下找到包含 val/Annotated_thermal_8_bit 的根目录。"""
    if os.path.isdir(os.path.join(data_path, "val", "Annotated_thermal_8_bit")):
        return data_path
    nested = os.path.join(data_path, "FLIR_ADAS_1_3")
    if os.path.isdir(os.path.join(nested, "val", "Annotated_thermal_8_bit")):
        return nested
    for root, dirs, _files in os.walk(data_path):
        if "val" in dirs and os.path.isdir(os.path.join(root, "val", "Annotated_thermal_8_bit")):
            return root
    return None


def copy_val_subset(flir_root: str, dest_root: str) -> tuple[int, bool]:
    """
    将 flir_root/val/Annotated_thermal_8_bit 与 val/thermal_annotations.json 复制到 dest_root，
    保持目标为 dest_root/val/...
    返回：(复制的图像文件数, 是否复制了 json)
    """
    val_src = os.path.join(flir_root, "val")
    anno_src = os.path.join(val_src, "Annotated_thermal_8_bit")
    json_src = os.path.join(val_src, "thermal_annotations.json")

    if not os.path.isdir(anno_src):
        raise FileNotFoundError(f"未找到目录: {anno_src}")

    val_dst = os.path.join(dest_root, "val")
    anno_dst = os.path.join(val_dst, "Annotated_thermal_8_bit")
    os.makedirs(anno_dst, exist_ok=True)

    n_images = 0
    for dirpath, _dirnames, filenames in os.walk(anno_src):
        rel = os.path.relpath(dirpath, anno_src)
        sub_dst = anno_dst if rel == "." else os.path.join(anno_dst, rel)
        os.makedirs(sub_dst, exist_ok=True)
        for fn in filenames:
            src_f = os.path.join(dirpath, fn)
            dst_f = os.path.join(sub_dst, fn)
            shutil.copy2(src_f, dst_f)
            n_images += 1

    copied_json = False
    if os.path.isfile(json_src):
        os.makedirs(val_dst, exist_ok=True)
        shutil.copy2(json_src, os.path.join(val_dst, "thermal_annotations.json"))
        copied_json = True

    return n_images, copied_json


def extract_zip_subset(zip_path: str, extract_to: str) -> int:
    os.makedirs(extract_to, exist_ok=True)
    n_files = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not is_target_zip_member(name):
                continue
            zf.extract(name, extract_to)
            if not name.endswith("/"):
                n_files += 1
    return n_files


def main() -> None:
    parser = argparse.ArgumentParser(description="只获取 FLIR val：Annotated_thermal_8_bit + thermal_annotations.json")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--zip", default=None, help="本地 FLIR zip 路径（与 --kaggle 二选一）")
    group.add_argument(
        "--kaggle",
        action="store_true",
        help="用 kagglehub 下载 deepnewbie/flir-thermal-images-dataset 后复制子集",
    )
    parser.add_argument(
        "--kaggle-dataset",
        default="deepnewbie/flir-thermal-images-dataset",
        help="Kaggle 数据集 slug（默认与官网示例一致）",
    )
    parser.add_argument(
        "--out",
        default="./my_flir_data",
        help="输出目录（得到 val/Annotated_thermal_8_bit/ 等）",
    )
    parser.add_argument(
        "--kaggle-clear-cache",
        action="store_true",
        help="下载前删除该数据集在本机的 kagglehub 缓存目录（用于 MD5 校验失败或中断后重下）",
    )
    args = parser.parse_args()

    extract_to = os.path.abspath(args.out)

    if args.kaggle:
        try:
            import kagglehub  # type: ignore[import-not-found]
            from kagglehub.exceptions import DataCorruptionError  # type: ignore[import-not-found]
        except ImportError:
            print("请先安装: pip install kagglehub", file=sys.stderr)
            sys.exit(1)

        cache_dir = kaggle_hub_dataset_cache_dir(args.kaggle_dataset)
        if args.kaggle_clear_cache and os.path.isdir(cache_dir):
            print("正在清除 kagglehub 缓存目录:", cache_dir)
            shutil.rmtree(cache_dir)

        print("正在从 Kaggle 下载/校验数据集（整包会进入本地缓存，首次较慢）…")
        try:
            cache_path = kagglehub.dataset_download(args.kaggle_dataset)
        except DataCorruptionError as e:
            print(
                f"\n下载内容与服务器声明的校验和不一致（多为网络中断或缓存不完整）：\n{e}\n\n"
                f"请删除该数据集的缓存后重试，例如 PowerShell：\n"
                f'  Remove-Item -Recurse -Force "{cache_dir}"\n'
                f"然后用本项目重新运行并建议加上：  --kaggle-clear-cache\n"
                f"（约 15GB，请尽量稳定网络、避免休眠/代理中断。）",
                file=sys.stderr,
            )
            sys.exit(1)
        print("Kaggle 缓存路径:", cache_path)

        flir_root = find_flir_root_with_val(cache_path)
        if not flir_root:
            print(
                f"错误：在 {cache_path} 下未找到 val/Annotated_thermal_8_bit。请检查数据集版本或目录结构。",
                file=sys.stderr,
            )
            sys.exit(1)

        n_img, has_json = copy_val_subset(flir_root, extract_to)
        print(f"已复制到: {extract_to}")
        print(f"Annotated_thermal_8_bit 中文件数: {n_img}")
        print(f"thermal_annotations.json: {'已复制' if has_json else '未找到（可能该版本用 index.json/coco.json，见 val 目录）'}")
        return

    zip_path = os.path.abspath(args.zip or "flir-thermal-dataset.zip")
    if not os.path.isfile(zip_path):
        print(
            f"错误：找不到压缩包: {zip_path}\n"
            "请用 --zip 指定本地 zip，或使用 --kaggle 从 Kaggle 拉取。",
            file=sys.stderr,
        )
        sys.exit(1)

    n_files = extract_zip_subset(zip_path, extract_to)
    print(f"解压完成。输出目录: {extract_to}")
    print(f"已解压文件数（不含纯目录项）: {n_files}")


if __name__ == "__main__":
    main()
