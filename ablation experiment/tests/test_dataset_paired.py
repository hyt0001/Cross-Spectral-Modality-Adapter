"""
FlirPairedDataset 单元 smoke 测试。

使用 tmp_path 构造最小 COCO JSON + 临时 IR/RGB 图像，无需真实数据集。
运行命令：
    CUDA_VISIBLE_DEVICES=0 conda run -n RGBtest python -m pytest tests/test_dataset_paired.py -v
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from PIL import Image


# ── 辅助函数 ─────────────────────────────────────────────────────────────────

def _make_coco_dir(root: Path, filenames: list[str]) -> None:
    """在 root 下创建最小 COCO 目录（图像文件 + _annotations.coco.json）。"""
    root.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    for i, fname in enumerate(filenames):
        img_id = i + 1
        (root / fname).write_bytes(b"")   # 占位文件，后续 mock Image.open
        images.append({"id": img_id, "file_name": fname, "width": 640, "height": 512})
        annotations.append({
            "id": img_id,
            "image_id": img_id,
            "category_id": 2,
            "bbox": [10.0, 10.0, 50.0, 80.0],
            "area": 4000.0,
            "iscrowd": 0,
        })
    coco = {"images": images, "annotations": annotations, "categories": [
        {"id": 1, "name": "car"},
        {"id": 2, "name": "person"},
    ]}
    (root / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")


def _make_fake_processor(h: int = 64, w: int = 64) -> MagicMock:
    """返回一个模拟 processor，其 image_processor 返回固定形状张量。"""
    proc = MagicMock()

    def _image_processor(images, annotations=None, return_tensors=None):
        out = MagicMock()
        out.__getitem__ = lambda self, key: {
            "pixel_values": torch.zeros(1, 3, h, w),
            "pixel_mask":   torch.ones(1, h, w, dtype=torch.long),
            "labels": [{
                "class_labels": torch.zeros(1, dtype=torch.long),
                "boxes":        torch.zeros(1, 4),
                "area":         torch.zeros(1),
                "iscrowd":      torch.zeros(1, dtype=torch.long),
                "orig_size":    torch.tensor([512, 640]),
            }],
        }[key]
        return out

    proc.image_processor.side_effect = _image_processor
    return proc


# ── 测试类 ────────────────────────────────────────────────────────────────────

class TestFlirPairedDatasetGetitem(unittest.TestCase):
    """__getitem__ 的 rgb 有无两种情况。"""

    def setUp(self) -> None:
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self._ir_root = Path(self._tmpdir) / "ir"
        self._rgb_root = Path(self._tmpdir) / "rgb"
        _make_coco_dir(self._ir_root, ["img_001.jpg", "img_002.jpg"])
        self._processor = _make_fake_processor()
        self._cat_map = {1: 1, 2: 0}

    def _make_dataset(self, rgb_root: str | Path):
        from src.dataset_paired import FlirPairedDataset
        with patch("PIL.Image.open", return_value=MagicMock(convert=lambda m: Image.new("RGB", (64, 64)))):
            ds = FlirPairedDataset(
                ir_root=str(self._ir_root),
                rgb_root=str(rgb_root),
                processor=self._processor,
                text_prompt="person. car.",
                coco_category_id_to_class_idx=self._cat_map,
            )
        return ds

    def test_getitem_with_rgb(self) -> None:
        """RGB 文件存在时，rgb_pixel_values 应为 [3, H, W] Tensor。"""
        self._rgb_root.mkdir(parents=True, exist_ok=True)
        (self._rgb_root / "img_001.jpg").write_bytes(b"")

        ds = self._make_dataset(self._rgb_root)
        with patch("PIL.Image.open", return_value=MagicMock(convert=lambda m: Image.new("RGB", (64, 64)))):
            sample = ds[0]

        self.assertIsNotNone(sample["rgb_pixel_values"])
        self.assertEqual(sample["rgb_pixel_values"].dim(), 3)

    def test_getitem_without_rgb(self) -> None:
        """RGB 文件不存在时，rgb_pixel_values 应为 None。"""
        empty_rgb = Path(self._tmpdir) / "empty_rgb"
        empty_rgb.mkdir(parents=True, exist_ok=True)

        ds = self._make_dataset(empty_rgb)
        with patch("PIL.Image.open", return_value=MagicMock(convert=lambda m: Image.new("RGB", (64, 64)))):
            sample = ds[0]

        self.assertIsNone(sample["rgb_pixel_values"])


class TestColllatePaired(unittest.TestCase):
    """collate_paired 的三种情况：全有 / 部分有 / 全无。"""

    def _make_sample(self, has_rgb: bool, h: int = 64, w: int = 64) -> dict:
        """构造最小 batch 样本字典。"""
        return {
            "pixel_values":  torch.zeros(3, h, w),
            "pixel_mask":    torch.ones(h, w, dtype=torch.long),
            "labels": {
                "class_labels": torch.zeros(1, dtype=torch.long),
                "boxes":        torch.zeros(1, 4),
                "area":         torch.zeros(1),
                "iscrowd":      torch.zeros(1, dtype=torch.long),
                "orig_size":    torch.tensor([512, 640]),
            },
            "image_path":   "/fake/path.jpg",
            "orig_size":    torch.tensor([512, 640], dtype=torch.int64),
            "rgb_pixel_values": torch.zeros(3, h, w) if has_rgb else None,
        }

    def test_collate_all_rgb(self) -> None:
        """全 batch 有 RGB → 输出含 rgb_pixel_values，形状 [B,3,H,W]。"""
        from src.dataset_paired import collate_paired
        batch = [self._make_sample(True), self._make_sample(True)]
        out = collate_paired(batch)
        self.assertIn("rgb_pixel_values", out)
        self.assertEqual(tuple(out["rgb_pixel_values"].shape), (2, 3, 64, 64))

    def test_collate_partial_rgb(self) -> None:
        """部分有 RGB → 输出不含 rgb_pixel_values。"""
        from src.dataset_paired import collate_paired
        batch = [self._make_sample(True), self._make_sample(False)]
        out = collate_paired(batch)
        self.assertNotIn("rgb_pixel_values", out)

    def test_collate_no_rgb(self) -> None:
        """全无 RGB → 输出不含 rgb_pixel_values。"""
        from src.dataset_paired import collate_paired
        batch = [self._make_sample(False), self._make_sample(False)]
        out = collate_paired(batch)
        self.assertNotIn("rgb_pixel_values", out)

    def test_collate_ir_fields_present(self) -> None:
        """collate_paired 不论 RGB 情况，IR 基础字段均须完整。"""
        from src.dataset_paired import collate_paired
        batch = [self._make_sample(False)]
        out = collate_paired(batch)
        for key in ("pixel_values", "pixel_mask", "labels", "image_paths", "orig_sizes"):
            self.assertIn(key, out, f"缺少键: {key}")


if __name__ == "__main__":
    unittest.main()
