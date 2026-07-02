"""
M3FD collate 单元测试：验证多分辨率 batch 可正确 pad+stack。

运行命令：
    CUDA_VISIBLE_DEVICES=0 conda run -n RGBtest python -m pytest tests/test_dataset_m3fd.py -v
"""

from __future__ import annotations

import unittest

import torch

from src.dataset_m3fd import (
    _pad_stack_tensors,
    build_m3fd_category_map,
    build_m3fd_category_map_for_training,
    build_m3fd_eval_categories,
    collate_m3fd,
)


def _make_sample(
    h: int,
    w: int,
    has_rgb: bool = False,
) -> dict:
    """构造最小 M3FD batch 样本。"""
    return {
        "pixel_values": torch.randn(3, h, w),
        "pixel_mask": torch.ones(h, w, dtype=torch.long),
        "labels": {
            "class_labels": torch.zeros(1, dtype=torch.long),
            "boxes": torch.zeros(1, 4),
            "area": torch.zeros(1),
            "iscrowd": torch.zeros(1, dtype=torch.long),
            "orig_size": torch.tensor([h, w]),
        },
        "image_path": f"/fake/{h}x{w}.png",
        "rgb_path": f"/fake/rgb_{h}x{w}.png" if has_rgb else None,
        "rgb_pixel_values": torch.randn(3, h, w) if has_rgb else None,
    }


class TestPadStackTensors(unittest.TestCase):
    """_pad_stack_tensors 形状与 mask 行为。"""

    def test_pad_stack_mixed_hw(self) -> None:
        """不同 H/W 的 [3,H,W] 张量应 pad 到 batch 最大尺寸。"""
        tensors = [
            torch.ones(3, 512, 682),
            torch.ones(3, 512, 853),
            torch.ones(3, 480, 640),
        ]
        out = _pad_stack_tensors(tensors, pad_value=0.0)
        self.assertEqual(tuple(out.shape), (3, 3, 512, 853))

    def test_pad_stack_2d_mask(self) -> None:
        """[H,W] mask pad 后 padding 区域应为 0。"""
        tensors = [
            torch.ones(512, 682),
            torch.ones(512, 853),
        ]
        out = _pad_stack_tensors(tensors, pad_value=0.0)
        self.assertEqual(tuple(out.shape), (2, 512, 853))
        self.assertEqual(out[0, 0, 682:].sum().item(), 0.0)
        self.assertEqual(out[0, :, 682:].sum().item(), 0.0)


class TestM3fdTwoClassPrompt(unittest.TestCase):
    """两类 prompt 动态映射。"""

    def test_two_class_training_map(self) -> None:
        cat_map, valid = build_m3fd_category_map_for_training("person. car.")
        self.assertEqual(cat_map[5], 0)
        self.assertEqual(cat_map[2], 1)
        self.assertEqual(valid, frozenset({5, 2}))

    def test_two_class_eval_map(self) -> None:
        cat_map, valid = build_m3fd_category_map("person. car.")
        self.assertEqual(cat_map[5], 1)
        self.assertEqual(cat_map[2], 2)
        cats = build_m3fd_eval_categories("person. car.")
        self.assertEqual([c["name"] for c in cats], ["person", "car"])


class TestCollateM3fd(unittest.TestCase):
    """collate_m3fd 多分辨率 batch。"""

    def test_collate_mixed_sizes(self) -> None:
        """M3FD 典型混合分辨率 batch 应成功 stack。"""
        batch = [
            _make_sample(512, 682),
            _make_sample(512, 853),
            _make_sample(480, 640),
        ]
        out = collate_m3fd(batch)
        self.assertEqual(tuple(out["pixel_values"].shape), (3, 3, 512, 853))
        self.assertEqual(tuple(out["pixel_mask"].shape), (3, 512, 853))
        self.assertEqual(len(out["labels"]), 3)

    def test_collate_with_rgb(self) -> None:
        """全 batch 有 RGB 时 rgb_pixel_values 与 IR 同形。"""
        batch = [
            _make_sample(512, 682, has_rgb=True),
            _make_sample(512, 853, has_rgb=True),
        ]
        out = collate_m3fd(batch)
        self.assertIn("rgb_pixel_values", out)
        self.assertEqual(
            tuple(out["rgb_pixel_values"].shape),
            tuple(out["pixel_values"].shape),
        )


if __name__ == "__main__":
    unittest.main()
