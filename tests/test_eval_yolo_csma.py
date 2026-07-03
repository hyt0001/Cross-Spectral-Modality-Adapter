"""Unit tests for eval_yolo_csma helpers."""

from __future__ import annotations

import unittest

import torch

from src.eval_yolo_csma import YOLO_CLS_TO_EVAL_CAT, _valid_hw


class TestEvalYoloHelpers(unittest.TestCase):
    def test_yolo_cls_mapping(self) -> None:
        self.assertEqual(YOLO_CLS_TO_EVAL_CAT[0], 1)
        self.assertEqual(YOLO_CLS_TO_EVAL_CAT[2], 2)

    def test_valid_hw_from_mask(self) -> None:
        pm = torch.zeros(128, 160, dtype=torch.int64)
        pm[:80, :100] = 1
        self.assertEqual(_valid_hw(pm), (80, 100))


if __name__ == "__main__":
    unittest.main()
