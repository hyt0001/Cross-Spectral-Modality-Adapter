"""Unit tests for YOLO training helpers."""
from __future__ import annotations

import unittest

import torch

from src.yolo_utils import FLIR_CLASS_IDX_TO_YOLO, labels_to_yolo_targets


class TestYoloUtils(unittest.TestCase):
    def test_class_mapping(self) -> None:
        self.assertEqual(FLIR_CLASS_IDX_TO_YOLO[0], 0)
        self.assertEqual(FLIR_CLASS_IDX_TO_YOLO[1], 2)

    def test_labels_to_yolo_targets(self) -> None:
        device = torch.device("cpu")
        labels = [
            {
                "class_labels": torch.tensor([0, 1]),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2]]),
            },
            {
                "class_labels": torch.tensor([1]),
                "boxes": torch.tensor([[0.6, 0.6, 0.15, 0.25]]),
            },
        ]
        t = labels_to_yolo_targets(labels, device)
        self.assertEqual(t["batch_idx"].tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(t["cls"].tolist(), [0.0, 2.0, 2.0])
        self.assertEqual(t["bboxes"].shape, (3, 4))

    def test_empty_labels(self) -> None:
        t = labels_to_yolo_targets(
            [{"class_labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)}],
            torch.device("cpu"),
        )
        self.assertEqual(t["bboxes"].shape[0], 0)


if __name__ == "__main__":
    unittest.main()
