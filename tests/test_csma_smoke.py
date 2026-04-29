"""Smoke tests for CSMA shapes and numerical sanity."""

from __future__ import annotations

import unittest

import torch

from src.config import CSMAConfig
from src.csma import CSMA


class TestCSMASmoke(unittest.TestCase):
    def test_forward_matches_input_spatial_shape(self) -> None:
        cfg = CSMAConfig()
        m = CSMA(cfg)
        h, w = 224, 224
        x = torch.randn(2, 3, h, w)
        y = m(x)
        self.assertEqual(tuple(y.shape), (2, 3, h, w))
        self.assertFalse(torch.isnan(y).any())

    def test_get_intermediate_features_token_count(self) -> None:
        cfg = CSMAConfig()
        m = CSMA(cfg)
        h, w = 224, 224
        x = torch.randn(1, 3, h, w)
        z = m.get_intermediate_features(x)
        l_expected = (h // 8) * (w // 8)
        self.assertEqual(z.shape, (1, l_expected, cfg.proto_dim))
        self.assertFalse(torch.isnan(z).any())

    def test_use_residual_false_no_add(self) -> None:
        cfg = CSMAConfig.from_overrides({"use_residual": False})
        m = CSMA(cfg)
        x = torch.zeros(1, 3, 64, 64)
        y = m(x)
        self.assertEqual(tuple(y.shape), (1, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
