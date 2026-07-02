"""FeatureAdapter 单元测试。"""

from __future__ import annotations

import torch

from src.config import CSMAConfig
from src.feature_adapter import FeatureAdapter


def test_feature_adapter_shape_and_residual() -> None:
    cfg = CSMAConfig(adapter_mode="feature", lambda_recon=0.0, align_layer_indices=[])
    fa = FeatureAdapter(cfg)
    x = torch.randn(2, 100, cfg.proto_dim, requires_grad=True)
    y = fa(x)
    assert y.shape == x.shape
    loss = y.sum()
    loss.backward()
    assert x.grad is not None


def test_feature_adapter_zero_init_near_identity() -> None:
    cfg = CSMAConfig(
        adapter_mode="feature",
        lambda_recon=0.0,
        align_layer_indices=[],
        fa_zero_init=True,
    )
    fa = FeatureAdapter(cfg)
    x = torch.randn(2, 16, cfg.proto_dim)
    y = fa(x)
    assert torch.allclose(x, y, atol=1e-6)
    assert fa.identity_delta_norm(x) < 1e-6


def test_feature_adapter_param_count() -> None:
    cfg = CSMAConfig(
        adapter_mode="feature",
        lambda_recon=0.0,
        align_layer_indices=[],
        fa_zero_init=False,
    )
    fa = FeatureAdapter(cfg)
    n = fa.count_parameters()
    assert 400_000 < n < 700_000
