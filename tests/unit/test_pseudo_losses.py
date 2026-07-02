"""pseudo 损失与 CSMA 前向约束单元测试。"""

from __future__ import annotations

import torch

from src.config import CSMAConfig
from src.csma import CSMA
from src.pseudo_losses import identity_loss, total_variation_loss


def test_identity_loss_zero_when_equal() -> None:
    x = torch.randn(2, 3, 32, 32)
    loss = identity_loss(x, x)
    assert loss.item() < 1e-6


def test_total_variation_non_negative() -> None:
    x = torch.randn(1, 3, 16, 16)
    tv = total_variation_loss(x)
    assert tv.item() >= 0.0


def test_csma_residual_scale_and_clamp() -> None:
    cfg = CSMAConfig(
        use_residual=True,
        residual_scale=0.05,
        pseudo_clamp=2.0,
    )
    model = CSMA(cfg)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape == x.shape
    assert out.abs().max().item() <= 2.0 + 1e-5
