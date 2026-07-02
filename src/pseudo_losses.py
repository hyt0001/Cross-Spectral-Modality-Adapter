"""
CSMA pseudo-RGB 辅助损失（对齐 OWL Final Model 配置说明）。

- identity：pseudo 贴近 IR（抑制 car 大亮块假框）
- total variation：pseudo 空间平滑
- logit 正则：压低检测置信度，减少满屏框
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def identity_loss(pseudo_rgb: torch.Tensor, ir_rgb: torch.Tensor) -> torch.Tensor:
    """
    L_id = MSE(pseudo, IR)，IR 与 pseudo 同形状 [B,3,H,W]。
    """
    if pseudo_rgb.shape != ir_rgb.shape:
        raise ValueError(
            f"identity_loss 形状不一致: pseudo {pseudo_rgb.shape} vs ir {ir_rgb.shape}"
        )
    return F.mse_loss(pseudo_rgb, ir_rgb.detach())


def total_variation_loss(pseudo_rgb: torch.Tensor) -> torch.Tensor:
    """
    各向同性 TV：相邻像素差分 L1 均值。
    """
    if pseudo_rgb.dim() != 4:
        raise ValueError(f"total_variation_loss 期望 4D 张量，得到 {pseudo_rgb.dim()}D")
    dh = (pseudo_rgb[:, :, 1:, :] - pseudo_rgb[:, :, :-1, :]).abs().mean()
    dw = (pseudo_rgb[:, :, :, 1:] - pseudo_rgb[:, :, :, :-1]).abs().mean()
    return dh + dw


def logit_regularization(outputs: Any, device: torch.device) -> torch.Tensor:
    """
    对检测 logits 的平均 sigmoid 施加惩罚，降低过度检测。

    训练态 forward 若无 logits 则返回 0（不参与反传）。
    """
    logits = getattr(outputs, "logits", None)
    if logits is None:
        return torch.tensor(0.0, device=device)
    return logits.sigmoid().mean()
