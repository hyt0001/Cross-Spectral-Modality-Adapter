"""
轻量级残差翻译网络：在 processor 归一化后的张量上学习小残差，近似恒等起步。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualTranslator(nn.Module):
    """
    将 3 通道（红外复制或伪 RGB）映射为同分辨率残差并加回原图。

    末层权重极小初始化，使训练初期接近恒等映射，便于过拟合与梯度稳定。
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )
        nn.init.constant_(self.conv_block[-1].weight, 1e-4)
        if self.conv_block[-1].bias is not None:
            nn.init.constant_(self.conv_block[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]，与 Grounding DINO processor 输出 pixel_values 同分布。

        Returns:
            pseudo_rgb: [B, 3, H, W]
        """
        noise = self.conv_block(x)
        return x + noise
