"""
训练用指数滑动平均（EMA）权重，用于 Final Model 短训选优。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ModelEMA:
    """
    维护适配器权重的 EMA 副本；decay 越接近 1 越平滑。
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay 须在 (0,1)，当前 {decay}")
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    def update(self, model: nn.Module) -> None:
        """用当前 model 权重更新 shadow；非浮点 buffer（如 BN 计数）直接同步。"""
        for key, val in model.state_dict().items():
            if not val.is_floating_point():
                self.shadow[key] = val.detach().clone()
                continue
            self.shadow[key].mul_(self.decay).add_(val.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """返回 EMA 权重副本。"""
        return {k: v.clone() for k, v in self.shadow.items()}

    def copy_to(self, model: nn.Module) -> None:
        """将 EMA 权重载入 model（用于保存/评测）。"""
        model.load_state_dict(self.shadow, strict=True)
