"""
特征级跨光谱模态适配器（FeatureAdapter）。

在 DINO encoder 入口的 patch token 空间 [B, L, D] 做残差 MLP 映射，
替代像素级 CSMA 的 IR→伪RGB 翻译路径。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import CSMAConfig


class FeatureAdapter(nn.Module):
    """
    残差 MLP 适配器：out = x + MLP(x)。

    插入点：Grounding DINO `input_proj` 之后、encoder 之前的 `vision_features`。

    Args:
        cfg: CSMAConfig，读取 fa_hidden_dim / fa_num_layers / fa_use_residual / proto_dim。
    """

    def __init__(self, cfg: CSMAConfig) -> None:
        super().__init__()
        self._cfg = cfg
        d_model = cfg.proto_dim
        hidden = cfg.fa_hidden_dim
        n_layers = cfg.fa_num_layers
        if n_layers < 2:
            raise ValueError("fa_num_layers 须 >= 2")

        layers: list[nn.Module] = []
        in_dim = d_model
        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            ])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, d_model))
        self.mlp = nn.Sequential(*layers)
        self.use_residual = cfg.fa_use_residual
        if cfg.fa_zero_init:
            self._zero_init_last_linear()

    def _zero_init_last_linear(self) -> None:
        """将 MLP 最后一层置零，残差路径初期为恒等映射 out≈x。"""
        last_linear: nn.Linear | None = None
        for mod in self.mlp.modules():
            if isinstance(mod, nn.Linear):
                last_linear = mod
        if last_linear is None:
            raise RuntimeError("FeatureAdapter MLP 中未找到 Linear 层")
        nn.init.zeros_(last_linear.weight)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)

    @torch.no_grad()
    def identity_delta_norm(self, tokens: torch.Tensor) -> float:
        """诊断：|MLP(x)| 均值，零初始化后应接近 0。"""
        return float(self.mlp(tokens).abs().mean().cpu())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, L, D] DINO 多尺度 flatten 后的视觉 token。

        Returns:
            同形状适配后 token。
        """
        assert tokens.dim() == 3, f"tokens 须为 [B,L,D]，收到 {tokens.shape}"
        delta = self.mlp(tokens)
        if self.use_residual:
            return tokens + delta
        return delta

    def count_parameters(self) -> int:
        """返回可训练参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
