"""
CSMA 消融变体模块。

本模块定义 Ablation B-2 所需的变体类：
  - FrozenMeanRPCA: 以离线计算的训练集 RGB 特征均值（冻结 buffer）替代可学习原型，
                    用于对照"可学习原型库是否真正学到结构化 RGB 分布"。
  - CSMAMeanProto:  与 CSMA 等价结构，仅将 self.rpca 替换为 FrozenMeanRPCA。

设计原则：
  1. 复用 src/csma.py 的 IREncoder 和 PixelDecoder，不复制代码。
  2. FrozenMeanRPCA 的 forward() 与 RGBPrototypeCrossAttention.forward() 逻辑完全一致。
  3. 默认路径（src/csma.py::CSMA）不受任何影响。

对应 docs/实验实施细节.md §1.3（Ablation B-2）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import CSMAConfig
from src.csma import IREncoder, PixelDecoder


class FrozenMeanRPCA(nn.Module):
    """
    RGB 均值原型交叉注意力（Ablation B-2）。

    与 RGBPrototypeCrossAttention 结构完全相同，唯一差别：
      K/V 来自离线预计算的训练集 RGB 特征均值，注册为不参与梯度更新的 buffer，
      而非可学习的 nn.Parameter。

    这样对比 B-3（可学习原型），若 B-3 > B-2，证明原型库学到了结构化 RGB 分布，
    而非简单记忆了训练集均值。

    Args:
        mean_proto:    [proto_dim] 预计算的 RGB 特征均值向量，来自 compute_mean_proto.py。
        ir_dim:        红外特征通道数，与 proto_dim 相同（IRE 最终层输出）。
        proto_dim:     原型维度，默认 256。
        num_heads:     多头注意力头数。
        num_prototypes: 原型数量 K，与主实验一致（默认 512）。
        ffn_dim:       FFN 中间层维度，默认 1024。
    """

    def __init__(
        self,
        mean_proto: torch.Tensor,
        ir_dim: int,
        proto_dim: int,
        num_heads: int,
        num_prototypes: int,
        ffn_dim: int = 1024,
    ) -> None:
        super().__init__()
        if proto_dim % num_heads != 0:
            raise ValueError("proto_dim must be divisible by num_heads")
        if mean_proto.shape != (proto_dim,):
            raise ValueError(
                f"mean_proto shape 应为 ({proto_dim},)，实际为 {tuple(mean_proto.shape)}"
            )

        # 将均值向量广播为 K 个相同原型，注册为 buffer（不参与梯度更新）
        frozen_proto = mean_proto.unsqueeze(0).expand(num_prototypes, -1).clone()
        self.register_buffer("rgb_prototypes", frozen_proto)   # [K, proto_dim]

        self.q_proj = nn.Linear(ir_dim, proto_dim)
        self.cross_attn = nn.MultiheadAttention(proto_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(proto_dim)
        self.ffn = nn.Sequential(
            nn.Linear(proto_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, proto_dim),
        )
        self.norm2 = nn.LayerNorm(proto_dim)

    def forward(self, ir_feat: torch.Tensor) -> torch.Tensor:
        """
        与 RGBPrototypeCrossAttention.forward() 逻辑完全一致，仅 K/V 来源不同。

        Args:
            ir_feat: [B, L, ir_dim] 红外特征序列。

        Returns:
            [B, L, proto_dim] 经交叉注意力 + FFN 后的特征序列。
        """
        q = self.q_proj(ir_feat)
        B = ir_feat.shape[0]
        kv = self.rgb_prototypes.unsqueeze(0).expand(B, -1, -1)  # type: ignore[union-attr]
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        x = self.norm1(ir_feat + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class CSMAMeanProto(nn.Module):
    """
    CSMA 均值原型变体（Ablation B-2）。

    与 src/csma.py::CSMA 结构完全对等，仅将 self.rpca 替换为 FrozenMeanRPCA。
    IREncoder 和 PixelDecoder 直接复用 src.csma 中的实现，不重复定义。

    对外接口与 CSMA 完全一致：
      forward(x) -> pseudo_rgb [B, 3, H, W]
      get_intermediate_features(x) -> [B, L, proto_dim]（供 L_align 使用）

    Args:
        cfg:        CSMAConfig，结构超参与主实验完全相同。
        mean_proto: [proto_dim] 离线预计算的 RGB 特征均值，由 compute_mean_proto.py 生成。
    """

    def __init__(self, cfg: CSMAConfig, mean_proto: torch.Tensor) -> None:
        super().__init__()
        self.cfg = cfg
        last_ch = cfg.ir_enc_channels[-1]
        if last_ch != cfg.proto_dim:
            raise ValueError(
                f"ir_enc_channels[-1] ({last_ch}) must equal proto_dim ({cfg.proto_dim})"
            )

        self.ire = IREncoder(cfg.ir_enc_channels)
        self.rpca = FrozenMeanRPCA(
            mean_proto=mean_proto,
            ir_dim=last_ch,
            proto_dim=cfg.proto_dim,
            num_heads=cfg.num_cross_attn_heads,
            num_prototypes=cfg.num_rgb_prototypes,
        )
        self.pd = PixelDecoder(
            in_ch=cfg.proto_dim,
            skip2_ch=cfg.ir_enc_channels[2],
            skip1_ch=cfg.ir_enc_channels[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        与 CSMA.forward() 完全一致的接口。

        Args:
            x: [B, 3, H, W] 红外图像（ImageNet 归一化）。

        Returns:
            pseudo_rgb: [B, 3, H, W] 伪 RGB 图像。
        """
        c1, c2, c3 = self.ire(x)
        B, _, H8, W8 = c3.shape
        feat = c3.flatten(2).transpose(1, 2)
        feat = self.rpca(feat)
        feat_map = feat.transpose(1, 2).reshape(B, self.cfg.proto_dim, H8, W8)
        pseudo_rgb = self.pd(feat_map, c2, c1)
        if self.cfg.use_residual:
            pseudo_rgb = pseudo_rgb + x
        return pseudo_rgb

    def get_intermediate_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        RPCA 输出 token [B, L, proto_dim]，供 L_align 使用（与 CSMA 接口一致）。

        Args:
            x: [B, 3, H, W] 红外图像。

        Returns:
            [B, L, proto_dim]，L = (H/8)*(W/8)。
        """
        _, _, c3 = self.ire(x)
        feat = c3.flatten(2).transpose(1, 2)
        return self.rpca(feat)
