"""
Cross-Spectral Modality Adapter (CSMA): IRE + RGB prototype cross-attention + pixel decoder.

Specs: docs/TD.md §1.2, docs/architecture.md §3–6.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import CSMAConfig


class IREncoder(nn.Module):
    """
    Multi-scale IR pyramid: stem + three stride-2 stages.
    Output spatial grid H/8 × W/8 at final channels (default 256).
    """

    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("IREncoder expects ir_enc_channels of length 4 [stem, L1, L2, L3]")
        c0, c1, c2, c3 = channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c0),
            nn.GELU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(c0, c1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        return c1, c2, c3


class RGBPrototypeCrossAttention(nn.Module):
    """
    Cross-attention: Q from IR tokens, K/V from learnable RGB prototypes (no RGB image at inference).
    Transformer-style block with FFN per docs/architecture.md §4.
    """

    def __init__(
        self,
        ir_dim: int,
        proto_dim: int,
        num_heads: int,
        num_prototypes: int,
        ffn_dim: int = 1024,
    ) -> None:
        super().__init__()
        if proto_dim % num_heads != 0:
            raise ValueError("proto_dim must be divisible by num_heads")

        self.rgb_prototypes = nn.Parameter(torch.randn(num_prototypes, proto_dim) * 0.02)
        self.q_proj = nn.Linear(ir_dim, proto_dim)
        self.cross_attn = nn.MultiheadAttention(
            proto_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(proto_dim)
        self.ffn = nn.Sequential(
            nn.Linear(proto_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, proto_dim),
        )
        self.norm2 = nn.LayerNorm(proto_dim)

    def forward(self, ir_feat: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(ir_feat)
        B = ir_feat.shape[0]
        kv = self.rgb_prototypes.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        x = self.norm1(ir_feat + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class PixelDecoder(nn.Module):
    """
    Upsample RPCA map H/8 → H with skip fusion from IRE (c2, c1).
    """

    def __init__(
        self,
        in_ch: int = 256,
        skip2_ch: int = 128,
        skip1_ch: int = 64,
        head_out_ch: int = 32,
    ) -> None:
        super().__init__()
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.fuse3 = nn.Conv2d(128 + skip2_ch, 128, kernel_size=1)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.fuse2 = nn.Conv2d(64 + skip1_ch, 64, kernel_size=1)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, head_out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(head_out_ch),
            nn.GELU(),
        )
        self.head = nn.Conv2d(head_out_ch, 3, kernel_size=1)
        nn.init.constant_(self.head.weight, 1e-4)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.0)

    def forward(
        self,
        feat: torch.Tensor,
        skip_c2: torch.Tensor,
        skip_c1: torch.Tensor,
    ) -> torch.Tensor:
        # 奇数输入尺寸时 ConvTranspose2d 输出可能比 skip 大 1，裁到 skip 的空间尺寸
        x = self.up3(feat)
        if x.shape[-2:] != skip_c2.shape[-2:]:
            x = torch.nn.functional.interpolate(
                x, size=skip_c2.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip_c2], dim=1)
        x = self.fuse3(x)
        x = self.up2(x)
        if x.shape[-2:] != skip_c1.shape[-2:]:
            x = torch.nn.functional.interpolate(
                x, size=skip_c1.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip_c1], dim=1)
        x = self.fuse2(x)
        x = self.up1(x)
        return torch.tanh(self.head(x))


class CSMA(nn.Module):
    """
    Plug-in adapter: pseudo_rgb = csma(ir_pixel_values), same contract as ResidualTranslator.
    """

    def __init__(self, cfg: CSMAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        last_ch = cfg.ir_enc_channels[-1]
        if last_ch != cfg.proto_dim:
            raise ValueError(
                f"ir_enc_channels[-1] ({last_ch}) must equal proto_dim ({cfg.proto_dim})"
            )

        self.ire = IREncoder(cfg.ir_enc_channels)
        self.rpca = RGBPrototypeCrossAttention(
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
        c1, c2, c3 = self.ire(x)
        B, _, H8, W8 = c3.shape
        feat = c3.flatten(2).transpose(1, 2)
        feat = self.rpca(feat)
        feat_map = feat.transpose(1, 2).reshape(B, self.cfg.proto_dim, H8, W8)
        pseudo_delta = self.pd(feat_map, c2, c1)
        # ConvTranspose2d 在奇数输入尺寸时可能多出 1 pixel，裁回原始 H/W
        if pseudo_delta.shape[-2:] != x.shape[-2:]:
            pseudo_delta = torch.nn.functional.interpolate(
                pseudo_delta, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        if self.cfg.use_residual:
            pseudo_rgb = x + self.cfg.residual_scale * pseudo_delta
        else:
            pseudo_rgb = pseudo_delta
        if self.cfg.pseudo_clamp > 0.0:
            lim = float(self.cfg.pseudo_clamp)
            pseudo_rgb = pseudo_rgb.clamp(-lim, lim)
        return pseudo_rgb

    def get_intermediate_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        RPCA output tokens [B, L, proto_dim] for L_align (before PD and without input residual).
        L = (H/8)*(W/8).
        """
        _, _, c3 = self.ire(x)
        feat = c3.flatten(2).transpose(1, 2)
        return self.rpca(feat)
