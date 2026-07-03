from __future__ import annotations

import torch
import torch.nn as nn

from src.config import CSMAConfig


def _make_norm(ch: int, use_group_norm: bool) -> nn.Module:
    if not use_group_norm:
        return nn.BatchNorm2d(ch)
    # Prefer many groups while keeping divisibility.
    for g in (32, 16, 8, 4, 2):
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class IREncoder(nn.Module):
    def __init__(self, channels: list[int], use_group_norm: bool = True) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("IREncoder expects ir_enc_channels of length 4")
        c0, c1, c2, c3 = channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, 3, 1, 1), _make_norm(c0, use_group_norm), nn.GELU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(c0, c1, 3, 2, 1), _make_norm(c1, use_group_norm), nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1), _make_norm(c2, use_group_norm), nn.GELU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, 2, 1), _make_norm(c3, use_group_norm), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        return c1, c2, c3


class RGBPrototypeCrossAttention(nn.Module):
    def __init__(self, ir_dim: int, proto_dim: int, num_heads: int, num_prototypes: int, ffn_dim: int = 1024) -> None:
        super().__init__()
        self.rgb_prototypes = nn.Parameter(torch.randn(num_prototypes, proto_dim) * 0.02)
        self.q_proj = nn.Linear(ir_dim, proto_dim)
        self.cross_attn = nn.MultiheadAttention(proto_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(proto_dim)
        self.ffn = nn.Sequential(nn.Linear(proto_dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, proto_dim))
        self.norm2 = nn.LayerNorm(proto_dim)

    def forward(self, ir_feat: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(ir_feat)
        kv = self.rgb_prototypes.unsqueeze(0).expand(ir_feat.shape[0], -1, -1)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        x = self.norm1(ir_feat + attn_out)
        return self.norm2(x + self.ffn(x))


class PixelDecoder(nn.Module):
    def __init__(
        self,
        in_ch: int = 256,
        skip2_ch: int = 128,
        skip1_ch: int = 64,
        head_out_ch: int = 32,
        use_group_norm: bool = True,
    ) -> None:
        super().__init__()
        self.up3 = nn.Sequential(nn.ConvTranspose2d(in_ch, 128, 4, 2, 1), _make_norm(128, use_group_norm), nn.GELU())
        self.fuse3 = nn.Conv2d(128 + skip2_ch, 128, 1)
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), _make_norm(64, use_group_norm), nn.GELU())
        self.fuse2 = nn.Conv2d(64 + skip1_ch, 64, 1)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(64, head_out_ch, 4, 2, 1), _make_norm(head_out_ch, use_group_norm), nn.GELU())
        self.head = nn.Conv2d(head_out_ch, 3, 1)
        nn.init.constant_(self.head.weight, 1e-4)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.0)

    def forward(self, feat: torch.Tensor, skip_c2: torch.Tensor, skip_c1: torch.Tensor) -> torch.Tensor:
        x = self.fuse3(torch.cat([self.up3(feat), skip_c2], dim=1))
        x = self.fuse2(torch.cat([self.up2(x), skip_c1], dim=1))
        return torch.tanh(self.head(self.up1(x)))


class CSMA(nn.Module):
    def __init__(self, cfg: CSMAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.ir_enc_channels[-1] != cfg.proto_dim:
            raise ValueError("ir_enc_channels[-1] must equal proto_dim")
        self.ire = IREncoder(cfg.ir_enc_channels, use_group_norm=cfg.use_group_norm)
        self.rpca = RGBPrototypeCrossAttention(
            ir_dim=cfg.proto_dim, proto_dim=cfg.proto_dim,
            num_heads=cfg.num_cross_attn_heads, num_prototypes=cfg.num_rgb_prototypes,
        )
        self.pd = PixelDecoder(
            in_ch=cfg.proto_dim,
            skip2_ch=cfg.ir_enc_channels[2],
            skip1_ch=cfg.ir_enc_channels[1],
            use_group_norm=cfg.use_group_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1, c2, c3 = self.ire(x)
        B, _, H8, W8 = c3.shape
        feat = self.rpca(c3.flatten(2).transpose(1, 2))
        feat_map = feat.transpose(1, 2).reshape(B, self.cfg.proto_dim, H8, W8)
        residual = self.pd(feat_map, c2, c1)
        pseudo_rgb = residual
        if self.cfg.use_residual:
            # Keep pseudo image close to input IR to avoid destabilizing frozen OWLv2.
            pseudo_rgb = x + self.cfg.residual_scale * residual
        return torch.clamp(pseudo_rgb, -self.cfg.pseudo_rgb_clamp, self.cfg.pseudo_rgb_clamp)
