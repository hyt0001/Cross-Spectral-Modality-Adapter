# CSMA 技术设计文档（TD）

> **论文工作线**：*"与其花费巨大算力从头预训练多光谱基础模型（如 M-SpecGene），不如利用其 GMM-CMSS 物理信息挖掘策略训练一个极轻量的即插即用模态适配器。实验证明：通用视觉大模型（Grounding DINO）+ 轻量适配器 ≥ 领域专有预训练大模型（M-SpecGene）。"*

---

## 技术决策

本节记录影响整体方案走向的关键技术抉择，每项决策均附备选方案及取舍依据。

### TD-01：适配器的作用位置——像素级 vs 特征级

| 方案 | 描述 | 优点 | 缺点 |
|:---|:---|:---|:---|
| **像素级（采用）** | 适配器将红外图像变换为伪 RGB 图像，再送入冻结 DINO | 与 DINO 内部结构完全解耦，现有训练代码零改动 | 变换目标模糊，需依赖蒸馏损失约束语义对齐 |
| 特征级 | 在 DINO 骨干网络内部插入适配器模块（如 Adapter Tuning） | 直接作用于语义特征空间 | 需修改 HuggingFace 模型内部，接口复杂，不具备普适即插即用性 |

**决策**：采用像素级方案。接口为 `pseudo_rgb = csma(pixel_values)`，与现有 `src/train_demo.py` 第 243 行调用方式完全兼容，不修改 Grounding DINO 任何代码。

---

### TD-02：小模型骨干——CNN 金字塔 vs 轻量 ViT

| 方案 | 参数量 | 感受野 | 与 CMSS Patch 对齐 | 训练稳定性 |
|:---|:---:|:---:|:---:|:---:|
| **CNN 金字塔（采用）** | ~0.3M | 局部，多尺度 | 可通过 stride 控制 Patch 大小 | 高，BatchNorm 稳定 |
| 轻量 ViT（如 DeiT-Tiny） | ~5M | 全局 | 天然 Patch 对齐 | 中，小数据集易过拟合 |
| MobileNetV3 | ~2.5M | 局部 | 需额外对齐 | 高 |

**决策**：采用三层步进卷积金字塔（IRE），参数量约 0.3M，stride=2 每层使输出分辨率与 DINO `PatchEmbed`（patch_size=16）的多尺度特征对齐。训练数据量小（~10K），ViT 的全局注意力易过拟合。

---

### TD-03：跨模态融合机制——可学习原型库 vs 运行时双流

| 方案 | 描述 | 推理时是否需要 RGB | 参数量 |
|:---|:---|:---:|:---:|
| **可学习 RGB 原型库（采用）** | K=512 个 nn.Parameter Token 作为 K/V，红外特征作为 Q | **不需要** | ~1.0M |
| 运行时双流 | 同时输入 RGB+IR，实时做交叉注意力（M-SpecGene 解码器方案） | **需要** | ~0M（额外参数） |

**决策**：采用可学习 RGB 原型库。这是本文的核心创新——将 M-SpecGene `CrossTransformerEncoderLayer` 中运行时依赖双流的 K/V 替换为离线学习的原型 Token，推理时将跨模态知识"内化"进模型权重，实现真正的单流红外推理。

---

### TD-04：DINO 参数策略——完全冻结 vs LoRA 微调

| 方案 | 优点 | 缺点 |
|:---|:---|:---|
| **完全冻结（采用）** | 保留全部零样本能力，训练稳定，梯度计算开销小 | DINO 内部无法自适应红外分布 |
| LoRA 微调（1% 参数） | 轻微适应红外分布 | 可能损害开放词汇能力，本文核心论点被削弱 |
| 全量微调 | 最高上限精度 | 灾难性遗忘，失去零样本能力，与论文故事线矛盾 |

**决策**：完全冻结。梯度检查逻辑已在 `src/train_demo.py` 第 271–275 行实现，沿用。

---

### TD-05：CMSS 高低值的物理含义——以代码为准

参考 `M-SpecGene-main/pretrain/mmpretrain-main_rgbt/mmpretrain/models/selfsup/GMM_CMSS_SAMPLE.py` 第 85–97 行：

```
CMSS = sqrt((cosine_sim + 1) / 2) / (var_rgb * var_ir)，再全局 max-归一化
```

- **低 CMSS（→ 0）**：高方差且跨模态差异大 → **目标核心区域**（行人、车辆热源）
- **高 CMSS（→ 1）**：低方差且跨模态一致 → **平滑背景**（天空、路面）

> 注意：初始参考计划 `plan_for_CSMM_adapter_initial.md` 中对高低值的描述与此相反，本文档以 M-SpecGene 原始代码为准。

**GMM 拟合后**：3 个均值 μ₁ < μ₂ < μ₃，分别对应目标核心、边缘过渡、背景。

---

### TD-06：L_align 特征提取点

**决策**：在 DINO 视觉编码器骨干网络输出端、经 `input_proj` 投影后的多尺度特征图（`hidden_dim=256`）上计算 MSE 对齐损失。理由：
- 该位置已将多通道图像压缩为语义丰富的 256 维 Token
- 高于像素级（太低层）且低于 Transformer 解码器（太高层，解耦 RGB-IR 差异更困难）
- 对应 HuggingFace `GroundingDinoForObjectDetection` 中 `model.backbone` 输出后、送入 `model.encoder` 前的 `projected_features`

---

---

## 目录

1. [模块设计](#1-模块设计)
   - 1.1 [配置管理（Config）](#11-配置管理config)
   - 1.2 [`src/csma.py` — 核心适配器模型](#12-srccsmapy--核心适配器模型)
   - 1.3 [`src/cmss_utils.py` — CMSS 计算与 GMM 工具](#13-srccmss_utilspy--cmss-计算与-gmm-工具)
   - 1.4 [`src/dataset_paired.py` — RGB-IR 配对数据集](#14-srcdataset_pairedpy--rgb-ir-配对数据集)
   - 1.5 [`src/train_csma.py` — CSMA 训练主程序](#15-srctrain_csmapy--csma-训练主程序)
   - 1.6 [`src/infer_csma.py` — 推理与可视化](#16-srcinfer_csmapy--推理与可视化)
2. [训练管线](#2-训练管线)
   - 2.1 [整体流程](#21-整体流程)
   - 2.2 [三阶段渐进课程](#22-三阶段渐进课程)
   - 2.3 [损失函数](#23-损失函数)
   - 2.4 [优化器与调度](#24-优化器与调度)
3. [实验计划](#3-实验计划)
   - 3.1 [数据集](#31-数据集)
   - 3.2 [评估指标](#32-评估指标)
   - 3.3 [主实验](#33-主实验)
   - 3.4 [消融实验](#34-消融实验)
   - 3.5 [Zero-shot 泛化实验](#35-zero-shot-泛化实验)
4. [文件结构与依赖](#4-文件结构与依赖)
   - 4.1 [项目文件树](#41-项目文件树)
   - 4.2 [新增依赖](#42-新增依赖)
   - 4.3 [与现有代码的向后兼容性](#43-与现有代码的向后兼容性)

---

## 1. 模块设计

### 1.1 配置管理（Config）

使用 Python `dataclass` 统一管理所有超参数，避免分散在 `argparse` 中难以追踪。已实现文件 `src/config.py`：

```python
# src/config.py  （已实现）
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

LossMode = Literal["align_only", "det_only", "full"]

@dataclass
class CSMAConfig:
    # ── 模型结构 ────────────────────────────────────────
    ir_enc_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    num_rgb_prototypes: int = 512       # RGB 原型库大小 K
    proto_dim: int = 256                # 原型向量维度，与 DINO hidden_dim 对齐
    num_cross_attn_heads: int = 8       # RPCA 多头注意力头数；须满足 proto_dim % num_heads == 0
    use_residual: bool = True           # 像素解码器是否加残差跳接

    # ── 训练管线 ────────────────────────────────────────
    total_epochs: int = 100             # 总训练 epoch，三阶段各占 1/3
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    loss_mode: LossMode = "full"

    # ── CMSS / GMM ──────────────────────────────────────
    mask_ratio: float = 0.75            # 阶段 B 的默认掩码比例
    gmm_n_components: int = 3
    gmm_update_every: int = 10          # 每隔多少 epoch 重新拟合 GMM

    # ── 损失权重 ────────────────────────────────────────
    stage_loss_weights: list[tuple[float, float]] = field(
        default_factory=lambda: [(1.0, 0.1), (0.5, 0.5), (0.1, 1.0)]
    )
    det_w_bbox: float = 5.0
    det_w_giou: float = 2.0
    det_w_ce_enc: float = 0.1
    det_w_bbox_enc: float = 0.5
    det_w_giou_enc: float = 0.5

    # ── 数据 ────────────────────────────────────────────
    ir_data_root: str = "train"
    rgb_data_root: str = "train/rgb"
    text_prompt: str = "person. car."
    num_workers: int = 4

    # ── 路径 ────────────────────────────────────────────
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    output_dir: str = "outputs_csma"
    vis_every: int = 10

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict，供日志或保存使用。"""
        return asdict(self)

    @classmethod
    def from_overrides(cls, overrides: dict[str, Any] | None = None) -> "CSMAConfig":
        """带覆盖项构建配置；未知键直接报错，避免静默误配置。"""
        ...

    def validate(self) -> None:
        """检验值域与结构约束，包含：proto_dim % num_cross_attn_heads == 0。"""
        ...
```

---

### 1.2 `src/csma.py` — 核心适配器模型

包含三个子模块类和顶层 `CSMA` 类，接口与现有 `ResidualTranslator` 完全一致。已实现文件 `src/csma.py`：

```python
# src/csma.py  （已实现，关键接口摘要）
import torch
import torch.nn as nn
from src.config import CSMAConfig


# ══════════════════════════════════════════════════════
#  子模块一：多尺度红外特征编码器（IR Encoder, IRE）
# ══════════════════════════════════════════════════════
class IREncoder(nn.Module):
    """
    三层步进卷积金字塔，提取多尺度红外特征。
    ir_enc_channels 长度须为 4：[stem_ch, c1_ch, c2_ch, c3_ch]。
    参数量约 0.39M（默认 channels=[32,64,128,256]）。
    """
    def __init__(self, channels: list[int]) -> None:
        # stem: Conv(3→c0, k=3, s=1, p=1) + BN + GELU
        # layer1: Conv(c0→c1, k=3, s=2, p=1) + BN + GELU → H/2
        # layer2: Conv(c1→c2, k=3, s=2, p=1) + BN + GELU → H/4
        # layer3: Conv(c2→c3, k=3, s=2, p=1) + BN + GELU → H/8
        ...

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 返回 (c1, c2, c3)，供 CSMA.forward 和 PixelDecoder 跳接
        # c1: [B, 64,  H/2, W/2]
        # c2: [B, 128, H/4, W/4]
        # c3: [B, 256, H/8, W/8]
        ...


# ══════════════════════════════════════════════════════
#  子模块二：RGB 原型交叉注意力（RPCA）
# ══════════════════════════════════════════════════════
class RGBPrototypeCrossAttention(nn.Module):
    """
    改造自 M-SpecGene CrossTransformerEncoderLayer：
      原版：Q=红外特征，K/V=运行时 RGB 特征（需双流输入）
      本版：Q=红外特征，K/V=可学习 RGB 原型库（推理时无需 RGB）

    含完整 Transformer Block：MHA + norm1 + FFN(256→1024→256) + norm2
    参数量约 1.0M（K=512, dim=256, 8头）。
    """
    def __init__(self, ir_dim: int, proto_dim: int,
                 num_heads: int, num_prototypes: int,
                 ffn_dim: int = 1024) -> None:
        # self.rgb_prototypes = nn.Parameter(torch.randn(K, proto_dim) * 0.02)
        # self.q_proj  = nn.Linear(ir_dim, proto_dim)
        # self.cross_attn = nn.MultiheadAttention(proto_dim, num_heads, batch_first=True)
        # self.norm1 = nn.LayerNorm(proto_dim)
        # self.ffn   = nn.Sequential(Linear(proto_dim, ffn_dim), GELU, Linear(ffn_dim, proto_dim))
        # self.norm2 = nn.LayerNorm(proto_dim)
        ...

    def forward(self, ir_feat: torch.Tensor) -> torch.Tensor:
        # q = self.q_proj(ir_feat)
        # kv = self.rgb_prototypes.unsqueeze(0).expand(B, -1, -1)
        # attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        # x = self.norm1(ir_feat + attn_out)   ← 残差作用在 ir_feat 上
        # x = self.norm2(x + self.ffn(x))
        # return x  # [B, L, proto_dim]
        ...


# ══════════════════════════════════════════════════════
#  子模块三：像素级重建解码器（Pixel Decoder, PD）
# ══════════════════════════════════════════════════════
class PixelDecoder(nn.Module):
    """
    三级转置卷积上采样，将 H/8 还原至 H，输出伪 RGB 图像。
    每级 concat 跳接后接 1×1 融合卷积（U-Net 风格）。
    末层 torch.tanh 将输出限幅至 [-1, 1]。
    head 权重近零初始化（1e-4），保证训练初期 delta ≈ 0。
    参数量约 0.72M。
    """
    def __init__(self, in_ch=256, skip2_ch=128, skip1_ch=64, head_out_ch=32) -> None:
        # up3: ConvTranspose2d(256→128, k=4, s=2, p=1) + BN + GELU
        # fuse3: Conv2d(128+skip2_ch→128, k=1)
        # up2: ConvTranspose2d(128→64,  k=4, s=2, p=1) + BN + GELU
        # fuse2: Conv2d(64+skip1_ch→64, k=1)
        # up1: ConvTranspose2d(64→32,   k=4, s=2, p=1) + BN + GELU
        # head: Conv2d(32→3, k=1)，权重 constant_(1e-4)
        ...

    def forward(self, feat: torch.Tensor,
                skip_c2: torch.Tensor, skip_c1: torch.Tensor) -> torch.Tensor:
        # 逐级上采样 + concat + fuse，末层 torch.tanh(self.head(x))
        # 输出 [B, 3, H, W]
        ...


# ══════════════════════════════════════════════════════
#  顶层：CSMA（即插即用跨光谱模态适配器）
# ══════════════════════════════════════════════════════
class CSMA(nn.Module):
    """
    即插即用接口：pseudo_rgb = csma(ir_pixel_values)
    与 ResidualTranslator 接口完全一致，可直接替换。
    总参数量约 2.0M（Grounding DINO Tiny 173M 的 1.16%）。

    运行时约束：ir_enc_channels[-1] 必须等于 proto_dim（均默认 256）。
    """
    def __init__(self, cfg: CSMAConfig) -> None:
        # 验证 ir_enc_channels[-1] == proto_dim
        # self.ire  = IREncoder(cfg.ir_enc_channels)
        # self.rpca = RGBPrototypeCrossAttention(
        #     ir_dim=last_ch, proto_dim=cfg.proto_dim,
        #     num_heads=cfg.num_cross_attn_heads,
        #     num_prototypes=cfg.num_rgb_prototypes)
        # self.pd = PixelDecoder(
        #     in_ch=cfg.proto_dim,
        #     skip2_ch=cfg.ir_enc_channels[2],
        #     skip1_ch=cfg.ir_enc_channels[1])
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # c1, c2, c3 = self.ire(x)
        # B, _, H8, W8 = c3.shape
        # feat = c3.flatten(2).transpose(1, 2)          # [B, L, 256]
        # feat = self.rpca(feat)
        # feat_map = feat.transpose(1, 2).reshape(B, proto_dim, H8, W8)
        # pseudo_rgb = self.pd(feat_map, c2, c1)
        # if cfg.use_residual: pseudo_rgb = pseudo_rgb + x
        # return pseudo_rgb  # [B, 3, H, W]
        ...

    def get_intermediate_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅供 L_align 计算使用，返回 RPCA 输出的语义特征序列。
        不经过 PD，不加输入残差。
        Returns: [B, L, proto_dim]，L = (H/8)*(W/8)
        """
        # _, _, c3 = self.ire(x)
        # feat = c3.flatten(2).transpose(1, 2)
        # return self.rpca(feat)
        ...
```

---

### 1.3 `src/cmss_utils.py` — CMSS 计算与 GMM 工具

直接移植 `M-SpecGene-main/pretrain/.../GMM_CMSS_SAMPLE.py` 的核心逻辑，去除 MAE 预训练依赖，封装为独立工具函数。已实现文件 `src/cmss_utils.py`：

```python
# src/cmss_utils.py  （已实现，关键接口摘要）
from __future__ import annotations
from typing import Optional
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from src.config import CSMAConfig

_DEFAULT_SORTED_MEANS: tuple[float, float, float] = (0.2, 0.5, 0.8)


def compute_cmss(feat_rgb: torch.Tensor, feat_ir: torch.Tensor) -> torch.Tensor:
    """
    计算每个 Patch 的 CMSS 值。直接对应 GMM_CMSS_SAMPLE.py 第 85–97 行。
    输入须为 [B, L, D]（批量版，原码 flat [N*L,D] 语义等价）。

    Args:
        feat_rgb: [B, L, D]  冻结 DINO 对真实 RGB 提取的 Patch 特征
        feat_ir:  [B, L, D]  冻结 DINO 对伪 RGB 提取的 Patch 特征
    Returns:
        cmss_map: [B, L]，值域 [0, 1]，全局 max 归一化
    """
    # norm 后余弦相似度 → r = sqrt[(cos+1)/2]
    # var_rgb = feat_rgb.var(dim=-1)，var_ir = feat_ir.var(dim=-1)
    # cmss = r / (var_rgb * var_ir + 1e-6)
    # cmss = cmss / cmss.max().clamp(min=1e-6)
    ...


def fit_gmm(
    cmss_values: np.ndarray,
    n_components: int = 3,
) -> tuple[np.ndarray, GaussianMixture]:
    """
    对全数据集 CMSS 值拟合 GMM，返回排序后均值与模型。
    μ₁ < μ₂ < μ₃，对应：目标核心 < 边缘过渡 < 背景

    Args:
        cmss_values: [N] 1D float32 数组，N = 样本数 × Patch 数
    Returns:
        sorted_means: [n_components] 升序均值数组
        gmm: GaussianMixture(covariance_type='full', random_state=42)
    """
    ...


def build_cmss_mask(
    cmss_map: torch.Tensor,
    stage: int,
    mu1: float, mu2: float, mu3: float,
    mask_ratio: float = 0.75,
    gmm: Optional[GaussianMixture] = None,
) -> torch.Tensor:
    """
    根据训练阶段和 GMM 均值生成 Patch 级掩码。
    mask=1 表示该 Patch 被掩蔽（不参与 L_align 计算）。

    阶段 A (stage=0)：(cmss_map > mu2).float()   —— 掩蔽背景
    阶段 B (stage=1)：GMM 采样噪声 → argsort → scatter（简化版，须传入 gmm）
    阶段 C (stage=2)：(cmss_map < mu1).float()   —— 掩蔽目标核心
    """
    ...


class CMSSScheduler:
    """
    管理三阶段课程切换和 GMM 定期更新。
    阶段边界：[T//3, 2T//3]，epoch 从 0 计数。
    """
    def __init__(self, cfg: CSMAConfig) -> None:
        # self._stage_boundaries = [T//3, T*2//3]
        # self._gmm_update_every = cfg.gmm_update_every
        # self._gmm:          Optional[GaussianMixture] = None
        # self._sorted_means: Optional[np.ndarray]      = None
        ...

    def get_stage(self, epoch: int) -> int:
        """返回 0（A）、1（B）、2（C）。"""
        ...

    def should_update_gmm(self, epoch: int) -> bool:
        """epoch % gmm_update_every == 0 时返回 True。"""
        ...

    def update_gmm(self, cmss_values: np.ndarray) -> None:
        """调用 fit_gmm 并更新内部 _sorted_means / _gmm。"""
        ...

    @property
    def gmm(self) -> Optional[GaussianMixture]:
        """已拟合的 GMM；未初始化时为 None。"""
        ...

    @property
    def sorted_means(self) -> tuple[float, float, float]:
        """
        返回 (μ₁, μ₂, μ₃)。
        GMM 尚未拟合时返回安全默认值 _DEFAULT_SORTED_MEANS = (0.2, 0.5, 0.8)。
        """
        ...

    def get_loss_weights(self, epoch: int) -> tuple[float, float]:
        """返回当前 epoch 的 (lambda_align, lambda_det)，查 cfg.stage_loss_weights。"""
        ...
```

---

### 1.4 `src/dataset_paired.py` — RGB-IR 配对数据集

继承现有 `FlirCocoOverfitDataset`，增加同步加载 RGB 配对图像的能力，供 CMSS 计算和 L_align 使用。已实现文件 `src/dataset_paired.py`：

```python
# src/dataset_paired.py  （已实现，关键接口摘要）
from src.dataset import FlirCocoOverfitDataset, collate_fn


class FlirPairedDataset(FlirCocoOverfitDataset):
    """
    在 COCO 格式红外数据集基础上，额外加载对应 RGB 配对图像。
    RGB 路径规则：取 IR 文件名，在 rgb_root 目录下查找同名文件。
    若 RGB 不存在，rgb_pixel_values 置 None，训练时跳过 L_align。
    """
    def __init__(self, ir_root: str, rgb_root: str, processor,
                 text_prompt: str, coco_category_id_to_class_idx: dict) -> None:
        super().__init__(ir_root, processor, text_prompt, coco_category_id_to_class_idx)
        self._rgb_root = os.path.abspath(rgb_root)

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)          # 获取完整 IR 样本
        rgb_path = Path(self._rgb_root) / Path(sample["image_path"]).name
        if rgb_path.exists():
            rgb_enc = self._processor.image_processor(
                images=Image.open(rgb_path).convert("RGB"), return_tensors="pt"
            )
            sample["rgb_pixel_values"] = rgb_enc["pixel_values"][0]   # [3, H, W]
        else:
            sample["rgb_pixel_values"] = None
        return sample


def collate_paired(batch: list[dict]) -> dict:
    """
    在 collate_fn 基础上额外处理 rgb_pixel_values（可能含 None）。

    "全有或全无"策略：
        全部样本均有 RGB → rgb_pixel_values: Tensor[B,3,H,W] 写入输出
        任一样本缺少 RGB → 该键不出现，训练循环凭 "rgb_pixel_values" in batch 判断
    """
    base = collate_fn(batch)
    rgb_list = [b.get("rgb_pixel_values") for b in batch]
    if all(v is not None for v in rgb_list):
        base["rgb_pixel_values"] = torch.stack(rgb_list)
    return base
```

---

### 1.5 `src/train_csma.py` — CSMA 训练主程序

> **实现状态**：✅ 已实现（`src/train_csma.py`）

在 `src/train_demo.py` 基础上增加 CMSS 引导的 L_align 逻辑，其余训练框架（SwanLab 日志、可视化、梯度检查）原样保留。

```python
# ── 公共函数接口 ─────────────────────────────────────────────────────────────

def _move_labels_to_device(
    labels: List[Dict[str, Any]], device: torch.device
) -> List[Dict[str, Any]]:
    """将 labels list 中所有 Tensor 批量移至指定设备。"""

def _build_swanlab_logger(
    enable: bool, project: str, run_name: str, config: Dict[str, Any]
) -> Optional[Any]:
    """按需初始化 SwanLab；未安装或 enable=False 时安全返回 None。"""

def _build_det_loss(
    outputs: Any, cfg: CSMAConfig
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    用 CSMAConfig 权重重建 L_det（6 分量加权求和）。
    返回：(loss_tensor, scalars_dict)
    L_det = L_ce + w_bbox*L_bbox + w_giou*L_giou
          + w_ce_enc*L_ce_enc + w_bbox_enc*L_bbox_enc + w_giou_enc*L_giou_enc
    """

def extract_dino_backbone_features(
    dino_model: GroundingDinoForObjectDetection,
    pixel_values: torch.Tensor,      # [B, 3, H, W]
    input_ids: torch.Tensor,         # [1, T]，函数内 expand → [B, T]
    attention_mask: torch.Tensor,    # [1, T]，函数内 expand → [B, T]
) -> torch.Tensor:                   # [B, L_total, 256]
    """
    forward hook 捕获 model.encoder 入口处特征（input_proj 投影后）。
    hook 注册在 dino_model.model.encoder.register_forward_hook，
    捕获 inp[0]（encoder.forward 第一个位置参数）。
    """

def compute_align_loss(
    feat_ir: torch.Tensor,   # [B, L, D]，有梯度
    feat_rgb: torch.Tensor,  # [B, L, D]，须已 detach
    mask: torch.Tensor,      # [B, L]，0=保留 1=掩蔽
) -> torch.Tensor:           # 标量 MSE
    """
    仅对 mask==0 的 Patch 计算 MSE 对齐损失。
    若全部 Patch 被掩蔽，返回 requires_grad=True 的 0 张量避免反传报错。
    """

def collect_cmss_values(
    dino_model: GroundingDinoForObjectDetection,
    csma: CSMA,
    loader: DataLoader,
    device: torch.device,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> np.ndarray:             # 1D float32，[N_total_patches]
    """
    torch.no_grad() 遍历全量训练集，收集所有 Patch 的 CMSS 值供 GMM 重拟合。
    无 rgb_pixel_values 的 batch 跳过（FLIR 配对数据集通常不触发）。
    """

def main() -> None:
    """
    argparse CLI 入口，可覆盖字段：
      --data-root, --rgb-data-root, --out-dir
      --epochs, --batch-size, --lr
      --use-swanlab, --swanlab-project, --swanlab-run-name
    训练结束后在 output_dir/logs/loss.png 保存 loss 曲线。
    """
```

---

### 1.6 `src/infer_csma.py` — 推理与可视化

> **实现状态**：✅ 已实现（`src/infer_csma.py`）

推理阶段仅需红外图像，无需 RGB、无需 GMM、无需 L_align，接口比训练更简单。
工具函数（`denormalize_pixel_values`、`cxcywh_norm_to_xyxy_pixels`、`draw_boxes`）直接从 `infer_vis.py` 导入，不重复实现。

```python
def run_inference(
    csma: CSMA,
    dino: GroundingDinoForObjectDetection,
    processor: Any,
    ir_image_path: str,
    text_prompt: str,
    device: torch.device,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    单张红外图像推理，返回检测结果与可视化用伪 RGB 图像。
    返回：{boxes[N,4], scores[N], labels: List[str], pseudo_rgb[1,3,H,W], pseudo_rgb_np[H,W,3]}
    """

def visualize_cmss_mask(
    ir_pv: Tensor,                    # [1, 3, H, W]
    csma: CSMA,
    dino: GroundingDinoForObjectDetection,
    cmss_sched: CMSSScheduler,
    stage: int,                       # 0=A 1=B 2=C
    device: torch.device,
    input_ids: Tensor,
    attention_mask: Tensor,
    processor: Any,
    rgb_pv: Optional[Tensor] = None,  # 有则真实 CMSS；无则 pseudo_rgb 自比较
    alpha: float = 0.5,
) -> np.ndarray:                      # HWC uint8，IR 图 + RdYlGn 热力图叠加
    """
    论文 Figure 专用：将 GMM-CMSS 掩码以热力图叠加在红外图像上。
    L_total 取第一尺度 n1=(H//8)*(W//8) tokens，reshape 后最近邻上采样至 (H,W)。
    """

def save_multi_sample_grid_csma(
    csma: CSMA,
    dino: GroundingDinoForObjectDetection,
    processor: Any,
    samples: List[Dict[str, Any]],
    text_prompt: str,
    device: torch.device,
    out_path: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> None:
    """
    多样本三联对比图（IR | 伪 RGB | 伪 RGB + 预测框/GT），保存为 PNG。
    与 infer_vis.save_multi_sample_grid 逻辑相同，仅 translator → csma（类型安全版）。
    """

def main() -> None:
    """
    argparse CLI：
      --ckpt（必填）, --data-root, --out, --num-samples
      --out-mask（可选，CMSS 热力图）, --stage 0/1/2
      --box-threshold, --text-threshold
    """
```

---

## 2. 训练管线

### 2.1 整体流程

```
初始化
  ├── 加载冻结 Grounding DINO（grounding-dino-tiny）
  ├── 初始化 CSMA（随机权重，最后一层近零初始化保证残差近恒等）
  ├── 初始化 CMSSScheduler（管理阶段切换和 GMM 更新）
  └── 加载 FlirPairedDataset（IR + 可选 RGB 配对）

训练循环（epoch 1 ~ T）
  ├── [每 10 epoch] 重新拟合 GMM（当 RGB 配对可用时）
  │     └── 遍历全部训练集，计算 CMSS 值，调用 fit_gmm()
  │
  ├── [每 batch] 前向计算
  │     ├── pseudo_rgb = csma(ir_pixel_values)          ← 适配器前向
  │     ├── L_det: dino(pseudo_rgb, labels) → loss_dict ← 检测损失
  │     └── L_align（若有 RGB 配对）:
  │           ├── feat_rgb = dino_backbone(rgb_pixel_values)  [stop_grad]
  │           ├── feat_ir  = dino_backbone(pseudo_rgb)
  │           ├── cmss_map = compute_cmss(feat_rgb, feat_ir)
  │           ├── mask     = build_cmss_mask(cmss_map, stage, μ₁, μ₂, μ₃)
  │           └── L_align  = MSE(feat_ir[mask=0], feat_rgb[mask=0])
  │
  ├── [每 batch] 反传
  │     ├── loss = λ_align * L_align + λ_det * L_det
  │     ├── loss.backward()         ← 梯度仅流向 CSMA，DINO 梯度为 None
  │     ├── clip_grad_norm_(csma, 1.0)
  │     └── optimizer.step()
  │
  └── [每 vis_every epoch] 保存权重 + 可视化检测结果

输出
  ├── outputs_csma/ckpt/csma_best.pt
  ├── outputs_csma/logs/（SwanLab + loss 曲线）
  └── outputs_csma/vis/（检测框可视化）
```

### 2.2 三阶段渐进课程

| 阶段 | Epoch 范围 | 掩码策略 | 被保留的 Patch | 学习重点 |
|:---:|:---:|:---|:---:|:---|
| **A（Easy）** | 1 ~ T/3 | 掩蔽 CMSS > μ₂（背景） | 低 CMSS 的目标核心 | 建立 IR → RGB 的基础映射，快速收敛 |
| **B（Mixed）** | T/3 ~ 2T/3 | GMM 概率分布随机掩蔽 | 全局混合 | 学习完整场景上下文和边缘纹理 |
| **C（Hard）** | 2T/3 ~ T | 掩蔽 CMSS < μ₁（目标核心） | 高 CMSS 的背景区域 | 从遮挡场景中脑补目标特征，提升鲁棒性 |

损失权重随阶段动态切换（见 TD-05 决策，由 `CMSSScheduler.get_loss_weights()` 返回）：

| 阶段 | λ_align | λ_det | 侧重 |
|:---:|:---:|:---:|:---|
| A | 1.0 | 0.1 | 以特征对齐为主导 |
| B | 0.5 | 0.5 | 对齐与检测并重 |
| C | 0.1 | 1.0 | 以检测精度为主导 |

### 2.3 损失函数

**L_align（CMSS 引导的跨模态特征蒸馏）**

$$\mathcal{L}_{align} = \frac{1}{|\Omega|} \sum_{i \in \Omega} \left\| F_{ir}^{(i)} - \mathrm{sg}\!\left(F_{rgb}^{(i)}\right) \right\|_2^2$$

其中 $\Omega = \{i \mid M_{cmss}^{(i)} = 0\}$ 为当前阶段 GMM-CMSS 掩码保留的 Patch 集合，$\mathrm{sg}(\cdot)$ 表示停止梯度。

**L_det（沿用 `_build_train_loss`，代码零改动）**

$$\mathcal{L}_{det} = \mathcal{L}_{ce} + 5.0\,\mathcal{L}_{bbox} + 2.0\,\mathcal{L}_{giou} + 0.1\,\mathcal{L}_{ce}^{enc} + 0.5\,\mathcal{L}_{bbox}^{enc} + 0.5\,\mathcal{L}_{giou}^{enc}$$

**总损失**

$$\mathcal{L}_{total} = \lambda_{align}(t) \cdot \mathcal{L}_{align} + \lambda_{det}(t) \cdot \mathcal{L}_{det}$$

### 2.4 优化器与调度

```
优化器：AdamW(csma.parameters(), lr=1e-4, weight_decay=1e-2)
调度器：CosineAnnealingLR(optimizer, T_max=total_epochs)
梯度裁剪：clip_grad_norm_(csma, max_norm=1.0)
```

---

## 3. 实验计划

### 3.1 数据集

| 数据集 | 用途 | 帧数（IR） | RGB 配对 | 获取 |
|:---|:---:|:---:|:---:|:---|
| **FLIR ADAS v2**（主力） | 训练 + 测试 | ~26k | ~9,748 对 | `download_FLIR.py` 已有 |
| **KAIST**（补充） | 跨数据集泛化测试 | ~95k | 95k 对 | 公开下载 |
| **LLVIP** | 跨数据集泛化测试 | 15k | 15k 对 | 公开下载 |

训练集与测试集按 FLIR 官方划分，不额外引入未标注数据。

### 3.2 评估指标

| 指标 | 含义 | 对应数据集 |
|:---:|:---|:---:|
| mAP@0.5 | COCO 标准检测精度 | FLIR, LLVIP |
| mAP@0.5:0.95 | 严格 COCO 精度 | FLIR |
| MR（Miss Rate）@FPPI=0.1 | 行人漏检率（越低越好） | KAIST |
| Zero-shot mAP | 新类别开放词汇检测精度 | FLIR（未见类别） |

### 3.3 主实验

| 模型方案 | 可训练参数 | IR 训练数据 | FLIR mAP@0.5 | Zero-shot |
|:---|:---:|:---:|:---:|:---:|
| M-SpecGene（目标 Baseline） | 100M+（预训练） | 550K 对 | ~79.3 | 无 |
| Grounding DINO Tiny（原版） | 0 | 0 | ~15 | 有 |
| DINO + ResidualTranslator（当前 MVP） | ~0.5K | 10 张 | 待测 | 有 |
| DINO + CSMA（无 CMSS 引导） | ~2M | ~10K 对 | 消融基线 | 有 |
| **DINO + CSMA（完整，本方案）** | **~2M** | **~10K 对** | **目标 ≥ 80** | **有** |

### 3.4 消融实验

**消融一：CSMA 各子模块的必要性**

| 配置 | IRE | RPCA | PD 跳接 | mAP@0.5 |
|:---|:---:|:---:|:---:|:---:|
| 仅残差 CNN（现有 MVP） | ✗ | ✗ | ✗ | - |
| + 多尺度编码器（IRE） | ✓ | ✗ | ✗ | - |
| + RGB 原型交叉注意力（RPCA） | ✓ | ✓ | ✗ | - |
| + 跳接解码器（完整 CSMA） | ✓ | ✓ | ✓ | - |

**消融二：GMM-CMSS 策略的贡献**

| 配置 | mAP@0.5 |
|:---|:---:|
| CSMA + 无 L_align（仅 L_det） | - |
| CSMA + L_align（随机掩码，无 CMSS） | - |
| CSMA + L_align（固定阈值 CMSS，无 GMM） | - |
| CSMA + L_align（GMM-CMSS，单阶段） | - |
| **CSMA + L_align（三阶段渐进 GMM-CMSS，完整方案）** | - |

**消融三：RGB 原型库大小 K 的影响**

K ∈ {64, 128, 256, **512**, 1024}，固定其他超参数，测试 FLIR mAP@0.5，以确认 K=512 为最优。

**消融四：L_align 特征提取位置的影响**

| 特征提取点 | mAP@0.5 |
|:---|:---:|
| 像素级（RGB 原图） | - |
| 骨干网络中间层（layer2） | - |
| **input_proj 投影后（采用，256 维）** | - |
| Transformer Encoder 输出 | - |

### 3.5 Zero-shot 泛化实验

**目的**：证明 CSMA 未损伤 Grounding DINO 的开放词汇能力。

训练时 prompt：`"person. car."`

测试时使用训练中从未出现的类别：

| 测试 prompt | 目标类别 | DINO vanilla（RGB） | DINO+CSMA（IR） | M-SpecGene |
|:---:|:---:|:---:|:---:|:---:|
| `"bicycle."` | 自行车 | - | - | 0（无法识别） |
| `"traffic light."` | 交通灯 | - | - | 0 |
| `"truck."` | 卡车 | - | - | 0 |

**预期结论**：DINO+CSMA 的 Zero-shot mAP 接近 DINO vanilla（RGB），远优于 M-SpecGene（=0），证明开放词汇能力完整保留。

---

## 4. 文件结构与依赖

### 4.1 项目文件树

```
demo_RGBT_net/
├── docs/
│   └── TD.md                        ← 本文档
│
├── src/
│   ├── __init__.py                  （现有，无需修改）
│   ├── config.py                    ← 【新增】CSMAConfig dataclass
│   ├── csma.py                      ← 【新增】CSMA 模型（IRE + RPCA + PD）
│   ├── cmss_utils.py                ← 【新增】CMSS 计算 + GMM 工具
│   ├── dataset.py                   （现有，无需修改，FlirCocoOverfitDataset）
│   ├── dataset_paired.py            ← 【新增】FlirPairedDataset（继承 dataset.py）
│   ├── translator.py                （现有，ResidualTranslator，作为消融对照保留）
│   ├── train_demo.py                （现有，无需修改，作为 MVP 基线保留）
│   ├── train_csma.py                ← 【新增】CSMA 完整训练主程序
│   ├── infer_vis.py                 （现有，无需修改）
│   └── infer_csma.py                ← 【新增】CSMA 推理与可视化
│
├── train/                           （现有，FLIR COCO 格式红外数据）
│   ├── ir/                          ← 建议重命名整理（现有 *.jpg 移入）
│   ├── rgb/                         ← 【新增目录】RGB 配对图像
│   └── _annotations.coco.json      （现有，无需修改）
│
├── GroundingDINO-main/              （现有，参考代码，不修改）
├── M-SpecGene-main/                 （现有，参考代码，不修改）
├── requirements.txt                 ← 【更新】新增 scikit-learn
└── outputs_csma/                    ← 【新增】训练输出目录
    ├── ckpt/
    ├── logs/
    └── vis/
```

**文件数量汇总**：新增 5 个 `.py` 文件 + 1 个目录（`train/rgb/`），修改 1 个文件（`requirements.txt`），现有全部代码无破坏性改动。

### 4.2 新增依赖

| 库 | 用途 | 安装命令 |
|:---|:---|:---|
| `scikit-learn` | GMM 拟合（`GaussianMixture`），已在 M-SpecGene 代码中使用 | `pip install scikit-learn` |

其余所有依赖（`torch`, `transformers`, `Pillow`, `matplotlib`, `swanlab`）已在现有 `requirements.txt` 中声明，无需变更。

### 4.3 与现有代码的向后兼容性

| 现有文件 | 修改情况 | 说明 |
|:---|:---:|:---|
| `src/train_demo.py` | 不修改 | 作为 MVP 基线对照实验保留 |
| `src/translator.py` | 不修改 | `ResidualTranslator` 作为消融实验最弱 baseline 保留 |
| `src/dataset.py` | 不修改 | `FlirPairedDataset` 通过继承扩展，不破坏原类 |
| `src/infer_vis.py` | 不修改 | `infer_csma.py` 通过 import 复用其可视化函数 |
| `train/_annotations.coco.json` | 不修改 | 标注格式完全复用 |

---

*文档版本：v2.0 | 日期：2026-04-27*
