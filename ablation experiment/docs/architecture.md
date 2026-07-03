# CSMA 统一结构设计

> **Cross-Spectral Modality Adapter**：一个即插即用的轻量级红外模态适配器，使冻结的 Grounding DINO 具备高精度红外目标检测能力，同时保留原生开放词汇（Zero-shot）能力。

---

## 目录

1. [核心定位](#1-核心定位)
2. [系统总览](#2-系统总览)
3. [模块一：多尺度红外编码器（IREncoder / IRE）](#3-模块一多尺度红外编码器irencoder--ire)
4. [模块二：RGB 原型交叉注意力（RPCA）](#4-模块二rgb-原型交叉注意力rpca)
5. [模块三：像素解码器（PixelDecoder / PD）](#5-模块三像素解码器pixeldecoder--pd)
6. [模块四：顶层适配器（CSMA）](#6-模块四顶层适配器csma)
7. [训练管线](#7-训练管线)
8. [端到端推理](#8-端到端推理)
9. [可训练 vs 冻结组件](#9-可训练-vs-冻结组件)
10. [核心数据结构](#10-核心数据结构)
11. [文件结构](#11-文件结构)

---

## 1. 核心定位

### 问题

主流视觉大模型（Grounding DINO、CLIP 等）在 RGB 数据上预训练，直接用于红外图像时性能急剧退化。在 FLIR ADAS 数据集上，Grounding DINO Tiny 的 mAP@0.5 约为 **15%**，而专用红外预训练模型 M-SpecGene 可达约 **79%**。

### 方案定位

```
┌─────────────────────────────────────────────────────────┐
│                   已有方案的代价                          │
│                                                          │
│  M-SpecGene      → 100M+ 参数预训练，550K 配对数据，      │
│                     8×A100 × 500 epoch，无零样本能力      │
│                                                          │
│  全量微调 DINO   → 灾难性遗忘，丧失零样本能力             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   本方案的定位                            │
│                                                          │
│  DINO（冻结）  +  CSMA（~2M 可训练参数）                  │
│                                                          │
│  ✓ 仅需 ~10K 配对训练数据                                │
│  ✓ 保留 Grounding DINO 全部零样本能力                    │
│  ✓ 即插即用，不修改 DINO 任何代码                        │
│  ✓ 目标精度：FLIR mAP@0.5 ≥ 80，超越 M-SpecGene         │
└─────────────────────────────────────────────────────────┘
```

### 核心论点

**"通用视觉大模型（Grounding DINO）+ 轻量适配器（CSMA）≥ 领域专有预训练大模型（M-SpecGene）"**

CSMA 通过将 M-SpecGene 的 GMM-CMSS 策略从"MAE 重建掩码"改造为"跨模态特征蒸馏引导信号"，以极低代价将 M-SpecGene 的核心跨模态知识迁移至 DINO 的推理链路中。

---

## 2. 系统总览

### 整体架构图

```
═══════════════════════════════════════════════════════════════════
 训练阶段（Training）— 需要 RGB-IR 配对数据
═══════════════════════════════════════════════════════════════════

     I_rgb ─────────────────────────────────┐
     [B,3,H,W]                              │
                                     冻结DINO Backbone
                                     (input_proj后)
                                            │ F_rgb [B,L,256]
                                            │ stop_gradient
     I_ir ──────► ┌──────────────────┐      │
     [B,3,H,W]   │   CSMA（可训练）  │      ▼
                  │  ┌────┐ ┌──────┐ │  ┌──────────────┐
                  │  │ IRE│►│ RPCA │ │  │  CMSS计算    │
                  │  └────┘ └──────┘ │  │  + GMM掩码   │  ──► M_cmss [B,L]
                  │       ┌──┘       │  └──────────────┘
                  │  ┌────┘          │        ▲
                  │  │  PD  │        │   F_ir [B,L,256]
                  └──┴──────┴────────┘        │
                         │ Î_rgb              │
                         │ [B,3,H,W]          │
                         │                    │
                         ▼                    │
                  冻结DINO Backbone ───────────┘
                         │
                         ▼
                  冻结DINO Transformer
                         │
                         ▼
                  检测输出 ──────────────────► L_det
                  (boxes, logits)
                                              L_align = MSE(F_ir, F_rgb)[M_cmss=0]
                                                   │
                                    L_total = λ₁·L_align + λ₂·L_det
                                                   │
                                              ▼ backward ▼
                                         仅更新 CSMA 权重

═══════════════════════════════════════════════════════════════════
 推理阶段（Inference）— 仅需红外图像
═══════════════════════════════════════════════════════════════════

     I_ir ──► CSMA ──► Î_rgb ──► 冻结Grounding DINO ──► 检测框 + 文本标签
              2M参数         text_prompt（任意开放词汇）
```

### 参数规模对比

| 组件 | 参数量 | 状态 |
|:---|:---:|:---:|
| Grounding DINO Tiny（完整） | 173M | 冻结 |
| — 其中 Backbone（Swin-T） | 28M | 冻结 |
| — 其中 Text Encoder（BERT） | 110M | 冻结 |
| — 其中 Transformer Decoder | 35M | 冻结 |
| **CSMA — IREncoder** | **~0.3M** | **可训练** |
| **CSMA — RPCA** | **~1.0M** | **可训练** |
| **CSMA — PixelDecoder** | **~0.7M** | **可训练** |
| **CSMA 合计** | **~2.0M** | **可训练** |
| **可训练占比** | **1.16%** | |

---

## 3. 模块一：多尺度红外编码器（IREncoder / IRE）

### 职责

将原始红外图像（经 ImageNet 均值方差归一化的 3 通道复制输入）通过三层步进卷积压缩为多尺度语义特征，同时保留各中间尺度的跳接张量供像素解码器使用。

### 设计依据

M-SpecGene 的 `MAEViT` 使用 ViT `PatchEmbed`（patch_size=16）对红外图像进行分块。本模块用 CNN 金字塔替代，每层 stride=2，三层后总下采样倍率为 8×，使最终输出空间分辨率 `H/8 × W/8` 与 DINO `input_proj` 后的多尺度特征尺度对齐。

选用 CNN 而非轻量 ViT 的理由：训练数据量小（~10K），ViT 全局注意力在此规模下易过拟合；CNN 的归纳偏置（局部性、平移不变性）对红外图像的热源边缘特征更友好。

### 网络结构

```
输入: I_ir  [B, 3, H, W]  （ImageNet 归一化）
  │
  ├── stem: Conv(3→32, k=3, s=1, p=1) + BN + GELU
  │         输出: [B, 32, H, W]          ← 保持分辨率，提取低级纹理
  │
  ├── layer1: Conv(32→64, k=3, s=2, p=1) + BN + GELU
  │           输出: c1 [B, 64, H/2, W/2]  ← 1/2 分辨率，浅层边缘特征
  │
  ├── layer2: Conv(64→128, k=3, s=2, p=1) + BN + GELU
  │           输出: c2 [B, 128, H/4, W/4] ← 1/4 分辨率，中层语义特征
  │
  └── layer3: Conv(128→256, k=3, s=2, p=1) + BN + GELU
              输出: c3 [B, 256, H/8, W/8] ← 1/8 分辨率，高层语义特征
```

### 输出

```python
forward(x: Tensor[B,3,H,W]) -> tuple[
    c1: Tensor[B, 64,  H/2, W/2],   # 跳接至 PD up1
    c2: Tensor[B, 128, H/4, W/4],   # 跳接至 PD up2
    c3: Tensor[B, 256, H/8, W/8],   # 送入 RPCA
]
```

### 参数量

| 层 | 参数量 |
|:---|:---:|
| stem | 3×32×9 + 32 ≈ 0.9K |
| layer1 | 32×64×9 + 64 ≈ 18.5K |
| layer2 | 64×128×9 + 128 ≈ 73.9K |
| layer3 | 128×256×9 + 256 ≈ 295.2K |
| **合计** | **~0.39M** |

---

## 4. 模块二：RGB 原型交叉注意力（RPCA）

### 职责

将红外特征序列与可学习的 RGB 原型库做交叉注意力，使红外特征"查询"到对应的 RGB 语义表示，从而完成跨模态域适配。

### 设计依据与改造逻辑

**M-SpecGene 原版**（`vision_transformer.py` `CrossTransformerEncoderLayer`）：

```python
# 运行时需要双流输入，Q=IR特征，K/V=RGB特征
def forward(self, x_ir, x_rgb):
    x = x_ir + self.attn(self.ln1(x_ir), self.ln1(x_rgb), self.ln1(x_rgb))
    return x
```

**本方案改造**：将 K/V 所依赖的运行时 RGB 特征替换为离线学习的**可学习原型 Token 库**：

```python
# 推理时仅需红外特征，K/V 来自参数库
def forward(self, x_ir):
    kv = self.rgb_prototypes.expand(B, -1, -1)  # [B, K, 256]
    x = x_ir + self.cross_attn(Q=x_ir, K=kv, V=kv)
    return x
```

**物理含义**：`rgb_prototypes`（形状 `[K=512, 256]`）是 K 个可学习向量，在训练过程中通过 `L_align` 的反向传播逐渐学习 RGB 语义域中不同类别（人、车、路面、天空等）的特征聚类中心。训练完成后这些原型被固化，推理时作为内置的跨模态知识库，无需任何 RGB 输入。

### 网络结构

```
输入: c3 → flatten → feat_seq  [B, L, 256]，L = (H/8)*(W/8)
  │
  ├── Q 投影: Linear(256→256)
  │
  ├── RGB 原型库: nn.Parameter[K=512, 256]（可学习）
  │   expand → kv  [B, 512, 256]
  │
  ├── MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
  │   Q=feat_seq, K=kv, V=kv
  │   输出: attn_out  [B, L, 256]
  │
  ├── 残差 + LayerNorm: feat_seq = LayerNorm(feat_seq + attn_out)
  │
  └── 点式前馈 FFN: Linear(256→1024) + GELU + Linear(1024→256)
      残差 + LayerNorm

输出: feat_aligned  [B, L, 256]
  → reshape → [B, 256, H/8, W/8]  送入 PixelDecoder
```

### 参数量

| 子层 | 参数量 |
|:---|:---:|
| rgb_prototypes | 512×256 = 131K |
| Q 投影 | 256×256 + 256 ≈ 65.8K |
| MultiheadAttention（Q/K/V/out proj） | 4×256² + 4×256 ≈ 263K |
| FFN（256→1024→256） | 256×1024 + 1024×256 + 1280 ≈ 524K |
| LayerNorm ×2 | 2×512 ≈ 1K |
| **合计** | **~985K ≈ 1.0M** |

---

## 5. 模块三：像素解码器（PixelDecoder / PD）

### 职责

将 RPCA 输出的语义特征图（`H/8 × W/8`）逐级上采样还原至原始分辨率（`H × W`），输出与 Grounding DINO `pixel_values` 输入空间对齐的伪 RGB 图像。

### 网络结构（U-Net 风格跳接）

```
输入: feat  [B, 256, H/8, W/8]  （来自 RPCA reshape 后）
  │
  ├── up3: ConvTranspose2d(256→128, k=4, s=2, p=1) + BN + GELU
  │        输出: [B, 128, H/4, W/4]
  │        cat(skip=c2[B,128,H/4,W/4]) → [B, 256, H/4, W/4]
  │        融合: Conv(256→128, k=1) → [B, 128, H/4, W/4]
  │
  ├── up2: ConvTranspose2d(128→64, k=4, s=2, p=1) + BN + GELU
  │        输出: [B, 64, H/2, W/2]
  │        cat(skip=c1[B,64,H/2,W/2]) → [B, 128, H/2, W/2]
  │        融合: Conv(128→64, k=1) → [B, 64, H/2, W/2]
  │
  ├── up1: ConvTranspose2d(64→32, k=4, s=2, p=1) + BN + GELU
  │        输出: [B, 32, H, W]
  │
  └── head: Conv2d(32→3, k=1) + Tanh
            输出: delta  [B, 3, H, W]  ∈ [-1, 1]

最终输出（含残差）:
  Î_rgb = delta + I_ir  （若 use_residual=True）
  Î_rgb ∈ [-1, 1]，与 ImageNet 归一化后的分布一致
```

### 残差设计意义

末层权重使用近零初始化（`nn.init.constant_(weight, 1e-4)`），保证训练初期 `delta ≈ 0`，即 `Î_rgb ≈ I_ir`。这使得梯度在训练起始阶段稳定流动，避免 DINO 骨干网络初期接收到随机生成的"伪 RGB"而产生极大的检测损失。此设计继承自现有 `src/translator.py` 中 `ResidualTranslator` 的初始化策略。

### 参数量

| 层 | 参数量 |
|:---|:---:|
| up3 + 融合 Conv | ~164K |
| up2 + 融合 Conv | ~41K |
| up1 | ~18K |
| head | 32×3×1 + 3 ≈ 0.1K |
| **合计** | **~0.72M** |

---

## 6. 模块四：顶层适配器（CSMA）

### 职责

串联 IRE、RPCA、PD 三个子模块，暴露与 `ResidualTranslator` 完全相同的接口，同时提供额外的中间特征提取接口供训练阶段 `L_align` 使用。

### 接口规范

```python
class CSMA(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        """
        主推理接口（训练和推理共用）。
        Args:  x  — [B, 3, H, W]，ImageNet 归一化的红外图像
        Returns: Î_rgb — [B, 3, H, W]，伪 RGB 图像，与 DINO pixel_values 同分布
        """

    def get_intermediate_features(self, x: Tensor) -> Tensor:
        """
        仅训练阶段使用，返回 RPCA 输出的语义特征图。
        Args:  x  — [B, 3, H, W]
        Returns: feat — [B, L, 256]，L = (H/8)*(W/8)
                 用于与 DINO 提取的 F_rgb 计算 L_align。
        """
```

### 完整数据流（Tensor 形状追踪）

以标准输入 `H=800, W=800`（DINO 默认处理尺寸）为例，Batch Size=4：

```
输入:   I_ir       [4, 3, 800, 800]
                        │
              ┌─────────┴──────────┐
        IRE stem                   │
              [4, 32, 800, 800]    │
        IRE layer1                 │
              c1 [4, 64, 400, 400] │ ──────────────── 跳接保存
        IRE layer2                 │
              c2 [4, 128, 200, 200]│ ──────── 跳接保存
        IRE layer3                 │
              c3 [4, 256, 100, 100]│
                        │
              flatten + transpose
              feat_seq  [4, 10000, 256]    L = 100×100 = 10000
                        │
              RPCA cross-attention
              feat_aln  [4, 10000, 256]
                        │
              reshape
              feat_map  [4, 256, 100, 100]
                        │
              PD up3 + cat(c2)
                        [4, 128, 200, 200]
              PD up2 + cat(c1)
                        [4, 64, 400, 400]
              PD up1
                        [4, 32, 800, 800]
              PD head + Tanh
              delta     [4, 3, 800, 800]
                        │
              + I_ir（残差跳接）
输出:   Î_rgb      [4, 3, 800, 800]   → 送入冻结 Grounding DINO
```

### 即插即用示例

```python
# 原有 MVP 代码（src/train_demo.py 第 243 行）
pseudo_rgb = translator(pixel_values)

# 升级后，一行替换，训练循环零改动
from src.csma import CSMA
from src.config import CSMAConfig
csma = CSMA(CSMAConfig()).to(device)
pseudo_rgb = csma(pixel_values)     # 接口完全相同
```

---

## 7. 训练管线

### 7.1 整体流程

```
【初始化阶段】
  ├── 加载 GroundingDinoForObjectDetection（HuggingFace，grounding-dino-tiny）
  │     dino.eval()，所有参数 requires_grad=False
  ├── 实例化 CSMA(CSMAConfig())，最后层近零初始化
  ├── 实例化 CMSSScheduler（管理三阶段切换 + GMM 更新计划）
  ├── 加载 FlirPairedDataset（IR 图像 + COCO 标注 + 可选 RGB 配对）
  └── 梯度检查预置（验证 CSMA 有梯度，DINO 无梯度）

【每 10 个 epoch：GMM 更新】
  ├── 遍历全训练集（torch.no_grad()）
  │     对每个样本：
  │       feat_rgb = dino_backbone(rgb_pixel_values)   # F_rgb
  │       feat_ir  = dino_backbone(csma(ir_pv))        # F_ir
  │       cmss_map = compute_cmss(feat_rgb, feat_ir)
  │     累积全量 CMSS 值 → cmss_vals [N_total]
  └── gmm.fit(cmss_vals)，更新 μ₁, μ₂, μ₃

【每 batch：前向 + 反传】
  ├── stage = CMSSScheduler.get_stage(epoch)
  ├── λ_align, λ_det = CMSSScheduler.get_loss_weights(epoch)
  │
  ├── [前向]
  │     Î_rgb    = csma(ir_pv)                        # CSMA 前向
  │     outputs  = dino(Î_rgb, input_ids, labels)     # 冻结 DINO 前向
  │     L_det    = _build_det_loss(outputs, cfg)     # 自定义加权，读 CSMAConfig 权重
  │
  ├── [L_align，若 rgb_pixel_values 存在]
  │     feat_rgb = extract_dino_backbone_features(dino, rgb_pv, ...).detach()  # stop_gradient
  │     feat_ir  = extract_dino_backbone_features(dino, Î_rgb, ...)            # 有梯度
  │     cmss_map = compute_cmss(feat_rgb, feat_ir)
  │     mask     = build_cmss_mask(cmss_map, stage, μ₁, μ₂, μ₃)
  │     L_align  = compute_align_loss(feat_ir, feat_rgb, mask)
  │
  ├── [反传]
  │     loss = λ_align * L_align + λ_det * L_det
  │     optimizer.zero_grad(set_to_none=True)
  │     loss.backward()
  │     clip_grad_norm_(csma.parameters(), max_norm=1.0)
  │     optimizer.step()
  │
  └── [日志] SwanLab 记录所有 loss 分项

【每 vis_every 个 epoch：持久化】
  ├── 保存 outputs_csma/ckpt/epoch_{N:04d}.pt
  └── 保存 outputs_csma/vis/epoch_{N:04d}.png（检测框可视化）
```

### 7.2 三阶段渐进课程（GMM-CMSS Curriculum）

三阶段设计来源于 M-SpecGene `GMM_CMSS_SAMPLE.py` 中 `maskratio_bias` 和 `sample_range` 的动态调整逻辑，本方案将其显式化为三个离散阶段：

```
Epoch  1         T/3        2T/3         T
       ├──────────┼───────────┼───────────┤
       │  阶段 A  │  阶段 B   │  阶段 C   │
       │  Easy    │  Mixed    │  Hard     │
       └──────────┴───────────┴───────────┘

阶段 A（Easy）— 关注显著目标
  掩码: CMSS > μ₂  →  mask=1（背景被掩，目标可见）
  保留: 低 CMSS 的目标核心 Patch
  目标: 快速建立 IR→RGB 基础映射（loss 下降迅速）
  权重: λ_align=1.0, λ_det=0.1

阶段 B（Mixed）— 全局上下文
  掩码: GMM 概率分布随机采样，mask_ratio=0.75
  保留: 全局混合区域
  目标: 学习完整场景纹理和上下文
  权重: λ_align=0.5, λ_det=0.5

阶段 C（Hard）— 遮挡增强
  掩码: CMSS < μ₁  →  mask=1（目标核心被掩，背景可见）
  保留: 高 CMSS 的背景 Patch
  目标: 从周边上下文"脑补"目标特征，提升遮挡鲁棒性
  权重: λ_align=0.1, λ_det=1.0
```

**CMSS 值物理语义**（以代码 `GMM_CMSS_SAMPLE.py` 第 85-97 行为准）：

```
CMSS = sqrt[(cosine_sim+1)/2] / (var_rgb × var_ir)，全局 max-归一化

→  低 CMSS（→0）：高方差 + 跨模态差异大  ≡  目标核心区域（热源行人/车辆）
→  高 CMSS（→1）：低方差 + 跨模态一致   ≡  平滑背景区域（天空/路面）
→  μ₁ < μ₂ < μ₃ 对应：目标核心 < 边缘过渡 < 背景
```

### 7.3 损失函数

**L_align — CMSS 引导的跨模态特征蒸馏**

$$\mathcal{L}_{align} = \frac{1}{|\Omega|} \sum_{i \in \Omega} \left\| F_{ir}^{(i)} - \mathrm{sg}\!\left(F_{rgb}^{(i)}\right) \right\|_2^2$$

- $F_{ir}^{(i)}$：`dino_backbone(Î_rgb)` 的第 $i$ 个 Patch，**有梯度，流向 CSMA**
- $F_{rgb}^{(i)}$：`dino_backbone(I_rgb)` 的第 $i$ 个 Patch，`detach()`，**无梯度**
- $\Omega$：当前阶段 GMM-CMSS 掩码中 `mask=0` 的 Patch 集合
- 特征维度：`[B, L, 256]`，提取点为 DINO `input_proj` 投影后

**L_det — 端到端检测损失（代码零改动）**

$$\mathcal{L}_{det} = \mathcal{L}_{ce} + 5.0\,\mathcal{L}_{bbox} + 2.0\,\mathcal{L}_{giou} + 0.1\,\mathcal{L}_{ce}^{enc} + 0.5\,\mathcal{L}_{bbox}^{enc} + 0.5\,\mathcal{L}_{giou}^{enc}$$

**总损失**

$$\mathcal{L}_{total} = \lambda_{align}(t) \cdot \mathcal{L}_{align} + \lambda_{det}(t) \cdot \mathcal{L}_{det}$$

### 7.4 优化器配置

| 项目 | 值 |
|:---|:---:|
| 优化器 | AdamW |
| 学习率 | 1e-4 |
| Weight Decay | 1e-2 |
| 学习率调度 | CosineAnnealingLR（T_max=total_epochs） |
| 梯度裁剪 | clip_grad_norm_(csma, 1.0) |
| Batch Size | 8 |
| 总 Epoch | 100 |
| GMM 更新频率 | 每 10 epoch |

### 7.5 `train_csma.py` 函数索引

| 函数 | 签名摘要 | 说明 |
|:---|:---|:---|
| `_move_labels_to_device` | `(labels: List[Dict], device) → List[Dict]` | labels list 中所有 Tensor 批量迁移设备 |
| `_build_swanlab_logger` | `(enable, project, run_name, config) → Optional` | 按需初始化 SwanLab，失败时安全降级 |
| `_build_det_loss` | `(outputs, cfg: CSMAConfig) → (Tensor, Dict[str, float])` | 6 分量加权求和 L_det，权重读自 `CSMAConfig` |
| `extract_dino_backbone_features` | `(dino, pixel_values, input_ids, attention_mask) → [B, L, 256]` | forward hook 捕获 `model.encoder` 入口特征 |
| `compute_align_loss` | `(feat_ir, feat_rgb, mask) → Tensor` | 仅对 mask=0 的 Patch 计算 MSE；feat_rgb 须已 detach |
| `collect_cmss_values` | `(dino, csma, loader, device, input_ids, attn_mask) → np.ndarray` | 全量 CMSS 收集（torch.no_grad），供 GMM 重拟合 |
| `main` | `() → None` | argparse CLI 入口，完整训练循环 + loss 曲线保存 |

---



### 推理流程

```
单张红外图像推理（无需 RGB，无需 GMM，无需 L_align）

  1. 读取红外图像
     image = Image.open(ir_path).convert("RGB")   # 单通道 → 3 通道复制

  2. 预处理
     inputs = processor(images=image, text=text_prompt, return_tensors="pt")
     ir_pv = inputs["pixel_values"]  [1, 3, H, W]，ImageNet 归一化

  3. CSMA 前向（~0.3ms/frame on A100）
     Î_rgb = csma(ir_pv)             [1, 3, H, W]

  4. Grounding DINO 前向（~50ms/frame）
     outputs = dino(
         pixel_values = Î_rgb,
         input_ids    = tokenized_prompt,
         attention_mask = ...
     )

  5. 后处理（HuggingFace API）
     results = processor.post_process_grounded_object_detection(
         outputs,
         input_ids      = input_ids,
         box_threshold  = 0.3,
         text_threshold = 0.25,
         target_sizes   = [(H_orig, W_orig)]
     )

  6. 输出
     boxes:  [[x1,y1,x2,y2], ...]  原始像素坐标
     scores: [0.87, 0.73, ...]
     labels: ["person", "car", ...]  ← 开放词汇，支持任意 text_prompt
```

### 推理时内存与速度

| 项目 | 值 |
|:---|:---:|
| CSMA 推理显存（batch=1） | ~50MB |
| DINO Tiny 推理显存（batch=1） | ~1.2GB |
| CSMA 前向耗时 | ~0.3ms / frame（A100） |
| DINO 前向耗时 | ~50ms / frame（A100） |
| CSMA 时间开销占比 | <0.6%，可忽略不计 |

### Zero-shot 开放词汇能力

推理时可使用任意文本 prompt，不受训练时 `"person. car."` 的限制：

```python
# 训练类别
text_prompt = "person. car."

# 推理时支持任意扩展（继承 Grounding DINO 原生能力）
text_prompt = "bicycle. motorcycle. truck. traffic light."
text_prompt = "firefighter. ambulance. police officer."
```

---

## 9. 可训练 vs 冻结组件

```
┌─────────────────────────────────────────────────────────────────┐
│                         完整系统组件图                           │
├──────────────────────────┬──────────────────────────────────────┤
│  组件                    │  状态          参数量                  │
├──────────────────────────┼──────────────────────────────────────┤
│                          │                                       │
│  CSMA                    │                                       │
│  ┌─────────────────────┐ │                                       │
│  │ IREncoder（IRE）    │ │  ✅ 可训练      ~0.39M                │
│  │ RPCA                │ │  ✅ 可训练      ~1.0M                 │
│  │  └ rgb_prototypes   │ │  ✅ 可训练      131K（K=512, D=256）  │
│  │ PixelDecoder（PD）  │ │  ✅ 可训练      ~0.72M                │
│  └─────────────────────┘ │                                       │
│                          │                                       │
│  Grounding DINO Tiny     │                                       │
│  ┌─────────────────────┐ │                                       │
│  │ Swin-T Backbone     │ │  ❄ 冻结         28M                  │
│  │ input_proj layers   │ │  ❄ 冻结         ~1M                  │
│  │ BERT Text Encoder   │ │  ❄ 冻结         110M                 │
│  │ Feature Enhancer    │ │  ❄ 冻结         ~4M                  │
│  │   (BiAttention)     │ │                                       │
│  │ Transformer Encoder │ │  ❄ 冻结         ~12M                 │
│  │   (Deformable Attn) │ │                                       │
│  │ Transformer Decoder │ │  ❄ 冻结         ~18M                 │
│  │ Contrastive Head    │ │  ❄ 冻结         ~256K                │
│  │ BBox MLP Head       │ │  ❄ 冻结         ~256K                │
│  └─────────────────────┘ │                                       │
├──────────────────────────┼──────────────────────────────────────┤
│  可训练参数合计           │  ~2.0M（占总系统 1.16%）              │
│  冻结参数合计             │  ~173M                               │
└──────────────────────────┴──────────────────────────────────────┘
```

### 梯度流向验证（继承自 `train_demo.py`）

训练第一个 batch 后执行以下断言（对应 `src/train_demo.py` 第 271–275 行）：

```python
# CSMA 梯度应非零
csma_grad = sum(p.grad.abs().sum() for p in csma.parameters() if p.grad is not None)
assert csma_grad > 0.0, "CSMA 梯度为 0，计算图断裂"

# DINO 参数梯度应全为 None
dino_grads = [p.grad for p in dino.parameters()]
assert all(g is None for g in dino_grads), "冻结的 DINO 不应有梯度"
```

---

## 10. 核心数据结构

### 10.1 CSMAConfig（`src/config.py`）

所有超参数的单一数据源，通过 `dataclass` 管理：

```python
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

LossMode = Literal["align_only", "det_only", "full"]

@dataclass
class CSMAConfig:
    # ── 模型结构 ─────────────────────────────────
    ir_enc_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    num_rgb_prototypes: int       = 512      # RPCA 原型库大小 K
    proto_dim: int                = 256      # 与 DINO hidden_dim 对齐
    num_cross_attn_heads: int     = 8        # 须满足 proto_dim % num_heads == 0
    use_residual: bool            = True

    # ── 训练管线 ─────────────────────────────────
    total_epochs: int             = 100
    batch_size: int               = 8
    lr: float                     = 1e-4
    weight_decay: float           = 1e-2
    grad_clip: float              = 1.0
    loss_mode: LossMode           = "full"

    # ── CMSS / GMM ───────────────────────────────
    mask_ratio: float             = 0.75     # 阶段 B 的掩码比例
    gmm_n_components: int         = 3
    gmm_update_every: int         = 10       # epoch 间隔

    # ── 损失权重（三阶段各一对）─────────────────
    stage_loss_weights: list[tuple[float, float]] = field(
        default_factory=lambda: [(1.0, 0.1), (0.5, 0.5), (0.1, 1.0)]
    )
    det_w_bbox: float             = 5.0
    det_w_giou: float             = 2.0
    det_w_ce_enc: float           = 0.1
    det_w_bbox_enc: float         = 0.5
    det_w_giou_enc: float         = 0.5

    # ── 数据路径 ─────────────────────────────────
    ir_data_root: str             = "train"
    rgb_data_root: str            = "train/rgb"
    text_prompt: str              = "person. car."
    num_workers: int              = 4
    model_id: str                 = "IDEA-Research/grounding-dino-tiny"
    output_dir: str               = "outputs_csma"
    vis_every: int                = 10

    def to_dict(self) -> dict[str, Any]: ...           # asdict 序列化
    def validate(self) -> None: ...                    # 含 proto_dim % num_heads == 0
    @classmethod
    def from_overrides(cls, overrides) -> "CSMAConfig": ...
```

### 10.2 训练 Batch 字典

`FlirPairedDataset` 经 `collate_paired` 后输出的 batch 字典：

```python
batch = {
    "pixel_values":     Tensor[B, 3, H, W],   # IR 图像，ImageNet 归一化
    "pixel_mask":       Tensor[B, H, W],       # DINO 有效区域掩码
    "labels":           List[Dict],            # COCO 检测标注，供 L_det 使用
                        # 每个 dict 含: class_labels, boxes, area, iscrowd
    "rgb_pixel_values": Tensor[B, 3, H, W],   # RGB 配对图像（可选，无则不含此键）
    "image_paths":      List[str],             # 调试用路径
    "orig_sizes":       Tensor[B, 2],          # [H_orig, W_orig]，后处理用
}
```

### 10.3 CMSSScheduler 状态

```python
class CMSSScheduler:
    # 私有属性
    _stage_boundaries: list[int]          # [T//3, T*2//3]，阶段切换 epoch（从 0 计数）
    _gmm_update_every: int                # GMM 重拟合间隔
    _gmm:          GaussianMixture | None # 当前拟合的 GMM 对象
    _sorted_means: np.ndarray | None      # [μ₁, μ₂, μ₃]，μ₁<μ₂<μ₃

    # 只读属性（property）
    @property
    def gmm(self) -> GaussianMixture | None: ...
    @property
    def sorted_means(self) -> tuple[float, float, float]:
        # GMM 未拟合时返回默认值 (0.2, 0.5, 0.8)
        ...
```

### 10.4 检测输出字典

```python
outputs = dino(pixel_values=Î_rgb, ...)
# outputs.loss_dict 包含：
{
    "loss_ce":        Tensor,   # 主分支分类损失
    "loss_bbox":      Tensor,   # 主分支 bbox 回归损失
    "loss_giou":      Tensor,   # 主分支 GIoU 损失
    "loss_ce_enc":    Tensor,   # encoder 分支分类损失
    "loss_bbox_enc":  Tensor,   # encoder 分支 bbox 损失
    "loss_giou_enc":  Tensor,   # encoder 分支 GIoU 损失
}
```

---

## 11. 文件结构

### 11.1 项目完整目录树

```
demo_RGBT_net/
│
├── docs/
│   ├── TD.md                        技术设计文档（决策 + 模块 + 实验计划）
│   └── architecture.md              本文档（统一结构设计）
│
├── src/
│   │
│   ├── ── 现有文件（不修改）─────────────────────────────────────────
│   ├── __init__.py                  包入口
│   ├── dataset.py                   FlirCocoOverfitDataset（COCO格式红外数据集）
│   ├── translator.py                ResidualTranslator（MVP，消融基线保留）
│   ├── train_demo.py                MVP 训练主程序（消融基线保留）
│   ├── infer_vis.py                 MVP 推理可视化
│   │
│   ├── ── 新增文件──────────────────────────────────────────────────
│   ├── config.py           【新增】CSMAConfig dataclass，统一超参数管理
│   ├── csma.py             【新增】CSMA 模型（IREncoder + RPCA + PixelDecoder）
│   ├── cmss_utils.py       【新增】compute_cmss / fit_gmm / build_cmss_mask / CMSSScheduler
│   ├── dataset_paired.py   【新增】FlirPairedDataset（继承 dataset.py，增加 RGB 配对）
│   ├── train_csma.py       【新增】CSMA 训练主程序（三阶段课程 + L_align + L_det）
│   └── infer_csma.py       【新增】CSMA 推理 + CMSS 掩码可视化
│
├── train/                           现有，FLIR COCO 格式红外数据
│   ├── ir/               【建议整理】现有 *.jpg 移入（原来直接在 train/ 下）
│   ├── rgb/              【新增目录】FLIR RGB 配对图像
│   └── _annotations.coco.json       现有，不修改
│
├── outputs_csma/         【新增目录】CSMA 训练输出
│   ├── ckpt/                        模型权重：epoch_{N:04d}.pt, csma_best.pt
│   ├── logs/                        loss 曲线 PNG + SwanLab 日志
│   └── vis/                         检测框可视化：epoch_{N:04d}.png
│
├── GroundingDINO-main/              参考代码，不修改
├── M-SpecGene-main/                 参考代码，不修改
├── requirements.txt                 【更新】新增 scikit-learn
├── download_FLIR.py                 现有，RGB 配对数据下载可复用
└── README.md                        现有
```

### 11.2 新增文件职责一览

| 文件 | 核心类 / 函数 | 职责 | 依赖 |
|:---|:---|:---|:---|
| `src/config.py` | `CSMAConfig` | 统一超参数管理 | 标准库 |
| `src/csma.py` | `IREncoder`, `RGBPrototypeCrossAttention`, `PixelDecoder`, `CSMA` | 核心适配器模型 | `torch`, `config.py` |
| `src/cmss_utils.py` | `compute_cmss`, `fit_gmm`, `build_cmss_mask`, `CMSSScheduler` | CMSS 计算 + GMM 策略 | `torch`, `sklearn`, `numpy` |
| `src/dataset_paired.py` | `FlirPairedDataset`, `collate_paired` | RGB-IR 配对数据加载 | `dataset.py`, `PIL` |
| `src/train_csma.py` | `_move_labels_to_device`, `_build_swanlab_logger`, `_build_det_loss`, `extract_dino_backbone_features`, `compute_align_loss`, `collect_cmss_values`, `main` | CSMA 训练主程序 | 以上所有 |
| `src/infer_csma.py` | `run_inference`, `visualize_cmss_mask`, `save_multi_sample_grid_csma`, `main` | 单流红外推理 + CMSS 可视化 | `csma.py`, `cmss_utils.py`, `infer_vis.py`, `train_csma.py` |

### 11.3 模块依赖关系

```
config.py
    └── csma.py
    └── cmss_utils.py
    └── train_csma.py

dataset.py（现有）
    └── dataset_paired.py
        └── train_csma.py

translator.py（现有，仅消融使用）
    └── train_demo.py（现有，仅消融使用）

infer_vis.py（现有）
    └── infer_csma.py

train_csma.py
    ├── import csma.py
    ├── import cmss_utils.py
    ├── import dataset_paired.py
    └── import config.py
```

### 11.4 新增依赖

| 库 | 版本要求 | 用途 | 是否已安装 |
|:---|:---:|:---|:---:|
| `scikit-learn` | ≥ 1.0 | `GaussianMixture` 拟合 GMM | 否，需新增 |
| `torch` | ≥ 2.0 | 所有张量计算 | 是 |
| `transformers` | ≥ 4.40 | HuggingFace GroundingDinoForObjectDetection | 是 |
| `Pillow` | ≥ 9.0 | 图像读取 | 是 |
| `numpy` | ≥ 1.23 | GMM 数据处理 | 是 |
| `matplotlib` | ≥ 3.5 | loss 曲线、可视化 | 是 |

`requirements.txt` 需新增一行：

```
scikit-learn>=1.0
```

---

*文档版本：v1.0 | 日期：2026-04-27 | 对应 TD.md v2.0*
