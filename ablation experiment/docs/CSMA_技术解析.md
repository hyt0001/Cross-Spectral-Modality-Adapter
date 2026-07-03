# CSMA 项目完整技术解析

> **Cross-Spectral Modality Adapter** — 一个即插即用的轻量级红外模态适配器  
> 使冻结参数的 RGB 视觉大模型（Grounding DINO）具备高精度红外目标检测能力，同时保留原生开放词汇（Zero-shot）能力。

---

## 目录

1. [项目定位与核心论点](#1-项目定位与核心论点)
2. [整体架构与数据流](#2-整体架构与数据流)
3. [配置系统 `config.py`](#3-配置系统-configpy)
4. [核心模型 `csma.py`](#4-核心模型-csmappy)
   - 4.1 [IREncoder — 多尺度红外编码器](#41-irencoder--多尺度红外编码器)
   - 4.2 [RGBPrototypeCrossAttention — RGB 原型交叉注意力](#42-rgbprototypecrossattention--rgb-原型交叉注意力)
   - 4.3 [PixelDecoder — 像素解码器](#43-pixeldecoder--像素解码器)
   - 4.4 [CSMA — 顶层适配器](#44-csma--顶层适配器)
5. [MVP 基线 `translator.py`](#5-mvp-基线-translatorpy)
6. [CMSS + GMM 系统 `cmss_utils.py`](#6-cmss--gmm-系统-cmss_utilspy)
   - 6.1 [CMSS 相似度计算](#61-cmss-相似度计算)
   - 6.2 [GMM 拟合](#62-gmm-拟合)
   - 6.3 [三阶段课程掩码生成](#63-三阶段课程掩码生成)
   - 6.4 [CMSSScheduler 调度器](#64-cmssscheduler-调度器)
7. [训练管线 `train_csma.py`](#7-训练管线-train_csmappy)
   - 7.1 [损失函数体系](#71-损失函数体系)
   - 7.2 [Forward Hook 特征提取](#72-forward-hook-特征提取)
   - 7.3 [GMM 定期重拟合](#73-gmm-定期重拟合)
   - 7.4 [AMP 混合精度训练](#74-amp-混合精度训练)
   - 7.5 [完整训练循环逻辑](#75-完整训练循环逻辑)
   - 7.6 [学习率调度 + 早停 + 权重保存](#76-学习率调度--早停--权重保存)
8. [数据集系统](#8-数据集系统)
   - 8.1 [FlirV1PairedDataset](#81-flirv1paireddataset)
   - 8.2 [FlirADASV2Dataset](#82-fliradasv2dataset)
   - 8.3 [FlirPairedDataset（legacy）](#83-flirpaireddatasetlegacy)
   - 8.4 [collate 函数的设计意图](#84-collate-函数的设计意图)
9. [模块间对接关系](#9-模块间对接关系)
10. [推理流程](#10-推理流程)
11. [实验设置与消融方向](#11-实验设置与消融方向)

---

## 1. 项目定位与核心论点

### 1.1 问题背景

主流视觉大模型（Grounding DINO、CLIP 等）在大规模 RGB 数据上预训练，直接输入红外图像时性能急剧下降。以 FLIR ADAS 数据集为例：

| 方案 | mAP@0.5 | 零样本能力 | 可训练参数 | 训练代价 |
|:---|:---:|:---:|:---:|:---:|
| Grounding DINO-Tiny（直接输入 IR） | ~15% | ✓ | 0 | 0 |
| M-SpecGene（ICCV'25） | ~79% | ✗ | 100M+ | 8×A100×500ep |
| 全量微调 DINO | 高但泛化差 | ✗（灾难性遗忘） | 173M | 大 |
| LoRA (r=8) DINO | 中 | 部分损失 | ~4M | 中 |
| **DINO + CSMA（ours）** | **目标 ≥ 80%** | **✓ 完整** | **~2M** | **~1×A100×100ep** |

### 1.2 核心定位

**"通用视觉大模型 + 轻量适配器 ≥ 领域专有预训练大模型"**

CSMA 不修改 Grounding DINO 的任何参数，而是在其前端插入一个约 2M 参数的"翻译器"：将红外图像转换成 Grounding DINO"能看懂"的伪 RGB 图像，从而利用 DINO 已有的视觉语义理解能力完成红外目标检测。

### 1.3 为什么不直接微调 DINO

- 全量微调（173M 参数）会发生**灾难性遗忘**，丧失零样本能力
- LoRA 修改了 DINO 内部权重，推理时仍需 IR 专属权重，且红外领域迁移不完全
- CSMA 完全独立于 DINO，训练结束后输出的 `Î_rgb` 是标准 3 通道图像，任何 RGB 检测器无需修改即可直接使用

---

## 2. 整体架构与数据流

### 2.1 训练阶段（需要 RGB-IR 配对数据）

```
I_rgb [B,3,H,W] ────────────────────────────────────────┐
                                                          │ no_grad
                                                 冻结 DINO Backbone
                                                 (input_proj 后特征)
                                                          │
                                                    F_rgb [B,L,256]
                                                          │ stop_gradient
I_ir [B,3,H,W] ──► ┌─────────────────────┐              │
                    │     CSMA（可训练）    │              ▼
                    │  ┌──────┐ ┌──────┐  │   ┌──────────────────┐
                    │  │ IRE  │►│ RPCA │  │   │   CMSS 计算      │
                    │  └──────┘ └──────┘  │   │   + GMM 掩码     │ ──► M_cmss [B,L]
                    │    c1,c2 ──┘        │   └──────────────────┘
                    │  ┌──────┐           │          ▲
                    │  │  PD  │◄──feat_map│    F_ir [B,L,256]
                    └──┴──────┴───────────┘          │
                           │ Î_rgb [B,3,H,W]          │
                           │                          │
                           ▼                          │
                  冻结 DINO Backbone ──────────────────┘
                           │ (同时捕获 F_ir，复用 hook)
                           ▼
                  冻结 DINO Transformer
                           │
                           ▼
                  检测输出 (boxes, logits) ──────────► L_det
                  
L_align = MSE( F_ir[M_cmss=0], sg(F_rgb[M_cmss=0]) )
L_total = λ_align · L_align + λ_det · L_det
                           │
                     backward（仅更新 CSMA）
```

**关键 Tensor Shape（默认 img_size=512，batch_size=8）：**

| 张量 | Shape | 含义 |
|:---|:---|:---|
| `I_ir` / `I_rgb` | `[B, 3, H, W]` | ImageNet 归一化后的图像，H/W ≈ 512 |
| `c1` | `[B, 64, H/2, W/2]` | IRE layer1 输出，1/2 分辨率 |
| `c2` | `[B, 128, H/4, W/4]` | IRE layer2 输出，1/4 分辨率 |
| `c3` | `[B, 256, H/8, W/8]` | IRE layer3 输出，1/8 分辨率 |
| `feat` (RPCA 输入) | `[B, L, 256]` | c3 flatten，L = (H/8)×(W/8) |
| `feat` (RPCA 输出) | `[B, L, 256]` | 经原型交叉注意力后的语义特征 |
| `feat_map` | `[B, 256, H/8, W/8]` | reshape 回空间图 |
| `Î_rgb` | `[B, 3, H, W]` | 伪 RGB 图像，tanh 压缩到 [-1,1] |
| `F_rgb` / `F_ir` | `[B, L_total, 256]` | DINO encoder 入口多尺度特征 |
| `M_cmss` | `[B, L_total]` | 0=保留参与对齐，1=掩蔽跳过 |

### 2.2 推理阶段（仅需红外图像）

```
I_ir [B,3,H,W]
    │
    ▼
 CSMA（~2M 参数，已训练好）
    │
    ▼
Î_rgb [B,3,H,W]  ← 标准 3 通道图像，无需任何特殊处理
    │
    ▼
任意冻结 RGB 检测器（Grounding DINO / YOLOv8 / OWL-v2 等）
    │
    ▼
检测框 + 文本标签（zero-shot 开放词汇）
```

---

## 3. 配置系统 `config.py`

所有超参数集中在 `CSMAConfig` 这一个 dataclass 中，无 yaml/json 配置文件。命令行通过 `from_overrides()` 覆盖。

```python
@dataclass
class CSMAConfig:
    # ── 模型结构 ──────────────────────────────────────────────────
    ir_enc_channels: list[int] = [32, 64, 128, 256]
    # IREncoder 各层通道数：[stem输出, layer1, layer2, layer3]
    # layer3 输出通道数必须 == proto_dim（RPCA 要求维度对齐）

    num_rgb_prototypes: int = 512
    # RPCA 中可学习 RGB 原型 token 的数量 K
    # 越大能表达的 RGB 风格越多，但显存占用线性增长

    proto_dim: int = 256
    # RPCA 内部注意力维度，必须 == ir_enc_channels[-1]
    # 也必须能被 num_cross_attn_heads 整除

    num_cross_attn_heads: int = 8
    # RPCA 多头注意力头数

    use_residual: bool = True
    # 是否在 CSMA 末端加 pseudo_rgb += I_ir 残差
    # 开启后：训练初期伪 RGB ≈ 原始红外图，梯度更稳定

    # ── 训练超参 ──────────────────────────────────────────────────
    total_epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-4          # AdamW 初始学习率
    weight_decay: float = 1e-2
    grad_clip: float = 1.0    # 梯度裁剪 L2 范数上限
    loss_mode: LossMode = "full"
    # "full"       → L_det + L_align（需要 RGB 配对，flir_v1 默认）
    # "det_only"   → 仅 L_det（无 RGB 配对，flir_v2 默认）
    # "align_only" → 仅 L_align（调试用）

    # ── CMSS / GMM ────────────────────────────────────────────────
    mask_ratio: float = 0.75
    # Stage B（Mixed）掩码比例：75% 的 patch 被掩蔽

    gmm_n_components: int = 3
    # GMM 分量数，对应目标核心 / 边缘过渡 / 背景三个区域

    gmm_update_every: int = 10
    # 每隔多少个 epoch 重新拟合 GMM（epoch % 10 == 0 时触发）

    gmm_max_batches: int = 100
    # GMM 采样时最多遍历多少 batch；100 batch ≈ 200~800 张图，
    # 统计上足够拟合 3 分量 GMM，且单次拟合 < 2 分钟

    stage_epoch_boundaries: list[int] | None = None
    # 手动指定 [easy_end, mixed_end]；None 时自动用 [T//3, 2T//3]

    hard_max_epochs: int | None = None
    # 限制 Hard 阶段最多 N 个 epoch，缩短伪 RGB 被"硬化"的时间

    stage_loss_weights: list[tuple[float,float]] = [(1.0,0.1),(0.5,0.5),(0.1,1.0)]
    # 三阶段 (λ_align, λ_det)：
    # Stage A（Easy）：对齐为主，检测辅助
    # Stage B（Mixed）：两者并重
    # Stage C（Hard）：检测为主

    # ── L_det 子项权重 ────────────────────────────────────────────
    det_w_bbox: float = 5.0     # L1 box 回归损失权重
    det_w_giou: float = 2.0     # GIoU 损失权重
    det_w_ce_enc: float = 0.1   # encoder 分类损失权重
    det_w_bbox_enc: float = 0.5 # encoder box 回归权重
    det_w_giou_enc: float = 0.5 # encoder GIoU 权重

    # ── 数据 ──────────────────────────────────────────────────────
    ir_data_root: str = "train"
    rgb_data_root: str = "train/rgb"    # 仅 legacy 模式使用
    text_prompt: str = "person. car."   # Grounding DINO 开放词汇 prompt
    num_workers: int = 4
    max_steps_per_epoch: int = -1       # -1=全量；smoke test 设 20

    # ── 内存 / 性能 ───────────────────────────────────────────────
    use_amp: bool = True    # fp16 混合精度，节省 ~50% 显存
    grad_ckpt: bool = False # Grounding DINO 不支持梯度检查点
    img_size: int = 512     # processor shortest_edge 上限；FLIR 原始 640×512

    # ── 路径 ──────────────────────────────────────────────────────
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    output_dir: str = "outputs_csma"
    vis_every: int = 10     # 每隔多少 epoch 保存可视化图和 ckpt
```

**关键约束（由 `validate()` 检查）：**
- `ir_enc_channels[-1]` 必须等于 `proto_dim`（IRE 输出维度对接 RPCA 输入）
- `proto_dim % num_cross_attn_heads == 0`（多头注意力维度整除）
- `stage_epoch_boundaries` 满足 `0 < b0 < b1 <= total_epochs`
- `mask_ratio ∈ [0, 1)`

---

## 4. 核心模型 `csma.py`

整个文件定义了三个子模块类和一个顶层类，共 195 行。

### 4.1 IREncoder — 多尺度红外编码器

**职责：** 将红外图像（ImageNet 归一化的 3 通道复制输入）通过三层步进卷积提取多尺度特征。

**为什么用 CNN 而不是 ViT：** 训练数据量约 10K，ViT 全局注意力在此规模下易过拟合；CNN 的局部性归纳偏置对红外图像热源边缘特征更友好。三层 stride=2 使总下采样 8×，与 DINO `input_proj` 后的多尺度特征尺度对齐。

**网络结构：**

```
输入: I_ir [B, 3, H, W]
  │
  ├── stem:   Conv(3→32, k=3, s=1, p=1) → BN → GELU
  │           输出: [B, 32, H, W]         ← 保持分辨率，提取低级纹理
  │
  ├── layer1: Conv(32→64, k=3, s=2, p=1) → BN → GELU
  │           输出: c1 [B, 64, H/2, W/2]  ← 浅层边缘特征（跳接到 PD）
  │
  ├── layer2: Conv(64→128, k=3, s=2, p=1) → BN → GELU
  │           输出: c2 [B, 128, H/4, W/4] ← 中层语义特征（跳接到 PD）
  │
  └── layer3: Conv(128→256, k=3, s=2, p=1) → BN → GELU
              输出: c3 [B, 256, H/8, W/8] ← 深层语义特征（送入 RPCA）

forward() 返回 (c1, c2, c3)，三个尺度全部保留供 PD 使用
```

**代码要点：**

```python
def forward(self, x):
    x = self.stem(x)
    c1 = self.layer1(x)    # stride=2 下采样
    c2 = self.layer2(c1)   # stride=2 下采样
    c3 = self.layer3(c2)   # stride=2 下采样
    return c1, c2, c3      # 三个尺度均返回，c3 送 RPCA，c1/c2 送 PD
```

**参数量：** ~0.3M

---

### 4.2 RGBPrototypeCrossAttention — RGB 原型交叉注意力

**职责：** 将红外特征"翻译"向 RGB 特征空间靠拢，同时避免推理时依赖真实 RGB 图像。

**核心思想：** 
- 不是让伪 RGB 图在视觉上"看起来像 RGB"，而是让它产生的 DINO 中间特征与真实 RGB 图产生的特征更接近
- 可学习的 `rgb_prototypes` token 在训练中逐渐编码"Grounding DINO 喜欢的 RGB 特征模式"，推理时作为固定的 K/V 提供给红外特征查询

**网络结构：**

```
输入: ir_feat [B, L, 256]       （IRE c3 flatten 后的 patch 序列）

1. q_proj: Linear(256 → 256)   → q [B, L, 256]    ← IR 特征投影为 Query
2. rgb_prototypes: 可学习参数 [K, 256]
   扩展为: kv [B, K, 256]                          ← K 个 RGB 原型作为 Key/Value
3. MultiheadAttention(256, 8 heads, batch_first=True)
   cross_attn(q, kv, kv) → attn_out [B, L, 256]
4. norm1: LayerNorm  +  残差连接:  x = norm1(ir_feat + attn_out)
5. FFN: Linear(256→1024) → GELU → Linear(1024→256)
6. norm2: LayerNorm  +  残差连接:  x = norm2(x + ffn(x))

输出: [B, L, 256]   每个 IR patch 都被"染色"了 RGB 语义
```

**`rgb_prototypes` 的物理意义：**

| 训练阶段 | 学到什么 |
|:---|:---|
| 早期 | 哪些通道对应人/车的热辐射模式 |
| 中期 | 不同 RGB 风格（白天/夜晚/路面）的语义向量分布 |
| 后期 | DINO 最"敏感"的特征激活模式 |

推理时，`rgb_prototypes` 已冻结，红外特征通过注意力从原型库中检索最匹配的 RGB 风格，**完全不需要运行时的真实 RGB 图像**。

**参数量：** ~1.0M（其中 rgb_prototypes 贡献 512×256 = 131K，其余为线性层）

---

### 4.3 PixelDecoder — 像素解码器

**职责：** 将 RPCA 输出的语义特征图（空间分辨率 H/8）逐级上采样恢复至原始分辨率 H，融合 IRE 的跳接特征，最终输出 3 通道伪 RGB 图像。

**为什么需要跳接（skip connection）：** RPCA 处理的是全局语义特征，细节纹理已丢失。从 IRE 引入 c2、c1 的跳接，让 PD 既有"全局 RGB 语义"，也有"局部红外纹理"，生成更自然的伪 RGB。

**网络结构：**

```
输入: feat [B, 256, H/8, W/8]  （RPCA 输出 reshape 回空间图）
      skip_c2 [B, 128, H/4, W/4]  （IRE layer2 跳接）
      skip_c1 [B, 64,  H/2, W/2]  （IRE layer1 跳接）

up3: ConvTranspose2d(256→128, k=4, s=2, p=1) → BN → GELU
     输出: [B, 128, H/4, W/4]
fuse3: cat([..., skip_c2], dim=1) → [B, 256, H/4, W/4]
       Conv2d(256→128, k=1)       → [B, 128, H/4, W/4]
          ↓
up2: ConvTranspose2d(128→64, k=4, s=2, p=1) → BN → GELU
     输出: [B, 64, H/2, W/2]
fuse2: cat([..., skip_c1], dim=1) → [B, 128, H/2, W/2]
       Conv2d(128→64, k=1)        → [B, 64,  H/2, W/2]
          ↓
up1: ConvTranspose2d(64→32, k=4, s=2, p=1) → BN → GELU
     输出: [B, 32, H, W]
          ↓
head: Conv2d(32→3, k=1)  ← 末层权重初始化为 1e-4（近零输出）
      tanh(...)           → Î_rgb [B, 3, H, W]，值域 [-1, 1]
```

**初始化设计：** `head` 层权重初始化为 1e-4，bias 为 0。这使 PD 初始输出约为 0，结合 `use_residual=True`，CSMA 整体输出 ≈ `I_ir`（红外图本身）。训练从接近恒等映射开始，梯度更稳定。

**参数量：** ~0.7M

---

### 4.4 CSMA — 顶层适配器

**职责：** 将 IRE、RPCA、PD 组装为端到端的适配器，提供统一的 `forward()` 接口。

**组装逻辑：**

```python
class CSMA(nn.Module):
    def __init__(self, cfg: CSMAConfig):
        self.ire  = IREncoder(cfg.ir_enc_channels)
        self.rpca = RGBPrototypeCrossAttention(
            ir_dim=256, proto_dim=256, num_heads=8, num_prototypes=512
        )
        self.pd   = PixelDecoder(in_ch=256, skip2_ch=128, skip1_ch=64)

    def forward(self, x):                         # x: [B,3,H,W]
        c1, c2, c3 = self.ire(x)                  # 多尺度特征提取
        B, _, H8, W8 = c3.shape
        feat = c3.flatten(2).transpose(1, 2)       # [B, L, 256]  序列化
        feat = self.rpca(feat)                     # [B, L, 256]  RGB 风格注入
        feat_map = feat.transpose(1,2).reshape(B, 256, H8, W8)  # 还原空间图
        pseudo_rgb = self.pd(feat_map, c2, c1)    # [B, 3, H, W]  解码
        if self.cfg.use_residual:
            pseudo_rgb = pseudo_rgb + x            # 残差：伪RGB ≈ IR + 小残差
        return pseudo_rgb

    def get_intermediate_features(self, x):
        """
        仅返回 RPCA 输出 token [B, L, 256]，供 L_align 计算使用。
        不经过 PD，不加残差。
        注意：训练中实际使用 forward hook 从 DINO 提取 F_ir，
              这个方法主要用于测试 / 特征可视化。
        """
```

**参数约束验证：** `__init__` 时直接检查 `ir_enc_channels[-1] == proto_dim`，违反则立即抛出 `ValueError`，避免维度不匹配在前向传播时才报错。

---

## 5. MVP 基线 `translator.py`

`ResidualTranslator` 是在正式 CSMA 之前的最简 MVP 基线，用于验证"红外→伪RGB→冻结DINO检测"这条路线是否可行。

```python
class ResidualTranslator(nn.Module):
    """3 层 CNN + 残差，约 0.5K 参数"""
    def __init__(self):
        self.conv_block = nn.Sequential(
            Conv2d(3, 16, k=3, p=1),  GELU,
            Conv2d(16, 16, k=3, p=1), GELU,
            Conv2d(16, 3, k=3, p=1),  # 末层权重 1e-4 初始化
        )
    def forward(self, x):
        return x + self.conv_block(x)  # 残差，初期 ≈ 恒等映射
```

**和 CSMA 的关系：**
- `ResidualTranslator` 是 CSMA 的消融基线，对应论文 Ablation A 中"仅残差 CNN（MVP）"一行
- 两者接口完全相同：输入 `[B,3,H,W]` → 输出 `[B,3,H,W]`，即插即用
- `train_demo.py` 使用 `ResidualTranslator`，`train_csma.py` 使用 `CSMA`

---

## 6. CMSS + GMM 系统 `cmss_utils.py`

这是本项目区别于普通知识蒸馏的核心创新，从 M-SpecGene 的 `GMM_CMSS_SAMPLE.py`（第 85–97 行）移植并改造而来。

### 6.1 CMSS 相似度计算

**函数：** `compute_cmss(feat_rgb, feat_ir) → cmss_map [B, L]`

CMSS（Cross-Modal Semantic Similarity）衡量每个 patch 在 RGB 和红外模态之间的**跨模态一致性**：

**公式推导：**

```
Step 1: 余弦相似度归一化到 [0, 1]
        norm_rgb = feat_rgb / ||feat_rgb||        # L2 归一化
        norm_ir  = feat_ir  / ||feat_ir||
        cos_sim  = sum(norm_rgb * norm_ir, dim=-1) # 每个 patch 的余弦值，∈ [-1,1]
        r = sqrt((cos_sim + 1) / 2)               # 映射到 [0,1]

Step 2: 各 patch 的特征方差（衡量局部结构复杂度）
        var_rgb = feat_rgb.var(dim=-1)             # [B, L]
        var_ir  = feat_ir.var(dim=-1)              # [B, L]

Step 3: CMSS 值（越小 → 越重要的目标区域）
        cmss = r / (var_rgb * var_ir + ε)         # 相似度高但方差大 → cmss 低
        cmss = cmss / max(cmss)                    # 全局 max 归一化到 [0, 1]
```

**CMSS 值的语义含义：**

| CMSS 值 | 特征方差 | 跨模态余弦 | 对应区域 | 在 L_align 中的处理 |
|:---:|:---:|:---:|:---:|:---:|
| **低（→0）** | 高（结构复杂） | 低（模态差异大） | 目标核心（人、车） | **重点对齐** |
| **中** | 中 | 中 | 目标边缘 / 过渡区 | 按阶段处理 |
| **高（→1）** | 低（平滑） | 高（模态一致） | 平坦背景 | 可跳过 |

**直觉理解：** 背景（天空、路面）在 RGB 和红外中都很"平"，两者方差都小，余弦相似度高，CMSS 高 → 这些 patch 不需要特别对齐。目标区域（人体热辐射 vs. RGB 纹理）方差高、跨模态差异大，CMSS 低 → 这些 patch 是对齐的重点。

---

### 6.2 GMM 拟合

**函数：** `fit_gmm(cmss_values, n_components=3) → (sorted_means, gmm)`

每隔 `gmm_update_every` 个 epoch，从训练集采样计算全量 CMSS 值，用 3 分量高斯混合模型拟合其分布：

```
cmss_values [N]  ← N = 采样样本数 × patch 数
      │
      ▼
GaussianMixture(n_components=3, covariance_type="full", random_state=42)
      │
      ▼
3 个高斯分量，均值排序后：
    μ₁ < μ₂ < μ₃
    μ₁ ≈ 0.2  ← 低CMSS，对应目标核心
    μ₂ ≈ 0.5  ← 中CMSS，对应边缘过渡
    μ₃ ≈ 0.8  ← 高CMSS，对应背景
```

**为什么用 GMM 而不是固定阈值：**
- 训练早期 CSMA 输出的伪 RGB 质量差，CMSS 分布偏高（大量 patch 难以对齐）
- 训练中期伪 RGB 质量提升，CMSS 分布向低端移动，μ₁/μ₂/μ₃ 都会动态变化
- GMM 自适应地重新估计三个区域的边界，保证掩码策略始终"知道"哪里是目标、哪里是背景
- GMM 每 10 个 epoch 刷新一次（`gmm_update_every=10`），平衡动态性与计算代价

---

### 6.3 三阶段课程掩码生成

**函数：** `build_cmss_mask(cmss_map, stage, μ₁, μ₂, μ₃, mask_ratio, gmm) → mask [B, L]`

`mask=1` 表示该 patch 被掩蔽（**不参与** L_align 计算），`mask=0` 表示保留（**参与**对齐）。

#### Stage A — Easy（早期，epoch ∈ [0, T/3)）

```python
mask = (cmss_map > μ₂).float()
```

**掩蔽高 CMSS 的背景 patch，保留低 CMSS 的目标核心 patch。**

此阶段 CSMA 刚开始训练，伪 RGB 质量差。先让模型专注于最容易对齐的目标核心区域（CMSS 低的部分），避免被噪声较大的背景干扰。损失权重 λ_align=1.0, λ_det=0.1，对齐任务主导。

#### Stage B — Mixed（中期，epoch ∈ [T/3, 2T/3)）

```python
noise_np = gmm.sample(B * L)           # 从 GMM 分布采样噪声
noise = tensor(noise_np).reshape(B, L)
ids = argsort(noise, dim=-1)           # 按噪声排序
len_keep = int(L * (1 - mask_ratio))   # 保留 25% 的 patch
mask = ones(B, L)
mask.scatter_(1, ids[:, :len_keep], 0) # 随机保留部分 patch
```

**按 GMM 概率分布随机采样掩码，混合对齐目标与背景。**

此阶段 CSMA 已有一定能力，开始同时处理不同难度的 patch。GMM 采样使保留的 patch 分布与训练集实际 CMSS 分布一致。损失权重 λ_align=0.5, λ_det=0.5，两者并重。

#### Stage C — Hard（后期，epoch ∈ [2T/3, T)）

```python
mask = (cmss_map < μ₁).float()
```

**掩蔽低 CMSS 的目标核心 patch，保留高 CMSS 的背景 patch。**

乍看反直觉，实际意图：此阶段检测损失 L_det 主导（λ_det=1.0），目标核心的对齐交给 L_det 来驱动。L_align 此时只对背景做轻量约束，避免过度对齐目标核心导致伪 RGB"过拟合"热成像风格而失去泛化性。

**课程渐进的整体逻辑：**
- A 阶段：先学会"把目标核心对齐好"
- B 阶段：扩展到混合区域，同时开始学检测
- C 阶段：主要靠 L_det 驱动，L_align 保护背景不崩塌

---

### 6.4 CMSSScheduler 调度器

`CMSSScheduler` 封装了阶段切换、GMM 管理、损失权重查询三个职责：

```python
class CMSSScheduler:
    def get_stage(self, epoch) → int:
        # 根据 epoch 返回 0(A), 1(B), 2(C)
        if epoch < b_easy:   return 0
        if epoch < b_mixed:  return 1
        return 2

    def should_update_gmm(self, epoch) → bool:
        return epoch % gmm_update_every == 0  # 默认每 10 epoch

    def update_gmm(self, cmss_values):
        self._sorted_means, self._gmm = fit_gmm(cmss_values)

    @property
    def sorted_means(self) → (μ₁, μ₂, μ₃):
        # GMM 未拟合时返回安全默认值 (0.2, 0.5, 0.8)

    def get_loss_weights(self, epoch) → (λ_align, λ_det):
        # 查表 stage_loss_weights[stage]
```

**默认阶段边界（total_epochs=100）：**
- Stage A（Easy）：epoch 0–32
- Stage B（Mixed）：epoch 33–65
- Stage C（Hard）：epoch 66–99

---

## 7. 训练管线 `train_csma.py`

这是整个项目最复杂的文件（960 行），负责将所有模块整合成一个完整的训练循环。

### 7.1 损失函数体系

#### L_det — 检测损失

```
L_det = loss_ce
      + 5.0 × loss_bbox      ← L1 框回归（权重最高，框精度最重要）
      + 2.0 × loss_giou      ← GIoU 框质量
      + 0.1 × loss_ce_enc    ← encoder 分类（权重小，主要走 decoder）
      + 0.5 × loss_bbox_enc  ← encoder 框回归
      + 0.5 × loss_giou_enc  ← encoder GIoU
```

这些损失由 Grounding DINO 内部的匈牙利匹配 + DETR 风格损失计算，不需要手工 NMS，直接从 `outputs.loss_dict` 取出。

#### L_align — 特征对齐损失

```python
def compute_align_loss(feat_ir, feat_rgb, mask):
    """
    L_align = mean( || F_ir[Ω] - sg(F_rgb[Ω]) ||² )
    Ω = { i | mask[i] == 0 }  ← 未被掩蔽的 patch 集合
    sg = stop_gradient（feat_rgb.detach()）
    """
    unmasked = (mask == 0)
    if not unmasked.any():
        return tensor(0.0)  # 全部掩蔽时跳过
    return F.mse_loss(feat_ir[unmasked], feat_rgb[unmasked].detach())
```

**关键设计：`feat_rgb.detach()`**  
RGB 特征来自冻结的 DINO，本身没有梯度。显式 `detach()` 是保险措施，防止任何路径意外反传进 DINO 权重。

#### 总损失

```python
loss = lambda_align * l_align + lambda_det * l_det
```

**lambda_det 最小值保护：**
```python
lambda_det = max(lambda_det, 0.05)
```
Stage A 的 `lambda_det=0.1` 在 FP16 训练中 `l_align` 量级约 1e-5，而 `l_det` 约 1e-1，若 `lambda_det` 继续缩小可能导致梯度被 fp16 舍入为 0，梯度路径断裂。最小值 0.05 保证 `l_det` 始终提供一条可靠的梯度通道。

---

### 7.2 Forward Hook 特征提取

**问题：** 为什么不直接调用 `CSMA.get_intermediate_features()`？

因为 L_align 需要的是 DINO encoder 入口处的特征（经过 backbone + input_proj 投影后），而不是 RPCA 的直接输出。两者维度虽然都是 256，但语义空间不同。

**Hook 机制：**

```python
hook_output = {}

def _hook(module, inp, out):
    # 捕获 DINO encoder 的输入特征
    if inp and isinstance(inp[0], Tensor):
        hook_output["feat"] = inp[0]
    elif hasattr(out, "last_hidden_state_vision"):
        hook_output["feat"] = out.last_hidden_state_vision
    # ... 兼容多版本 transformers API

handle = dino.model.encoder.register_forward_hook(_hook)
try:
    outputs = dino(pixel_values=pseudo_rgb, ...)  # 正常前向
finally:
    handle.remove()  # 确保 hook 被清理，避免内存泄漏
```

**训练中的优化：L_det 和 F_ir 共享一次 DINO 前向**

```python
# 在 L_det 的 DINO 前向中，同时用 hook 捕获 feat_ir
_feat_ir_cache = {}
_hook_handle = dino.model.encoder.register_forward_hook(_capture_hook)
outputs = dino(pixel_values=pseudo_rgb, ...)  # 这次前向同时得到：
_hook_handle.remove()                          # 1. outputs（用于 L_det）
                                               # 2. _feat_ir_cache["feat"]（用于 L_align）

# 额外只做一次 no_grad 的 RGB 前向（获取 F_rgb）
with torch.no_grad():
    feat_rgb = extract_dino_backbone_features(dino, rgb_pv, ...)
```

**这样每个 batch 只需要 2 次 DINO 前向，而非 3 次，节省 33% 时间。**

---

### 7.3 GMM 定期重拟合

`collect_cmss_values()` 函数在训练循环外周期性调用：

```python
def collect_cmss_values(dino, csma, loader, device, ...):
    csma.eval()                          # 关闭 dropout/BN 训练模式
    all_vals = []
    for batch in loader:
        if n_batches >= max_batches: break  # 默认最多 100 batch
        with torch.no_grad():
            pseudo_rgb = csma(ir_pv)
            feat_rgb = extract_dino_backbone_features(dino, rgb_pv, ...)
            feat_ir  = extract_dino_backbone_features(dino, pseudo_rgb, ...)
            cmss_map = compute_cmss(feat_rgb, feat_ir)  # [B, L]
            all_vals.append(cmss_map.cpu().numpy().flatten())
    csma.train()
    return np.concatenate(all_vals)  # 1D array，喂给 fit_gmm
```

**时机：** 训练循环最开始检查 `cmss_sched.should_update_gmm(epoch)`，即 epoch=0,10,20... 时重新拟合。完成后 `torch.cuda.empty_cache()` 释放碎片显存。

---

### 7.4 AMP 混合精度训练

```python
use_amp = cfg.use_amp and device.type == "cuda"
scaler = GradScaler(enabled=use_amp)

# 前向（自动降精度）
with _maybe_autocast(use_amp):
    pseudo_rgb = csma(ir_pv)
    outputs = dino(pixel_values=pseudo_rgb, ...)
    loss = lambda_align * l_align + lambda_det * l_det

# 反向（scaler 防止 fp16 梯度下溢）
optimizer.zero_grad(set_to_none=True)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
nn.utils.clip_grad_norm_(csma.parameters(), cfg.grad_clip)
scaler.step(optimizer)
scaler.update()
```

**AMP 带来的问题与解决：**
- fp16 的最小正数约 6e-8，`l_align` 量级 ~1e-5 可能下溢为 0
- GradScaler 自动放大 loss（初始 scale=65536），backward 后再缩回，防止梯度下溢
- 若 scale 后梯度溢出（NaN/Inf），`scaler.step()` 会跳过该步并减小 scale
- **梯度检查机制**：第一个有效步检查 CSMA 梯度 > 0 且 DINO 梯度 == None

---

### 7.5 完整训练循环逻辑

每个 epoch 的执行顺序：

```
for epoch in range(start_epoch, total_epochs):

  1. GMM 更新（若 epoch % 10 == 0）
     collect_cmss_values() → cmss_sched.update_gmm()
     torch.cuda.empty_cache()

  2. 查询当前 stage + 损失权重
     stage = cmss_sched.get_stage(epoch)
     λ_align, λ_det = cmss_sched.get_loss_weights(epoch)
     μ₁, μ₂, μ₃ = cmss_sched.sorted_means

  3. for batch in loader:
     a. CSMA 前向：pseudo_rgb = csma(ir_pv)
     b. DINO 前向（带 hook）：outputs + feat_ir
     c. 计算 L_det
     d. 若 batch 含 rgb_pixel_values：
        - no_grad DINO 前向获取 feat_rgb
        - compute_cmss(feat_rgb, feat_ir) → cmss_map
        - build_cmss_mask(cmss_map, stage, ...) → mask
        - compute_align_loss(feat_ir, feat_rgb, mask) → l_align
     e. loss = λ_align * l_align + λ_det * l_det
     f. AMP 反传 + 梯度裁剪 + optimizer.step()

  4. lr_scheduler.step()

  5. 写入 latest.pt + latest_meta.json

  6. 若 epoch % vis_every == 0：保存 ckpt + 可视化图

  7. Val 早停评测（若 use_val_early_stop 且在窗口内）
     → 若 mAP@0.5 > 历史最佳 → 保存 best_stage1.pt

  8. 若 stop_after_stage1 且 epoch >= mixed末轮：break
```

---

### 7.6 学习率调度 + 早停 + 权重保存

#### 学习率调度

```python
# 可选 warmup + cosine 衰减
if warmup_epochs > 0:
    warmup_sched = LinearLR(start_factor=0.1, end_factor=1.0, total_iters=warmup)
    cosine_sched = CosineAnnealingLR(T_max=total_epochs - warmup)
    lr_scheduler = SequentialLR([warmup_sched, cosine_sched], milestones=[warmup])
else:
    lr_scheduler = CosineAnnealingLR(T_max=total_epochs)
```

#### 早停（Val Early Stop）

专门针对 **Stage B（Mixed）末段**设置的早停机制，避免训练进入 Hard 阶段后过度优化检测而破坏伪 RGB 质量：

```
val 窗口（默认自动对齐 Mixed 末 8 个 epoch）:
  val_end   = stage1_last_epoch() = mixed_end - 1
  val_start = max(start_epoch, val_end - 8)

每个 epoch 在窗口内：
  → 跑一轮 val mAP@0.5
  → 若 > 历史最佳 → 保存 best_stage1.pt
  → 记录到 val_early_stop.jsonl
```

#### 权重保存体系

| 文件 | 触发时机 | 用途 |
|:---|:---|:---|
| `latest.pt` | 每个 epoch 结束 | 断点续训 |
| `latest_meta.json` | 每个 epoch 结束 | 记录当前 epoch 信息 |
| `epoch_XXXX.pt` | 每 vis_every epoch | 定期快照 |
| `best_stage1.pt` | val mAP 创新高 | Stage B 最佳模型 |
| `emergency_XXXX.pt` | 收到 SIGUSR1 信号 | 紧急保存 |
| `csma_last.pt` | 训练结束 | 最终权重 |

---

## 8. 数据集系统

### 8.1 FlirV1PairedDataset

**文件：** `dataset_flir_v1.py`

**适用场景：** FLIR ADAS v1 数据集，有 RGB-IR 配对，支持完整 `loss_mode=full`。

**目录结构：**
```
FLIR_License/train/
├── thermal_annotations.json    ← COCO 格式标注
├── thermal_8_bit/
│   ├── FLIR_00001.jpeg         ← 热红外图像
│   └── ...
└── RGB/
    ├── FLIR_00001.jpg          ← 对应 RGB 图像（同名，不同扩展名）
    └── ...
```

**配对逻辑：**
```
IR 文件名 stem:  "FLIR_00001"（去掉 .jpeg 扩展名）
RGB 查找路径:   RGB/FLIR_00001.jpg
配对成功率:     > 99%
```

**类别映射：**
- `category_id=1 (person)` → `class_idx=0`
- `category_id=3 (car)` → `class_idx=1`

**每个样本返回：**
- `pixel_values [3, H, W]` — 热红外图，ImageNet 归一化
- `pixel_mask [H, W]` — 有效像素掩码（DINO processor 所需）
- `labels` — DINO 格式标注（cxcywh 归一化坐标）
- `rgb_pixel_values [3, H, W]` — 对应 RGB 图（若不存在则 None）
- `image_path`, `rgb_path` — 调试用路径

**RGB resize 策略：**
```python
if rgb_img.size != (ir_w, ir_h):
    rgb_img = rgb_img.resize((ir_w, ir_h), Image.BILINEAR)
```
强制 RGB 调整到 IR 的原始尺寸，确保两路图像经 processor resize+pad 后形状完全一致，F_rgb 和 F_ir 才能做 shape 对齐的 CMSS 计算。

---

### 8.2 FlirADASV2Dataset

**文件：** `dataset_flir_v2.py`

**适用场景：** FLIR ADAS v2（Roboflow 版本），只有热红外图，无 RGB 配对，强制 `loss_mode=det_only`。

**目录结构：**
```
FLIR_ADAS_v2/train/
├── coco.json      ← 标注文件（非 thermal_annotations.json）
└── data/
    └── *.jpeg     ← 热红外图像
```

**和 v1 的关键差异：**
- 标注文件名不同（`coco.json` vs `thermal_annotations.json`）
- 无 `RGB/` 目录 → 无 `rgb_pixel_values` 字段 → `l_align` 自动跳过
- 类别 ID：`person=1, car=3`（与 v1 相同）

---

### 8.3 FlirPairedDataset（legacy）

**文件：** `dataset_paired.py`

**适用场景：** 早期开发阶段使用的简化版配对数据集，IR 和 RGB 放在两个独立目录，通过 COCO 格式 JSON 对应。

```
train/                          ← ir_data_root
├── _annotations.coco.json
└── *.jpg
train/rgb/                      ← rgb_data_root
└── *.jpg
```

---

### 8.4 collate 函数的设计意图

三个 `collate_fn` 有一个共同的关键设计：

```python
rgb_list = [b.get("rgb_pixel_values") for b in batch]
if all(v is not None for v in rgb_list):
    result["rgb_pixel_values"] = torch.stack(rgb_list)  # 全有才 stack
# 否则 result 中没有 "rgb_pixel_values" 键
```

**为什么是"全有才 stack"而不是"有多少 stack 多少"：**

训练循环中通过 `"rgb_pixel_values" in batch` 判断是否计算 L_align：

```python
if "rgb_pixel_values" in batch and "feat" in _feat_ir_cache:
    rgb_pv = batch["rgb_pixel_values"].to(device)
    ...
    l_align = compute_align_loss(feat_ir, feat_rgb, mask)
```

如果只有部分样本有 RGB，则无法 stack 成统一 batch 张量，且 CMSS 计算要求 `feat_rgb.shape == feat_ir.shape`。**全有才 stack** 保证了 API 的简洁性：要么整个 batch 都计算 L_align，要么整个 batch 跳过 L_align。这避免了在 batch 内做复杂的样本级条件分支。

---

## 9. 模块间对接关系

### 9.1 数据流串联图

```
CSMAConfig
    │
    ├─► IREncoder(ir_enc_channels=[32,64,128,256])
    │         输入:  I_ir [B,3,H,W]
    │         输出:  c1[B,64,H/2,W/2], c2[B,128,H/4,W/4], c3[B,256,H/8,W/8]
    │
    │   c3 ──► flatten(2).transpose(1,2) ──► [B, L, 256]  (L=(H/8)×(W/8))
    │
    ├─► RGBPrototypeCrossAttention(ir_dim=256, proto_dim=256, heads=8, K=512)
    │         输入:  [B, L, 256]
    │         输出:  [B, L, 256]  (RGB 风格注入后的 token 序列)
    │
    │   transpose(1,2).reshape ──► feat_map [B, 256, H/8, W/8]
    │
    ├─► PixelDecoder(in_ch=256, skip2_ch=128, skip1_ch=64)
    │         输入:  feat_map, skip_c2(=c2), skip_c1(=c1)
    │         输出:  [B, 3, H, W]  (tanh 激活)
    │
    │   + I_ir (residual, use_residual=True)
    │         输出:  Î_rgb [B, 3, H, W]
    │
    ├─► 冻结 Grounding DINO
    │         输入:  Î_rgb, text_prompt tokens
    │         输出:  
    │           ├── outputs.loss_dict ──► L_det
    │           └── (via hook) F_ir [B, L_total, 256]
    │
    ├─► 冻结 Grounding DINO (no_grad)
    │         输入:  I_rgb, text_prompt tokens
    │         输出:  F_rgb [B, L_total, 256]
    │
    ├─► compute_cmss(F_rgb, F_ir) ──► cmss_map [B, L_total]
    │
    ├─► build_cmss_mask(cmss_map, stage, μ₁, μ₂, μ₃) ──► mask [B, L_total]
    │
    ├─► compute_align_loss(F_ir, F_rgb, mask) ──► L_align (scalar)
    │
    └─► L_total = λ_align × L_align + λ_det × L_det
              │
              ▼ backward  (仅 CSMA 权重更新)
```

### 9.2 模块依赖关系图

```
config.py
    │
    ├── csma.py         (IREncoder, RPCA, PixelDecoder, CSMA)
    │
    ├── cmss_utils.py   (compute_cmss, fit_gmm, build_cmss_mask, CMSSScheduler)
    │
    ├── translator.py   (ResidualTranslator, MVP 基线)
    │
    ├── dataset_flir_v1.py  ──┐
    ├── dataset_flir_v2.py  ──┤
    ├── dataset_paired.py   ──┤── train_csma.py（主训练程序）
    └── dataset.py          ──┘        │
                                        ├── infer_csma.py（推理）
                                        ├── eval_csma.py（评估）
                                        └── infer_vis.py（可视化）
```

### 9.3 关键维度对齐约束

以下维度必须严格对应，否则前向传播会报形状错误：

| 约束 | 相关参数 | 原因 |
|:---|:---|:---|
| `ir_enc_channels[-1] == proto_dim` | 256 == 256 | IRE c3 直接 flatten 送入 RPCA Query |
| `proto_dim % num_cross_attn_heads == 0` | 256 % 8 == 0 | PyTorch MHA 要求 embed_dim 整除 num_heads |
| `PixelDecoder.in_ch == proto_dim` | 256 | RPCA 输出 reshape 后送入 PD |
| `PixelDecoder.skip2_ch == ir_enc_channels[2]` | 128 | PD fuse3 拼接 c2 |
| `PixelDecoder.skip1_ch == ir_enc_channels[1]` | 64 | PD fuse2 拼接 c1 |
| `F_ir.shape == F_rgb.shape` | [B, L_total, 256] | CMSS 计算和 L_align 计算都要求 shape 完全一致 |

---

## 10. 推理流程

### 10.1 CSMA 推理

推理时只需红外图像，不需要 RGB 图像和 GMM：

```python
# 加载权重
csma = CSMA(cfg).to(device)
csma.load_state_dict(torch.load("outputs_csma/ckpt/csma_last.pt"))
csma.eval()

# 推理单张图
ir_image = Image.open("flir.jpg").convert("RGB")  # 红外图转 3 通道
inputs = processor(images=ir_image, text="person. car.", return_tensors="pt")
ir_pv = inputs["pixel_values"].to(device)

with torch.no_grad():
    pseudo_rgb = csma(ir_pv)                    # [1,3,H,W] 伪 RGB
    outputs = dino(pixel_values=pseudo_rgb, ...) # 送入冻结 DINO 检测
```

### 10.2 即插即用到其他 RGB 检测器

CSMA 输出的 `Î_rgb` 是标准 3 通道 [-1,1] 范围的图像张量（tanh 激活），可直接送入任何 RGB 检测器。只需将 `Î_rgb` 反归一化为 PIL Image，即可用于 YOLOv8、OWL-v2、Faster RCNN 等。

```python
# 保存为标准图像文件
pseudo_rgb_pil = transforms.ToPILImage()(
    (pseudo_rgb[0].cpu().clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]
)
pseudo_rgb_pil.save("pseudo_rgb.jpg")

# 任意 RGB 检测器无缝接入
yolo_model.predict("pseudo_rgb.jpg")
```

### 10.3 推理额外延迟

CSMA（~2M 参数）对单帧图像约增加 **+0.3ms** 推理延迟（相比直接将红外图送入 DINO），可忽略不计。

---

## 11. 实验设置与消融方向

### 11.1 消融 A — CSMA 子模块必要性

| 配置 | IRE | RPCA | PD 跳接 | 参数量 |
|:---|:---:|:---:|:---:|:---:|
| 仅残差 CNN（MVP，`translator.py`） | ✗ | ✗ | ✗ | ~0.5K |
| + IRE only | ✓ | ✗ | ✗ | ~0.39M |
| + IRE + 简单线性投影 | ✓ | Linear | ✗ | ~0.5M |
| **完整 CSMA** | ✓ | ✓ | ✓ | ~2.0M |

### 11.2 消融 B — RPCA 原型设计

| RPCA 变体 | K/V 来源 | 推理需 RGB？ |
|:---|:---|:---:|
| 双流实时（M-SpecGene 原版） | 运行时 RGB 特征 | ✓ |
| 均值原型（训练集 RGB 特征均值） | 离线计算，frozen | ✗ |
| **可学习原型库（ours, K=512）** | 训练时反传，frozen@推理 | ✗ |

### 11.3 消融 C — GMM-CMSS 策略

| 配置 | λ 策略 | 掩码策略 |
|:---|:---:|:---:|
| 随机掩码，固定权重 | 0.5/0.5 | random |
| 固定 CMSS 阈值 μ₂ | 0.5/0.5 | fixed |
| GMM-CMSS 单阶段 B | 0.5/0.5 | GMM |
| **三阶段渐进 A→B→C（完整）** | **动态** | **GMM** |

### 11.4 运行命令

```bash
# 完整训练（FLIR v1，有 RGB 配对）
bash scripts/01_train.sh

# 仅检测损失（FLIR v2，无 RGB 配对）
python -m src.train_csma --dataset flir_v2 --data-root FLIR_ADAS_v2/train

# Smoke test（2 epoch，每 epoch 20 步）
bash scripts/00_smoke_test.sh

# 评估
python -m src.eval_csma --dataset flir_v1 --data-root FLIR_License/val --ckpt outputs_csma/ckpt/csma_last.pt

# 推理可视化
python -m src.infer_csma --ckpt outputs_csma/ckpt/csma_last.pt --ir-dir FLIR_License/val/thermal_8_bit
```

---

## 附录：文件清单

| 文件 | 作用 |
|:---|:---|
| `src/config.py` | 全局超参数配置（CSMAConfig dataclass） |
| `src/csma.py` | 核心模型：IREncoder + RPCA + PixelDecoder + CSMA |
| `src/cmss_utils.py` | CMSS 计算 + GMM 拟合 + 课程掩码 + 调度器 |
| `src/translator.py` | MVP 基线 ResidualTranslator（~0.5K 参数） |
| `src/train_csma.py` | 主训练程序（完整管线） |
| `src/train_demo.py` | MVP 训练程序（使用 ResidualTranslator） |
| `src/dataset_flir_v1.py` | FLIR ADAS v1 RGB-IR 配对数据集 |
| `src/dataset_flir_v2.py` | FLIR ADAS v2 纯热红外数据集 |
| `src/dataset_paired.py` | legacy 配对数据集 |
| `src/dataset.py` | COCO demo 数据集（过拟合测试） |
| `src/eval_csma.py` | mAP 评估程序 |
| `src/infer_csma.py` | 推理 + 可视化程序 |
| `src/infer_vis.py` | MVP 可视化工具 |
| `docs/architecture.md` | 官方架构文档（详细） |
| `docs/TD.md` | 技术决策记录（TD-01 ~ TD-06） |
| `docs/ablation_experiments.md` | 消融实验设计文档 |

---

*文档生成于 2026-05-21，基于 Cross-Spectral-Modality-Adapter 代码库全量阅读。*
