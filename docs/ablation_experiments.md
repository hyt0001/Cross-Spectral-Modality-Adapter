# CSMA 消融与对比实验手册

| 字段 | 内容 |
|:---|:---|
| 版本 | v2.0 |
| 更新日期 | 2026-05-20 |
| 前置条件 | 主实验完成：FLIR **mAP@0.5 > M-SpecGene（~79.3）** |
| 关联文档 | [`TD.md`](TD.md) §3、`[architecture.md](architecture.md)`、`[reference-paper.md](../reference-paper.md)` |
| 快速脚本 | [`scripts/04_ablation.sh`](../scripts/04_ablation.sh)、[`scripts/02_eval.sh`](../scripts/02_eval.sh) |

---

## 1. 实验要证明什么

论文主张可归纳为四条 **Claim**，每条对应一组实验，避免「堆表无叙事」。

| Claim | 论文表述 | 必做实验 |
|:---|:---|:---|
| **C1 效能** | ~2M 参数的 CSMA + 冻结 GDINO，在 ~10K 配对数据上达到或超过 M-SpecGene | Table 1 主对比 |
| **C2 结构** | IRE / RPCA / PD 与 GMM-CMSS 各自必要，且 RPCA 可学习原型不可替代简单线性层 | Exp-1、Exp-2 |
| **C3 即插即用** | 推理输出标准伪 RGB，可接**未参与训练**的其它检测器 | Exp-4 |
| **C4 开放词汇** | 训练仅 `person/car`，测试未见类仍可用文本 prompt 检测 | Exp-5 |

**不必换 DINO 为其它模型再训一整条管线**；C3 通过在**固定 CSMA 权重**下更换推理检测器即可验证（见 §5）。

---

## 2. 评测协议（全实验统一）

### 2.1 数据与划分

| 数据集 | 角色 | 说明 |
|:---|:---|:---|
| **FLIR ADAS v2** | 训练 + 主评测 | 官方 train/val，配对 RGB~9.7K |
| KAIST / LLVIP / M3FD | 补充泛化 | 可选，附录或 rebuttal |

### 2.2 指标

| 指标 | 用途 |
|:---|:---|
| **mAP@0.5** | 主文核心指标 |
| mAP@0.5:0.95 | 严格 COCO |
| Zero-shot AP@0.5（按类） | C4 |
| MR@FPPI=0.1 | KAIST 行人 |
| 可训练参数量 / 训练 GPU·天 / 推理延迟 | 效率表 |

### 2.3 训练默认值（消融与主实验对齐）

| 项 | 值 |
|:---|:---|
| 检测器 | `IDEA-Research/grounding-dino-tiny`，**全程冻结** |
| CSMA 可训练参数 | ~2.0M（约 1.16%） |
| 优化 | AdamW，lr=1e-4，wd=1e-2，Cosine，100 epoch（快速消融可用 30） |
| Batch | 8 |
| 训练 prompt | `"person. car."` |
| 快速消融输出目录 | `outputs_ablation/<exp_id>/` |

---

## 3. Table 1 — 主对比（论文主表）

> **审稿底线**：必须包含同任务、同基准的近期竞品，不能只有 M-SpecGene + Ours。

| ID | 方法 | 可训练参数 | 训练数据 | mAP@0.5 | mAP@0.5:0.95 | Zero-shot |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| T0 | M-SpecGene (ICCV'25) | 100M+ | 550K 对 | ~79.3 | ~44.7 | ✗ |
| T1 | GDINO-Tiny，IR 直输（zero-shot） | 0 | 0 | ~15 | — | ✓ |
| T2 | GDINO-Tiny，**全量微调** | 173M | ~10K | | | ✗ |
| T3 | GDINO-Tiny，**LoRA r=4** | ~4M | ~10K | | | 待测 |
| T4 | *From Words to Wavelengths* (2025) | 轻量适配 | ~10K | 引文/复现 | — | ✓ |
| T5 | GDINO + ResidualTranslator (MVP) | ~0.5K | ~10K | | — | ✓ |
| T6 | GDINO + CSMA（无 GMM，仅 det+align） | ~2M | ~10K | | | ✓ |
| **T7** | **GDINO + CSMA（完整，Ours）** | **~2M** | **~10K** | **主实验** | **主实验** | **✓** |

**文献锚点**（`reference-paper.md`）：

- **M-SpecGene**：CMSS/GMM 与 RGBT 预训练；检测 mAP50 84.8（ViT-B 设置，引用时注明骨干差异）。
- **From Words to Wavelengths**：GDINO/YOLO-World + 轻量多光谱适配，FLIR/M3FD；全监督与 few-shot 均有报告。
- **Grounding DINO**：开放词汇基线；IR 直输 ~15% mAP@0.5 为项目内测参考。

**T2/T3 目的**：回应「为何不直接微调 DINO」——全量微调预期损害 zero-shot；LoRA 为参数量更接近的强基线。

---

## 4. 消融实验（按 Claim 组织）

### Exp-1 — CSMA 模块堆叠（Claim C2：结构）

逐步增加模块；**每行写明参数量**，并增加「线性替代 RPCA」对照。

| Exp ID | IRE | RPCA | PD 跳接 | 说明 | 参数量 | mAP@0.5 |
|:---|:---:|:---:|:---:|:---|:---:|:---:|
| 1-A0 | ✗ | ✗ | ✗ | MVP `ResidualTranslator` | ~0.5K | |
| 1-A1 | ✓ | ✗ | ✗ | 仅多尺度编码 | ~0.39M | |
| 1-A2 | ✓ | Linear | ✗ | 用 `Linear(256→256)` 代替 RPCA | ~0.5M | |
| 1-A3 | ✓ | ✓ | ✗ | 无 U-Net 跳接 | ~1.4M | |
| **1-A4** | ✓ | ✓ | ✓ | **完整 CSMA（=T7）** | **~2.0M** | **主实验** |

**判读**：

- 1-A4 − 1-A2 显著 → RPCA 非「多加一层线性」。
- 1-A4 − 1-A3 显著 → PD 跳接对细节/定位有必要。

**实现**：`src/csma.py` 子模块开关或独立 config；1-A0 走 `train_demo.py`。

---

### Exp-2 — RPCA 原型机制（Claim C2：核心创新）

| Exp ID | K/V 来源 | 推理是否需要 RGB | mAP@0.5 |
|:---|:---|:---:|:---:|
| 2-B0 | 运行时 RGB 特征（M-SpecGene 双流） | **是** | |
| 2-B1 | 随机初始化原型，训练时不更新 | 否 | |
| 2-B2 | 训练集 RGB 特征全局均值，frozen | 否 | |
| **2-B3** | **可学习原型库 K=512（Ours）** | **否** | **主实验** |

**判读**：2-B3 > 2-B1、2-B2 → 原型学到结构化语义；2-B3 ≈ 2-B0 且推理免 RGB → **离线原型替代运行时双流**。

**论文图**：对 512 个原型做 PCA/t-SNE（Fig-Proto，见 §7）。

---

### Exp-3 — 损失与 GMM-CMSS 课程（Claim C2：训练策略）

| Exp ID | `loss_mode` | 掩码 | 课程顺序 | mAP@0.5 | 脚本/状态 |
|:---|:---|:---|:---|:---:|:---|
| 3-C0 | `det_only` | — | — | | `04_ablation` ✅ |
| 3-C1 | `align_only` | GMM | A→B→C | | `04_ablation` ✅ |
| 3-C2 | `full` | **random** | — | | 需 `--mask-mode random` |
| 3-C3 | `full` | 固定 CMSS 阈值 | — | | 待实现 |
| 3-C4 | `full` | GMM | **仅阶段 B** | | 待实现 |
| 3-C5 | `full` | GMM | **逆序 C→B→A** | | 待实现 |
| **3-C6** | **`full`** | **GMM** | **A→B→C** | **主实验** | `04_ablation` full_gmm ✅ |

**三阶段定义**（与 `architecture.md` §7.2 一致）：

```
Epoch:  |---- 阶段 A ----|---- 阶段 B ----|---- 阶段 C ----|
掩码:   掩背景(高CMSS)    GMM随机75%       掩目标(低CMSS)
λ:      align:det=1:0.1   0.5:0.5          0.1:1.0
```

**判读**：3-C6 − 3-C0 → 对齐损失必要；3-C6 − 3-C2 → CMSS 优于随机掩码；3-C6 − 3-C5 → 课程顺序合理。

---

### Exp-4 — 即插即用：跨检测器推理（Claim C3）

**固定**主实验 `csma_best.pt`，**不重训** CSMA，只换推理端检测器。

| Exp ID | 输入 | 推理模型 | CSMA 训练时是否见过该模型 | mAP@0.5 |
|:---|:---|:---|:---:|:---:|
| 4-D0 | IR | YOLOv8-m | — | |
| 4-D1 | IR | GDINO-Tiny | — | ~15 |
| **4-D2** | **伪 RGB** | **GDINO-Tiny** | 是（蒸馏 teacher） | **主实验** |
| 4-D3 | 伪 RGB | GDINO-Base | **否** | |
| 4-D4 | 伪 RGB | YOLOv8-m | **否** | |

**操作流程**：

```text
1. Î_rgb = csma(ir_pv)   # 与主实验相同预处理
2. 将 Î_rgb 送入目标检测器（冻结权重）
3. 同一 COCO 评测脚本统计 mAP
```

**判读**：4-D4 ≫ 4-D0 → 伪 RGB 具跨架构可迁移性；4-D3 与 4-D2 接近 → 不绑定 Tiny 尺度。

**工作量**：D3/D4 为推理侧实验，**1 天内可完成**，建议 P0。

---

### Exp-5 — Zero-shot 开放词汇（Claim C4）

训练类别：`person`, `car`。测试换 prompt，**不重训**。

| Exp ID | 测试 prompt | GDINO+RGB（上界） | GDINO+CSMA+IR | M-SpecGene |
|:---|:---|:---:|:---:|:---:|
| 5-Z1 | `"bicycle."` | | | 0 |
| 5-Z2 | `"traffic light."` | | | 0 |
| 5-Z3 | `"truck."` | | | 0 |

**判读**：CSMA 曲线接近 RGB 上界，且远高于 M-SpecGene（无开放词汇）→ C4 成立。

---

### Exp-6 — 数据效率（强化 C1 叙事）

在 FLIR 配对训练集上按比例子采样（固定 seed），其余超参与 T7 一致。

| 子集规模 | CSMA mAP@0.5 | 备注 |
|:---:|:---:|:---|
| 100 | | M-SpecGene 难以在此规模预训练 |
| 500 | | |
| 1K | | |
| 5K | | |
| 10K | **主实验** | 全量 |

**预期**：1K–5K 时 CSMA 已具竞争力 → 「数据效率」对比 M-SpecGene 550K。

---

### Exp-7 — 超参数与实现细节（附录 / rebuttal）

#### 7a 原型数量 K

| K | mAP@0.5 |
|:---:|:---:|
| 64 | |
| 128 | |
| 256 | |
| **512** | **默认** |
| 1024 | |

配置：`CSMAConfig.num_rgb_prototypes`

#### 7b L_align 特征层

| 提取位置 | mAP@0.5 | 状态 |
|:---|:---:|:---:|
| 像素级 MSE | | 待做 |
| Swin 中间层 | | hook |
| **input_proj 后 256 维** | **主实验** | ✅ 已实现 |
| Encoder 输出 | | hook |

#### 7c 其它（可选）

| 项 | 变体 |
|:---|:---|
| IRE 骨干 | CNN（默认）/ MobileNetV3-S / DeiT-Tiny |
| DINO 策略 | 冻结（默认）/ LoRA（见 T3） |
| 跨数据集 | KAIST MR、LLVIP/M3FD mAP |

---

## 5. 是否必须「换 DINO」？

| 问题 | 结论 |
|:---|:---|
| 要不要用 YOLO-World 等再训一套 CSMA？ | **不要**（成本高，且偏离「轻量适配冻结 VLM」故事线） |
| 怎样证明 plug-and-play？ | **Exp-4**：同一 CSMA，伪 RGB → GDINO-Base / YOLOv8-m |
| 和 Words to Wavelengths 关系？ | 同属 VLM+轻量适配；用 **Table 1 T4** 数字对比，而非再换 backbone 训练 |

**两层 plug-and-play**：

1. **训练层**：`L_align` 对齐 GDINO 特征空间（方法绑定 teacher，可接受）。
2. **推理层**：输出标准 `[B,3,H,W]` 伪 RGB → **任意 RGB 检测器**（Exp-4 证明）。

---

## 6. 效率对比表（主文或附录）

| 方法 | 可训练参数 | 训练数据 | 训练算力（估） | 推理额外延迟 | Zero-shot |
|:---|:---:|:---:|:---:|:---:|:---:|
| M-SpecGene | 100M+ | 550K | 8×A100×500ep | — | ✗ |
| 全量微调 GDINO-T | 173M | 10K | ~1×A100×100ep | — | ✗ |
| LoRA GDINO-T | ~4M | 10K | ~1×A100×100ep | — | 部分 |
| **CSMA (Ours)** | **~2M** | **10K** | **~1×A100×100ep** | **~+0.3ms** | **✓** |

---

## 7. 可视化清单

| 图 | 内容 | 工具 |
|:---|:---|:---|
| Fig-1 | IR \| 伪 RGB \| GT RGB 对照 | `infer_csma.py` |
| Fig-2 | 三阶段 CMSS 掩码热力图 | `visualize_cmss_mask` |
| Fig-3 | 512 原型 PCA/t-SNE | notebook（待写） |
| Fig-4 | Zero-shot 检测框对比 | `infer_csma.py` + 扩展 prompt |
| Fig-5 | 失败案例 | 人工筛选 |

---

## 8. 执行计划

### 8.1 优先级

```text
P0（无则易被拒）
  Table 1: T4 Words to Wavelengths、T2/T3 微调基线
  Exp-2: 2-B1/B2 vs 2-B3
  Exp-3: 3-C0/C2/C6（补 random 掩码）
  Exp-4: 4-D4 伪RGB→YOLOv8-m
  Exp-5: Zero-shot 三类

P1（Spotlight 叙事）
  Exp-1: 全表含 1-A2
  Exp-3: 3-C5 逆序课程
  Exp-6: 数据效率曲线
  Fig-3 原型可视化
  §6 效率表

P2（附录）
  Exp-7 全部
  跨数据集 KAIST/LLVIP/M3FD
```

### 8.2 脚本与命令

| 任务 | 命令 |
|:---|:---|
| 快速损失消融（30 epoch） | `bash scripts/04_ablation.sh` |
| 单组评测 | `CKPT=outputs_ablation/<id>/ckpt/csma_last.pt bash scripts/02_eval.sh` |
| 主实验训练 | `bash scripts/01_train.sh` |

**`04_ablation.sh` 当前覆盖**：3-C0、3-C1、3-C6（`det_only` / `align_only` / `full_gmm`）。3-C2 待 `--mask-mode random` 后解注释第 84 行。

### 8.3 待实现功能

| 功能 | CLI / 模块 | 解锁实验 |
|:---|:---|:---|
| `--mask-mode {gmm,fixed,random}` | `train_csma.py`, `cmss_utils.py` | 3-C2 |
| `--curriculum {abc,cba,single_b}` | `CMSSScheduler` | 3-C4, 3-C5 |
| CSMA 子模块 ablation 开关 | `CSMAConfig`, `csma.py` | Exp-1 |
| RPCA `prototype_mode` | `RGBPrototypeCrossAttention` | Exp-2 B1/B2 |
| 跨检测器 eval | 新脚本 `scripts/05_eval_cross_detector.sh` | Exp-4 |

---

## 9. 结果记录规范

每组实验在 `outputs_ablation/<exp_id>/` 保存：

```yaml
exp_id: 3-C2_full_random
date: 2026-05-20
config:
  loss_mode: full
  mask_mode: random
  epochs: 30
  seed: 42
checkpoint: outputs_ablation/full_random/ckpt/csma_last.pt
metrics:
  mAP_50: null      # 填数
  mAP_50_95: null
  notes: "相对 3-C6 的 ΔmAP = ..."
```

论文表格：将上表「mAP@0.5」列从空填为实测值；主实验行加粗 **T7 / 1-A4 / 2-B3 / 3-C6**。

---

## 10. 与 TD.md 对照

| 本文档 | `TD.md` §3.4–3.5 |
|:---|:---|
| Table 1 | §3.3 扩展（竞品 + LoRA） |
| Exp-1 | 原「消融一」+ 1-A2 |
| Exp-3 | 原「消融二」+ 课程/随机掩码 |
| Exp-7a/b | 原「消融三、四」 |
| Exp-5 | §3.5 Zero-shot |
| Exp-2, 4, 6 | **新增** |

---

## 附录：审稿人问题速查

| 审稿意见 | 回复实验 |
|:---|:---|
| 只在 DINO 上有效 | Exp-4（YOLOv8-m / GDINO-Base） |
| 和 Words to Wavelengths 重复 | Table 1 T4 + 方法差异（CMSS 蒸馏 + 原型 RPCA） |
| 原型是否学到语义 | Exp-2 + Fig-3 |
| 为何不全量微调 | T2 vs T7 + Exp-5 |
| GMM 课程是否随意 | 3-C5 vs 3-C6 |
| 数据不公平 | Exp-6 + Table 1 训练数据列 |

---

*维护说明：主实验或代码能力更新后，同步修订 §8.3「待实现」与表中空值。*
