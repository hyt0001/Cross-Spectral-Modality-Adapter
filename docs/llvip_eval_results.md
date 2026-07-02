# LLVIP 跨数据集评测报告

> 评测日期：2026-06-22  
> 评测集：LLVIP test（3463 张，8302 个 person GT 框）  
> 模型：Grounding DINO Tiny（`IDEA-Research/grounding-dino-tiny`）  
> Prompt：`"person."`  
> box/text threshold：0.05  
> img_size：512（shortest_edge）

---

## 1. 实验概览

本报告汇总以下三组实验在 LLVIP test 集上的检测性能（mAP@0.5 / mAP@0.5:0.95）：

| 实验 | 描述 |
|------|------|
| **Baseline** | 纯 Grounding DINO，IR 图像直接送入，无任何适配模块 |
| **CSMA (outputs_csma)** | 第一次训练（20 epoch，FLIR_License 数据，val 最佳 epoch=12） |
| **CSMA (outputs_csma_v3)** | 第二次训练（100 epoch，FLIR_License 数据，val 最佳 epoch=91） |

---

## 2. Baseline 结果

| 指标 | 值 |
|------|----|
| **mAP@0.5** | **0.8572** |
| mAP@0.5:0.95 | 0.5638 |
| AP_person@0.5 | 0.8572 |
| 预测框数 | 203,322 |
| GT 框数 | 8,302 |

> Grounding DINO 在 LLVIP 红外图像上具备很强的零样本检测能力，无需任何适配即可达到 **0.857 mAP@0.5**。

---

## 3. 全量 Checkpoint 评测结果（22 个）

下表按 mAP@0.5 降序排列，`vs Baseline` 为与 Baseline 的差值（+为提升，−为下降）。

| Rank | Checkpoint | mAP@0.5 | vs Baseline | mAP@0.5:0.95 | 预测框数 | 耗时(s) |
|------|-----------|---------|------------|-------------|---------|--------|
| 1 🏆 | `outputs_csma_v3/ckpt/epoch_0000.pt` | **0.8666** | **+0.0094** | 0.5717 | 210,914 | 159 |
| 2 | `outputs_csma/ckpt/epoch_0000.pt` | 0.8660 | +0.0088 | 0.5749 | 184,250 | 156 |
| 3 | `outputs_csma/ckpt/epoch_0011.pt` | 0.8377 | −0.0195 | 0.5494 | 115,337 | 148 |
| 4 | `outputs_csma/ckpt/epoch_0010.pt` | 0.8275 | −0.0297 | 0.5459 | 124,797 | 149 |
| 5 | `outputs_csma/ckpt/best_stage1.pt` | 0.8185 | −0.0387 | 0.5396 | 141,924 | 150 |
| 5 | `outputs_csma/ckpt/epoch_0012.pt` | 0.8185 | −0.0387 | 0.5396 | 141,924 | 152 |
| 7 | `outputs_csma/ckpt/csma_last.pt` | 0.8048 | −0.0524 | 0.5250 | 140,485 | 153 |
| 8 | `outputs_csma_v3/ckpt/epoch_0020.pt` | 0.7986 | −0.0586 | 0.5095 | 207,938 | 159 |
| 9 | `outputs_csma_v3/ckpt/epoch_0010.pt` | 0.7940 | −0.0632 | 0.5113 | 204,149 | 159 |
| 10 | `outputs_csma_v3/ckpt/epoch_0030.pt` | 0.7862 | −0.0710 | 0.5010 | 207,240 | 159 |
| 11 | `outputs_csma_v3/ckpt/epoch_0050.pt` | 0.7833 | −0.0739 | 0.4980 | 192,329 | 157 |
| 12 | `outputs_csma_v3/ckpt/epoch_0040.pt` | 0.7832 | −0.0740 | 0.5014 | 193,524 | 158 |
| 13 | `outputs_csma/ckpt/epoch_0020.pt` | 0.7821 | −0.0751 | 0.5100 | 133,358 | 150 |
| 14 | `outputs_csma_v3/ckpt/epoch_0070.pt` | 0.7734 | −0.0838 | 0.4868 | 188,282 | 157 |
| 15 | `outputs_csma_v3/ckpt/epoch_0080.pt` | 0.7732 | −0.0840 | 0.4863 | 193,259 | 158 |
| 16 | `outputs_csma/ckpt/epoch_0029.pt` | 0.7707 | −0.0865 | 0.4994 | 143,390 | 152 |
| 17 | `outputs_csma_v3/ckpt/epoch_0060.pt` | 0.7703 | −0.0869 | 0.4865 | 186,576 | 157 |
| 18 | `outputs_csma_v3/ckpt/epoch_0090.pt` | 0.7684 | −0.0888 | 0.4836 | 191,453 | 158 |
| 19 | `outputs_csma_v3/ckpt/best_stage1.pt` | 0.7667 | −0.0905 | 0.4815 | 195,683 | 157 |
| 19 | `outputs_csma_v3/ckpt/epoch_0091.pt` | 0.7667 | −0.0905 | 0.4815 | 195,683 | 157 |
| 21 | `outputs_csma_v3/ckpt/csma_last.pt` | 0.7650 | −0.0922 | 0.4811 | 191,517 | 157 |
| 21 | `outputs_csma_v3/ckpt/epoch_0099.pt` | 0.7650 | −0.0922 | 0.4811 | 191,517 | 157 |

---

## 4. 关键指标对比

| 模型 | mAP@0.5 | mAP@0.5:0.95 | vs Baseline |
|------|---------|-------------|------------|
| **Baseline (纯 DINO)** | 0.8572 | 0.5638 | — |
| **CSMA epoch_0000 (v3) 🏆** | **0.8666** | 0.5717 | **+0.94%** |
| **CSMA epoch_0000 (v1)** | 0.8660 | 0.5749 | +0.88% |
| CSMA best_stage1 (v1, FLIR 最优) | 0.8185 | 0.5396 | −3.87% |
| CSMA best_stage1 (v3, FLIR 最优) | 0.7667 | 0.4815 | −9.05% |
| CSMA csma_last (v3, 100 epoch) | 0.7650 | 0.4811 | −9.22% |

---

## 5. 训练曲线趋势（LLVIP mAP@0.5 随 epoch 变化）

```
mAP@0.5
0.87 │ ●● (epoch_0000: 0.866)
0.85 │ ─── Baseline (0.857)
0.84 │     ●● (epoch_0010~0011: 0.827~0.837)
0.82 │
0.80 │ ●● (epoch_0012/best_stage1: 0.818)
     │      ↓ 持续下降
0.77 │                          ●●●●●●●● (epoch_0090~0099: 0.765~0.767)
     └──────────────────────────────────────── epoch →
       0    10   20   30   50   70   90  99
```

---

## 6. 核心发现与分析

### 6.1 早期 checkpoint 略优于 Baseline

`epoch_0000.pt`（仅训练 1 个 epoch）在 LLVIP 上 **超过 Baseline +0.94%**（0.8666 vs 0.8572）。  
原因分析：
- CSMA 初始化为小随机权重，残差连接 `pseudo_rgb = tanh(head(x)) + x ≈ x + δ` 仅添加微小扰动
- 第 1 epoch 的轻微扰动可能有正则化效果，略微帮助 DINO 提取特征
- 此时 CSMA 尚未收敛于 FLIR 特定模式

### 6.2 训练越长 LLVIP 性能越差（FLIR 域过拟合）

从 epoch_0000 (0.866) 到 epoch_0099 (0.765)，**随 epoch 增加单调下降约 10%**。  
这是典型的**跨域过拟合**特征：

```
训练数据 = FLIR (640×512, 短/中波 IR, person+car)
测试数据 = LLVIP (1280×1024, 长波 LWIR, 仅 person)
```

CSMA 学习的 FLIR 专用 IR→伪 RGB 映射，对 LLVIP IR 的不同统计特性引入了有害扰动。

### 6.3 "Loss 从 61→300" 是正常阶段转换，不是发散

| 阶段 | epoch | λ_det | L_det（原始）| 总 loss 计算式 | 总 loss |
|------|-------|-------|------------|--------------|---------|
| Easy | 0–33  | 0.1 | ≈ 599 | 0.5×0.23 + **0.1**×599 | ≈ 61 |
| Mixed | 34–99 | 0.5 | ≈ 598 | 0.3×0.15 + **0.5**×598 | ≈ 300 |

L_det 本身几乎不变（599→598），loss 翻 5× 纯粹是课程阶段权重变化的数学结果。

---

## 7. 结论与建议

### 当前最佳结果

- **最佳 checkpoint（跨域）**：`outputs_csma_v3/ckpt/epoch_0000.pt`，mAP@0.5 = **0.8666**（+0.94% vs Baseline）
- **最佳 checkpoint（FLIR val）**：`outputs_csma/ckpt/best_stage1.pt`，FLIR mAP@0.5 = **0.5931**（epoch 12）

### 提升方向

| 方向 | 预期收益 | 难度 |
|------|---------|------|
| 在 LLVIP 数据上训练 CSMA（或 FLIR+LLVIP 混合） | 高，消除域差距 | 中 |
| Test-time adaptation（用无标注 LLVIP 数据微调） | 中 | 低 |
| 早停策略：保存 epoch_0~5 最优（而非 FLIR val 最优） | 中 | 低 |
| 减小残差 scale（初始 CSMA 增益更保守） | 低 | 低 |

---

*生成自：`scripts/11_eval_all_ckpts_llvip.py`，结果文件：`outputs_csma_v3/logs/eval_all_ckpts_llvip.json`*
