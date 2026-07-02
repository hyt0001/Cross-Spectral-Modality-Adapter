# CSMA - Grounding Dino Tiny 实验结果汇总

> 最后更新：2026-07-02（FLIR Final 协议 interim 主表 + Final 训练启动）  
> **§0–§3** 为论文用精简表（仅方法 + 指标）；**§4–§6** 为工程附录（目录、权重、原始文件）。

---

## 0. 主实验总表

| 数据集 | 设定 | 方法 | mAP@0.5 | mAP | AP@0.75 | AR@100 | AP person | AP car |
|--------|------|------|---------|-----|---------|--------|-----------|--------|
| FLIR val | Final 协议；`person. car.`；T=0.2 | DINO 直推（Baseline） | 0.655 | 0.367 | 0.357 | 0.469 | 0.684 | 0.626 |
| FLIR val | Final 协议；T=0.2；CSMA=Phase0 best（待 Final 微调） | **DINO + CSMA** | **0.796** | **0.439** | **0.417** | **0.541** | **0.793** | **0.800** |
| LLVIP test | 零样本；`person.` | DINO 直推（Baseline） | 0.857 | 0.564 | 0.627 | 0.715 | 0.857 | — |
| LLVIP test | 零样本；`person.` | **DINO + CSMA** | **0.867** | **0.572** | **0.636** | **0.717** | **0.867** | — |
| M3FD val | OWL Final；`person. car.`；T=0.2 | DINO 直推（Baseline） | 0.632 | 0.400 | 0.428 | 0.468 | 0.696 | 0.569 |
| M3FD val | OWL Final；`person. car.`；T=0.2 | **DINO + CSMA（OWL）** | **0.670** | **0.414** | **0.435** | **0.490** | **0.715** | **0.626** |

> mAP = AP @IoU=0.50:0.95；AR@100 = AR @IoU=0.50:0.95, maxDets=100, area=all。LLVIP 无 car GT。**FLIR CSMA 行 interim**：Phase0 `best_stage1.pt` 在 T=0.2 下重评，Final Round-2/3 训完后再替换。Phase0 @ T=0.05 见 §1。M3FD 为 Final 微调最佳 @ T=0.2（§3.1）。

---

## 1. FLIR 数据集（Phase0 预训练）

评测集：**FLIR v1 val**（1360 张，GT=11211）；Prompt：`person. car.`

### 1.0 Baseline vs CSMA：COCO 指标对比

| 指标（IoU=0.50:0.95） | Baseline | CSMA | Δ |
|------------------------|----------|------|---|
| **AP @IoU=0.50（mAP@0.5）** | 0.490 | **0.593** | +0.103 |
| **AP @IoU=0.50:0.95（mAP）** | 0.258 | **0.332** | +0.074 |
| **AP @IoU=0.75** | 0.236 | **0.320** | +0.084 |
| AP small | 0.141 | **0.156** | +0.015 |
| AP medium | 0.349 | **0.434** | +0.085 |
| AP large | 0.591 | **0.704** | +0.113 |
| AR maxDets=10 | 0.407 | **0.416** | +0.009 |
| **AR maxDets=100** | 0.454 | **0.483** | +0.029 |
| AR small | 0.297 | **0.319** | +0.022 |
| AR medium | 0.541 | **0.577** | +0.036 |
| AR large | 0.791 | **0.792** | +0.001 |

> 主表见 §0。

### 1.1 分类别指标（person / car）
| 类别     | 指标                | Baseline | CSMA      | Δ      |
| ------ | ----------------- | -------- | --------- | ------ |
| person | AP @IoU=0.50      | 0.622    | **0.687** | +0.065 |
| person | AP @IoU=0.75      | 0.318    | **0.331** | +0.013 |
| person | AP @IoU=0.50:0.95 | 0.339    | **0.363** | +0.024 |
| person | AR maxDets=100    | 0.552    | 0.552     | 0.000  |
| car    | AP @IoU=0.50      | 0.441    | **0.499** | +0.058 |
| car    | AP @IoU=0.75      | 0.238    | **0.308** | +0.070 |
| car    | AP @IoU=0.50:0.95 | 0.244    | **0.301** | +0.057 |
| car    | AR maxDets=100    | 0.391    | **0.413** | +0.022 |


> 分类别数值来自 `eval_csma` 统一协议重跑（`eval_flir_baseline_val.json` / `eval_flir_csma_best_ep12_coco.json`）。

### 1.2 结论

1. Phase0 @ T=0.05：Baseline mAP@0.5=49.0% → CSMA 59.3%（**+10.3 pp**）；mAP、AP@0.75、AR@100 同步提升。

### 1.3 Final 协议 interim（T=0.2，与 M3FD 主表口径一致）

> **CSMA 权重仍为 Phase0 best**（`outputs_csma/ckpt/best_stage1.pt`），非 Final 续训权重；Final Round-2/3 完成后更新主表 CSMA 行。

| 指标 | Baseline | CSMA（Phase0 best） | Δ |
|------|----------|---------------------|---|
| **mAP@0.5** | 0.655 | **0.796** | +0.141 |
| mAP | 0.367 | **0.439** | +0.072 |
| AP@0.75 | 0.357 | **0.417** | +0.060 |
| AR@100 | 0.469 | **0.541** | +0.072 |
| AP person @0.5 | 0.684 | **0.793** | +0.109 |
| AP car @0.5 | 0.626 | **0.800** | +0.174 |

> 来源：`eval_flir_baseline_val_t02.json` / `eval_phase0_best_t02.json`。

---

## 2. LLVIP 零样本实验（跨数据集泛化）

评测集：**LLVIP test**（3463 张，GT=8302）；Prompt：`person.`（仅 person 类）  
设定：FLIR Phase0 权重零样本迁移，**未在 LLVIP 上训练**

### 2.0 Baseline vs CSMA 最佳：COCO 对比

> LLVIP 仅有 **person** GT；car 无 GT，不参与评测（见 §2.1）。主表见 §0。

#### 汇总指标（IoU=0.50:0.95，area=all）

| 指标 | Baseline | CSMA | Δ |
|------|----------|------|---|
| **mAP@0.5** | 0.857 | **0.867** | **+0.009** |
| **mAP（0.50:0.95）** | 0.564 | **0.572** | **+0.008** |
| **AP@0.75** | 0.627 | **0.636** | **+0.009** |
| AR maxDets=10 | 0.665 | 0.669 | +0.004 |
| **AR maxDets=100** | 0.715 | **0.717** | **+0.002** |
| AP medium | 0.235 | 0.231 | −0.004 |
| AP large | 0.581 | **0.591** | +0.010 |
| AR medium | 0.526 | 0.526 | 0.000 |
| AR large | 0.725 | **0.728** | +0.003 |


### 2.1 分类别指标
| 类别     | 指标                | Baseline | CSMA      | Δ      |
| ------ | ----------------- | -------- | --------- | ------ |
| person | AP @IoU=0.50      | 0.857    | **0.867** | +0.009 |
| person | AP @IoU=0.75      | 0.627    | **0.636** | +0.009 |
| person | AP @IoU=0.50:0.95 | 0.564    | **0.572** | +0.008 |
| person | AR maxDets=100    | 0.715    | **0.717** | +0.002 |
| car    | —                 | N/A      | N/A       | —      |


### 2.2 结论

1. LLVIP 上 CSMA 初始化权重最优（mAP@0.5=**86.7%**，+0.9 pp vs Baseline）。
2. FLIR 训练加深后 LLVIP 性能下降，存在跨域过拟合。

---

## 3. M3FD 主实验（val 20%，829 张，1024×768）

评测协议：M3FD 全量按 image_id 排序后 80/20 划分 val；`canonical_size=1024×768`；**box/text threshold=0.2**（OWL Final Model 协议）。

训练：Round-2（20 epoch，`lr=1e-5`，`λ_id=0.05`，`λ_tv=0.01`，`clamp=3.0`，`res_scale=0.1`）→ Round-3（2 epoch 短训）；早停指标 `person_car_mean`。

### 3.1 两类 Prompt 主对比（`person. car.`）

> 仅 person、car 参与 GT 与 mAP 计算；bus 等其余类忽略。推理须与训练一致：`pseudo_clamp=3.0`、`residual_scale=0.1`。

#### 3.1.0 Baseline vs CSMA OWL 最佳：COCO 指标对比

| 指标（IoU=0.50:0.95） | Baseline | CSMA OWL | Δ |
|------------------------|----------|----------|---|
| **AP @IoU=0.50（mAP@0.5）** | 0.632 | **0.670** | **+0.038** |
| **AP @IoU=0.50:0.95（mAP）** | 0.400 | **0.414** | +0.014 |
| **AP @IoU=0.75** | 0.428 | **0.435** | +0.007 |
| **AR maxDets=100** | 0.468 | **0.490** | +0.022 |

> 主表见 §0。Baseline：`eval_m3fd_baseline_val.json`（T=0.2）。CSMA OWL 最佳：Round-2 **epoch 8** → `outputs_m3fd_ckpt_b/ckpt/best_stage1.pt`；独立评测 `eval_csma_best_stage1_val_owl.json`（val 早停 person_car_mean=**0.668**）。

#### 3.1.1 分类别指标（person / car）

| 类别 | 指标 | Baseline | CSMA OWL | Δ |
|------|------|----------|----------|---|
| person | AP @IoU=0.50 | 0.696 | **0.715** | +0.019 |
| person | AP @IoU=0.75 | **0.455** | 0.452 | −0.003 |
| person | AP @IoU=0.50:0.95 | 0.427 | **0.432** | +0.005 |
| person | AR maxDets=100 | 0.498 | **0.507** | +0.009 |
| car | AP @IoU=0.50 | 0.569 | **0.626** | +0.057 |
| car | AP @IoU=0.75 | 0.402 | **0.418** | +0.016 |
| car | AP @IoU=0.50:0.95 | 0.373 | **0.396** | +0.023 |
| car | AR maxDets=100 | 0.437 | **0.474** | +0.037 |

#### 3.1.2 结论

1. OWL 管线下 CSMA mAP@0.5=**67.0%**，较 Baseline **+3.8 pp**；**person、car 双超 Baseline**（+1.9 pp / +5.7 pp @IoU=0.50）。
2. 全局最佳为 Round-2 epoch 8；Round-3 短训未进一步提升（val 最佳 epoch 0，person_car_mean=0.666）。
3. 旧版 v3（T=0.05）仅 +1.6 pp，且 car 几乎无增益；OWL 损失与 T=0.2 评测显著改善 car。
4. **不可**将本节与 §3.2 六类 prompt 数值直接对比（评测类别集不同）。

### 3.2 六类 Prompt 实验（`person. car. bus. motorcycle. truck. lamp.`）

#### 3.2.1 mAP@0.5 分类别对比

| 方法 | mAP@0.5 | person | car | bus | motorcycle | truck | lamp |
|------|---------|--------|-----|-----|------------|-------|------|
| DINO 直推（Baseline） | **0.238** | **0.684** | 0.531 | **0.202** | **0.008** | **0.001** | **0.000** |
| DINO + Pixel CSMA 微调 v1 | 0.224 | 0.681 | **0.536** | 0.108 | 0.014 | 0.002 | 0.003 |

> Baseline：`eval_m3fd_baseline_val_6cls.json`。CSMA v1：`eval_m3fd_csma_v1_6cls.json`（`epoch_0012.pt`）。与 §3.1 不可直接对比。

#### 3.2.2 结论

1. 六类 mAP@0.5 上 CSMA v1（22.4%）仍低于 Baseline（23.8%）；person 略降、car 略升，均未双超。
2. 六类协议下 (person+car)/2 与两类协议 mAP@0.5 **不可混比**（见 §3.1.2）。

---

## 4. 工程附录：实验目录索引

| 目录 | 说明 |
|------|------|
| `outputs_csma` | FLIR Phase0 主实验；mAP@0.5=59.32% |
| `outputs_csma_v3` | FLIR 长训 + LLVIP 批量 eval |
| `outputs_m3fd_finetune` | M3FD 像素 CSMA v1/v2/v3（T=0.05，历史） |
| `outputs_m3fd_ckpt_b` | M3FD OWL Round-2；**最佳 epoch 8** |
| `outputs_m3fd_final` | M3FD OWL Round-3 短训 |
| `outputs_m3fd_fa` | FeatureAdapter 主实验 |

---

## 5. 工程附录：关键结论（内部）

1. **两类 prompt OWL（§3.1）**：CSMA 最佳 mAP@0.5=**67.0%**，超 Baseline +3.8 pp；person +1.9 pp、car +5.7 pp @IoU=0.50，**双超**。
2. **六类 prompt（§3.2）**：CSMA v1 六类 mAP@0.5 仍低于 Baseline；与 §3.1 不可混比。
3. Phase0：FLIR +10.3 pp；LLVIP 零样本 CSMA +0.9 pp。

---

## 6. 工程附录：原始数据文件

| 文件 | 用途 |
|------|------|
| `outputs_csma/logs/eval_flir_baseline_val.json` | FLIR Baseline |
| `outputs_csma/logs/eval_flir_csma_best_ep12_coco.json` | FLIR CSMA 最佳 |
| `outputs_csma/logs/eval_llvip_baseline.json` | LLVIP Baseline |
| `outputs_csma/logs/eval_llvip_csma_best_coco.json` | LLVIP CSMA 最佳 |
| `outputs_csma_v3/logs/eval_all_ckpts_llvip.json` | LLVIP 批量零样本 |
| `outputs_m3fd_finetune/logs/eval_m3fd_baseline_val.json` | M3FD Baseline（person. car.，T=0.2） |
| `outputs_m3fd_ckpt_b/logs/eval_csma_best_stage1_val_owl.json` | M3FD CSMA OWL 最佳（Round-2 ep8） |
| `outputs_m3fd_ckpt_b/logs/val_early_stop.jsonl` | M3FD OWL Round-2 逐 epoch val |
| `outputs_m3fd_finetune/logs/eval_m3fd_baseline_val_2cls.json` | M3FD Baseline（T=0.05，历史） |
| `outputs_m3fd_finetune/logs/eval_m3fd_csma_v3_val_2cls.json` | M3FD Pixel CSMA v3（T=0.05，历史） |
| `outputs_m3fd_finetune/logs/eval_m3fd_baseline_val_6cls.json` | M3FD Baseline（六类） |
| `outputs_m3fd_finetune/logs/eval_m3fd_csma_v1_6cls.json` | M3FD Pixel CSMA v1 六类最佳 |
| `.report/experiments_summary_data.json` | 机器可读汇总 |


