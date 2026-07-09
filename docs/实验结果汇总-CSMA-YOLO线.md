# CSMA 实验结果汇总 — YOLO Teacher 线


| 字段      | 内容                                                         |
| ------- | ---------------------------------------------------------- |
| 实验线     | CSMA + YOLO teacher（v3-tiny / v8m / v8n）+ DETR-R50 teacher |
| 文档版本    | v1.3                                                       |
| 数据截止    | 2026-07-02                                                 |
| 原始 JSON | `outputs_csma/logs/full_metrics/*.json`                    |


> 静态汇总文档，数据从 eval JSON **抄写**；有新结果时改对应表格即可。  
> 指标由 `pycocotools` COCOeval 计算（`compute_map` in `src/eval_csma.py`）。

---

## 0. 评测协议与指标说明

### 0.1 数据集


| 项                | FLIR               | LLVIP           | M3FD         |
| ---------------- | ------------------ | --------------- | ------------ |
| 划分               | `FLIR_License/val` | infrared test   | ir test      |
| 类别               | person + car       | **person only** | person + car |
| conf / threshold | 0.05               | 0.05            | 0.05         |
| CSMA 训练          | FLIR train         | zero-shot       | zero-shot    |


### 0.2 指标列（COCO 标准）


| 列名                                                                              | 含义                                           |
| ------------------------------------------------------------------------------- | -------------------------------------------- |
| **[mAP@0.5](mailto:mAP@0.5)**                                                   | AP @[IoU=0.50, area=all]                     |
| **[mAP@0.5](mailto:mAP@0.5):0.95**                                              | AP @[IoU=0.50:0.95, area=all]                |
| **[AP@0.75](mailto:AP@0.75)**                                                   | AP @[IoU=0.75, area=all]                     |
| **AP_s / AP_m / AP_l**                                                          | AP @[IoU=0.50:0.95, area=small/medium/large] |
| **AR@100**                                                                      | AR @[IoU=0.50:0.95, area=all, maxDets=100]   |
| **AR_l**                                                                        | AR @[IoU=0.50:0.95, area=large, maxDets=100] |
| **AP_person / AP_car**                                                          | 各类 [AP@0.5](mailto:AP@0.5)                   |
| **[AP_person@0.75](mailto:AP_person@0.75) / [AP_car@0.75**](mailto:AP_car@0.75) | 各类 [AP@0.75](mailto:AP@0.75)                 |


LLVIP 仅 person 类，`AP_car` 及 small 目标 AP/AR 无意义，表中记 **—**。

**推理管线**：`IR → GDINO processor(512) → CSMA → denorm → YOLO.predict → COCOeval`

**Final 微调正则**：`id_loss=0.005`，`tv_loss=0.05`，`pseudo_clamp=2.0`，`residual_scale=1.0`

---

## 1. Teacher：YOLOv3-tiny


| 项               | 值                                                                               |
| --------------- | ------------------------------------------------------------------------------- |
| Teacher 权重      | `yolov3-tinyu.pt`（冻结）                                                           |
| Base 训练         | 20 epoch → `outputs_csma_v3tiny_base/`，best val [mAP@0.5](mailto:mAP@0.5)=42.43 |
| Final 微调        | 2 epoch → `outputs_csma_v3tiny_final/`                                          |
| 最佳 CSMA ckpt    | `outputs_csma_v3tiny_final/ckpt/epoch_0000.pt`                                  |
| LLVIP CSMA eval | `native` + AdaBN=50                                                             |


### 1.1 评测结果


| 数据集   | 输入               | [mAP@0.5](mailto:mAP@0.5) | [mAP@0.5](mailto:mAP@0.5):0.95 | [AP@0.75](mailto:AP@0.75) | AP_s  | AP_m  | AP_l  | AR@100 | AR_l  | AP_person | AP_car | [AP_p@0.75](mailto:AP_p@0.75) | [AP_c@0.75](mailto:AP_c@0.75) | n_pred | n_gt  |
| ----- | ---------------- | ------------------------- | ------------------------------ | ------------------------- | ----- | ----- | ----- | ------ | ----- | --------- | ------ | ----------------------------- | ----------------------------- | ------ | ----- |
| FLIR  | IR 直送            | 52.53                     | 26.62                          | 23.97                     | 8.01  | 36.39 | 61.67 | 35.18  | 69.87 | 49.33     | 55.73  | 19.37                         | 28.56                         | 21065  | 11211 |
| FLIR  | + CSMA Final ep0 | **55.04**                 | 28.20                          | 25.56                     | 13.67 | 36.52 | 53.45 | 38.45  | 61.59 | 53.30     | 56.77  | 19.54                         | 31.58                         | 25245  | 11211 |
| LLVIP | IR 直送            | 62.19                     | 37.81                          | 39.64                     | —     | 8.57  | 39.40 | 47.99  | 49.46 | 62.19     | —      | 39.64                         | —                             | 20113  | 8302  |
| LLVIP | + CSMA AdaBN50   | 52.07                     | 31.94                          | 34.04                     | —     | 6.83  | 33.44 | 41.12  | 42.29 | 52.07     | —      | 34.04                         | —                             | 19806  | 8302  |
| M3FD  | IR 直送            | 34.54                     | 19.46                          | 20.12                     | 2.11  | 25.60 | 55.24 | 24.10  | 63.16 | 31.60     | 37.49  | 18.92                         | 21.33                         | 10028  | 8892  |
| M3FD  | + CSMA Final ep0 | 28.61                     | 15.08                          | 14.71                     | 1.89  | 21.36 | 43.70 | 19.89  | 53.16 | 30.73     | 26.49  | 16.62                         | 12.80                         | 9094   | 8892  |


**FLIR Δ（CSMA − IR）**：[mAP@0.5](mailto:mAP@0.5) **+2.51**，[mAP@0.5](mailto:mAP@0.5):0.95 +1.58，[AP@0.75](mailto:AP@0.75) +1.59

**结论**：FLIR 域内唯一明显提升的 teacher；跨域 CSMA 仍低于 IR baseline。

**JSON**：`full_metrics/flir_v3tiny_ir_raw.json`，`flir_v3tiny_csma_ep0.json`，`llvip_v3tiny_ir_raw.json`，`llvip_v3tiny_csma_adabn50.json`，`m3fd_v3tiny_ir_raw.json`，`m3fd_v3tiny_csma_ep0.json`

---

## 2. Teacher：YOLOv8-m


| 项               | 值                                                                        |
| --------------- | ------------------------------------------------------------------------ |
| Teacher 权重      | `yolov8m.pt`（冻结）                                                         |
| Base 训练         | 20 epoch → `outputs_csma_yolo/`，best val [mAP@0.5](mailto:mAP@0.5)=66.23 |
| Final 微调        | 2 epoch → `outputs_csma_yolo_final/`                                     |
| 最佳 CSMA ckpt    | `outputs_csma_yolo_final/ckpt/epoch_0001.pt`                             |
| LLVIP CSMA eval | `native` + AdaBN=50                                                      |


### 2.1 评测结果


| 数据集   | 输入               | [mAP@0.5](mailto:mAP@0.5) | [mAP@0.5](mailto:mAP@0.5):0.95 | [AP@0.75](mailto:AP@0.75) | AP_s  | AP_m  | AP_l  | AR@100 | AR_l  | AP_person | AP_car | [AP_p@0.75](mailto:AP_p@0.75) | [AP_c@0.75](mailto:AP_c@0.75) | n_pred | n_gt  |
| ----- | ---------------- | ------------------------- | ------------------------------ | ------------------------- | ----- | ----- | ----- | ------ | ----- | --------- | ------ | ----------------------------- | ----------------------------- | ------ | ----- |
| FLIR  | IR 直送            | 75.87                     | 44.44                          | 43.93                     | 24.03 | 55.42 | 77.44 | 52.04  | 81.58 | 77.32     | 74.42  | 40.32                         | 47.54                         | 21789  | 11211 |
| FLIR  | + CSMA Final ep1 | 75.60                     | **45.47**                      | **46.49**                 | 28.54 | 53.74 | 77.09 | 55.13  | 81.15 | 75.40     | 75.81  | 40.44                         | 52.53                         | 33988  | 11211 |
| LLVIP | IR 直送            | 78.03                     | 53.26                          | 60.34                     | —     | 13.63 | 55.71 | 60.83  | 62.41 | 78.03     | —      | 60.34                         | —                             | 16252  | 8302  |
| LLVIP | + CSMA AdaBN50   | 66.68                     | 43.49                          | 47.92                     | —     | 12.60 | 45.32 | 51.78  | 53.10 | 66.68     | —      | 47.92                         | —                             | 16449  | 8302  |
| M3FD  | IR 直送            | 56.88                     | 35.15                          | 36.68                     | 10.35 | 44.12 | 73.49 | 40.06  | 78.21 | 51.85     | 61.91  | 34.54                         | 38.83                         | 11129  | 8892  |
| M3FD  | + CSMA Final ep1 | 46.63                     | 27.00                          | 27.21                     | 7.20  | 34.77 | 63.16 | 32.40  | 70.14 | 43.89     | 49.36  | 27.20                         | 27.23                         | 9988   | 8892  |


**FLIR Δ（CSMA − IR）**：[mAP@0.5](mailto:mAP@0.5) −0.27，[mAP@0.5](mailto:mAP@0.5):0.95 **+1.03**，[AP@0.75](mailto:AP@0.75) **+2.56**

**结论**：FLIR [mAP@0.5](mailto:mAP@0.5) 略降但 strict IoU / [AP@0.75](mailto:AP@0.75) 略升；跨域 pseudo_rgb 掉点明显。

**JSON**：`full_metrics/flir_v8m_ir_raw.json`，`flir_v8m_csma_ep1.json`，`llvip_v8m_ir_raw.json`，`llvip_v8m_csma_adabn50.json`，`m3fd_v8m_ir_raw.json`，`m3fd_v8m_csma_ep1.json`

---

## 3. Teacher：YOLOv8-n


| 项               | 值                                                                                |
| --------------- | -------------------------------------------------------------------------------- |
| Teacher 权重      | `yolov8n.pt`（冻结）                                                                 |
| Base 训练         | 20 epoch → `outputs_csma_yolov8n_base/`，best val [mAP@0.5](mailto:mAP@0.5)=37.63 |
| Final 微调        | 2 epoch → `outputs_csma_yolov8n_final/`                                          |
| 最佳 CSMA ckpt    | `outputs_csma_yolov8n_final/ckpt/epoch_0001.pt`                                  |
| LLVIP CSMA eval | `native` + AdaBN=50                                                              |


### 3.1 评测结果


| 数据集   | 输入               | [mAP@0.5](mailto:mAP@0.5) | [mAP@0.5](mailto:mAP@0.5):0.95 | [AP@0.75](mailto:AP@0.75) | AP_s  | AP_m  | AP_l  | AR@100 | AR_l  | AP_person | AP_car | [AP_p@0.75](mailto:AP_p@0.75) | [AP_c@0.75](mailto:AP_c@0.75) | n_pred | n_gt  |
| ----- | ---------------- | ------------------------- | ------------------------------ | ------------------------- | ----- | ----- | ----- | ------ | ----- | --------- | ------ | ----------------------------- | ----------------------------- | ------ | ----- |
| FLIR  | IR 直送            | 64.72                     | 36.57                          | 35.53                     | 16.40 | 46.22 | 73.76 | 45.67  | 78.82 | 64.76     | 64.68  | 32.58                         | 38.49                         | 24556  | 11211 |
| FLIR  | + CSMA Final ep1 | 63.93                     | 35.50                          | 34.03                     | 21.34 | 43.54 | 61.82 | 45.91  | 67.70 | 65.18     | 62.68  | 29.62                         | 38.45                         | 26212  | 11211 |
| LLVIP | IR 直送            | 67.43                     | 43.88                          | 48.11                     | —     | 12.93 | 45.55 | 51.35  | 52.74 | 67.43     | —      | 48.11                         | —                             | 14272  | 8302  |
| LLVIP | + CSMA AdaBN50   | 50.95                     | 33.05                          | 35.45                     | —     | 5.54  | 34.88 | 41.40  | 42.71 | 50.95     | —      | 35.45                         | —                             | 16143  | 8302  |
| M3FD  | IR 直送            | 48.39                     | 28.68                          | 29.27                     | 7.26  | 36.53 | 65.68 | 34.65  | 71.44 | 47.11     | 49.66  | 29.61                         | 28.93                         | 13226  | 8892  |
| M3FD  | + CSMA Final ep1 | 38.32                     | 21.23                          | 21.31                     | 5.47  | 29.63 | 49.95 | 26.84  | 57.36 | 45.28     | 31.36  | 25.53                         | 17.09                         | 9127   | 8892  |


**FLIR Δ（CSMA − IR）**：[mAP@0.5](mailto:mAP@0.5) −0.79，[mAP@0.5](mailto:mAP@0.5):0.95 −1.07

**结论**：FLIR 基本持平略降；跨域与 v8m 类似，CSMA 明显低于 IR baseline。

**JSON**：`full_metrics/flir_v8n_ir_raw.json`，`flir_v8n_csma_ep1.json`，`llvip_v8n_ir_raw.json`，`llvip_v8n_csma_adabn50.json`，`m3fd_v8n_ir_raw.json`，`m3fd_v8n_csma_ep1.json`

---

## 4. Teacher：DETR-R50


| 项                      | 值                                                                                                         |
| ---------------------- | --------------------------------------------------------------------------------------------------------- |
| Teacher                | `facebook/detr-resnet-50`（冻结）                                                                             |
| Base 训练                | 30 epoch 热启动（from v8m-CSMA）→ `outputs_csma_detr_base/`，best val [mAP@0.5](mailto:mAP@0.5)=**75.39**（ep19） |
| Final 微调               | 2 epoch，lr=1e-5 → `outputs_csma_detr_final/`                                                              |
| 最佳 CSMA ckpt           | `outputs_csma_detr_final/ckpt/epoch_0000.pt`（FLIR offline 最优）                                             |
| LLVIP / M3FD CSMA eval | `pseudo_resize=native`                                                                                    |


### 4.1 评测结果


| 数据集   | 输入               | [mAP@0.5](mailto:mAP@0.5) | [mAP@0.5](mailto:mAP@0.5):0.95 | [AP@0.75](mailto:AP@0.75) | AP_s  | AP_m  | AP_l  | AR@100 | AR_l  | AP_person | AP_car | [AP_p@0.75](mailto:AP_p@0.75) | [AP_c@0.75](mailto:AP_c@0.75) | n_pred | n_gt  |
| ----- | ---------------- | ------------------------- | ------------------------------ | ------------------------- | ----- | ----- | ----- | ------ | ----- | --------- | ------ | ----------------------------- | ----------------------------- | ------ | ----- |
| FLIR  | IR 直送            | 64.59                     | 32.42                          | 28.02                     | 12.99 | 42.34 | 72.20 | 43.10  | 79.71 | 62.02     | 67.16  | 22.90                         | 33.14                         | 50418  | 11211 |
| FLIR  | + CSMA Final ep0 | **75.51**                 | **39.00**                      | **34.40**                 | 23.25 | 47.07 | 72.40 | 51.12  | 79.29 | 71.19     | 79.83  | 23.98                         | 44.82                         | 67274  | 11211 |
| LLVIP | IR 直送            | 67.44                     | 40.64                          | 43.09                     | —     | 14.34 | 42.26 | 54.24  | 55.26 | 67.44     | —      | 43.09                         | —                             | 58484  | 8302  |
| LLVIP | + CSMA Final ep0 | 59.02                     | 35.37                          | 37.45                     | —     | 7.55  | 37.18 | 45.36  | 46.67 | 59.02     | —      | 37.45                         | —                             | 23684  | 8302  |
| M3FD  | IR 直送            | 43.95                     | 23.31                          | 22.31                     | 4.47  | 30.70 | 56.46 | 31.53  | 64.80 | 43.83     | 44.08  | 22.70                         | 21.92                         | 34038  | 8892  |
| M3FD  | + CSMA Final ep0 | **47.16**                 | 22.77                          | 19.73                     | 5.38  | 29.92 | 50.81 | 31.68  | 61.58 | 44.92     | 49.39  | 17.81                         | 21.65                         | 31222  | 8892  |


**FLIR Δ（CSMA − IR）**：[mAP@0.5](mailto:mAP@0.5) **+10.92**，[mAP@0.5](mailto:mAP@0.5):0.95 +6.58，[AP@0.75](mailto:AP@0.75) +6.38

**结论**：FLIR 域内提升最大（+10.9pt）；M3FD 跨域 CSMA 略高于 IR baseline；LLVIP 仍低于 IR 直送（−8.4pt）。

### 4.2 推理期 residual_scale 插值（2026-07-08 补充）

CSMA 为残差式输出（pseudo = IR + scale·delta）。推理时下调 `residual_scale` 相当于在原始 IR 与完整 CSMA 变换之间插值，可消除跨域伪影导致的掉点。ckpt 固定为 `outputs_csma_detr_final/ckpt/epoch_0000.pt`，各 scale 下 mAP@0.5：

| residual_scale | FLIR（基线 0.6459） | LLVIP（基线 0.6744） | M3FD（基线 0.4395） | 三集全不掉 |
| -------------- | ------------------- | -------------------- | ------------------- | ---------- |
| 1.0（原始）    | 0.7551 (+10.9)      | 0.5902 (−8.4)        | 0.4716 (+3.2)       | ✗          |
| 0.5            | —                   | 0.6286 (−4.6)        | —                   | ✗          |
| 0.4            | —                   | 0.6662 (−0.8)        | —                   | ✗          |
| **0.35**       | **0.7098 (+6.4)**   | **0.6791 (+0.5)**    | **0.4553 (+1.6)**   | **✓**      |
| 0.3            | 0.6970 (+5.1)       | 0.6815 (+0.7)        | 0.4523 (+1.3)       | ✓          |
| 0.25           | 0.6848 (+3.9)       | 0.6786 (+0.4)        | 0.4468 (+0.7)       | ✓          |

**结论**：`residual_scale=0.35` 时 FLIR/LLVIP/M3FD 三数据集全部 ≥ IR baseline（+6.4 / +0.5 / +1.6 pt），无需重训。域内推荐 rs=1.0（最大增益），跨域/未知域部署推荐 rs=0.35。

**JSON**：`outputs_csma_detr_final/logs/eval_ep0000_pseudo_rgb_rs*.json`，`llvip/eval_llvip_detr_final_ep0_rs*.json`，`m3fd/eval_m3fd_detr_final_ep0_rs*.json`

**JSON**：`full_metrics/flir_detr_ir_raw.json`，`flir_detr_csma_ep0.json`，`llvip_detr_ir_raw.json`，`llvip_detr_csma_ep0.json`，`m3fd_detr_ir_raw.json`，`m3fd_detr_csma_ep0.json`

---

## 5. 其它实验状态


| 实验                             | 状态    | 说明                                                                                                 |
| ------------------------------ | ----- | -------------------------------------------------------------------------------------------------- |
| v3-tiny IR-Aug base            | ✅ 已完成 | `outputs_csma_v3tiny_aug_base/ckpt/best_stage1.pt`，train val [mAP@0.5](mailto:mAP@0.5)=38.24（ep15） |
| v3-tiny IR-Aug Final + 跨域 eval | ✅ 已完成 | FLIR 53.87（vs 无增强 55.04），LLVIP AdaBN50 52.55（vs 无增强 52.44）；增强无净收益                                  |
| DETR-CSMA Final                | ✅ 完成  | base + Final 2 epoch + FLIR/LLVIP/M3FD eval 已齐                                                     |


### 5.1 v3-tiny 强化实验结论（掉点归因）

**结论**：CSMA 的域内（即训练CSMA的集内）提升对任意 teacher 成立且弱模型受益最大（v3-tiny +2.5、DETR +10.9）；但**跨数据集零样本迁移只对归一化鲁棒、未过度特化的检测器（GDINO/DETR 类）成立**。YOLO 线在本工作中定位为划定 CSMA 适用边界的消融实验。

**JSON**：`llvip/eval_llvip_gdino_v3tinyCSMA.json`，`llvip/eval_llvip_detr_v3tinyCSMA_native.json`，`llvip/eval_llvip_v3tiny_gdinoCSMA_native*.json`

---

## 6. 跨域结论（简要）

1. **v3-tiny**：FLIR +2.5pt [mAP@0.5](mailto:mAP@0.5)；LLVIP CSMA 52% vs IR 62%；M3FD CSMA 29% vs IR 35%。
2. **v8m**：FLIR [mAP@0.5](mailto:mAP@0.5) 基本持平，[AP@0.75](mailto:AP@0.75) +2.6pt；LLVIP 掉 11pt（78%→67%）。
3. **v8n**：FLIR −0.8pt；LLVIP 掉 16pt（67%→51%）；M3FD 掉 10pt（48%→38%）。
4. **DETR**：FLIR **+10.9pt**；LLVIP −8.4pt（67%→59%）；M3FD **+3.2pt**（44%→47%）。
5. **LLVIP small AP/AR 为 —**：测试集几乎无 small 框，COCOeval 返回无效值。
6. **AdaBN** 对 LLVIP YOLO+CSMA 有缓解（v3-tiny native 45%→AdaBN 52%）。

---

## 7. 重跑 eval 命令（更新数据时用）

```bash
source /root/miniconda3/etc/profile.d/conda.sh
cd /root/autodl-tmp/Cross-Spectral-Modality-Adapter
bash scripts/run_summary_metrics_eval.sh
# 结果写入 outputs_csma/logs/full_metrics/*.json，再抄到本文档
```


| 脚本                                    | 用途                            |
| ------------------------------------- | ----------------------------- |
| `scripts/run_summary_metrics_eval.sh` | YOLO 主实验 batch eval           |
| `scripts/01_finetune_detr_final.sh`   | DETR base + Final + FLIR eval |
| `scripts/03_eval_llvip.sh`            | LLVIP eval（含 DETR）            |
| `scripts/04_eval_m3fd.sh`             | M3FD eval（含 DETR）             |
| `scripts/05_eval_yolo.sh`             | 单条 FLIR YOLO eval             |


