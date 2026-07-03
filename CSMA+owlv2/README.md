# CSMA × OWLv2 代码提交

Cross-Spectral Modality Adapter (CSMA) + 冻结 OWLv2 的红外目标检测训练与外部验证代码。

## 目录结构

```
src/
  train_csma.py          # 训练入口
  eval_csma.py           # 评估入口（FLIR / LLVIP / M3FD 等）
  eval_owl_baseline.py   # OWLv2 直接吃 IR 的 baseline 评估
  config.py              # 超参配置
  csma.py                # CSMA 模型
  cmss_utils.py          # CMSS 对齐 mask 与三阶段 schedule
  dataset_flir_v1.py     # FLIR 训练/验证数据
  dataset_llvip.py       # LLVIP 评估数据
  dataset_m3fd.py        # M3FD 评估数据
  dataset_kaist.py       # （eval 模块依赖）
  dataset_not156.py      # （eval 模块依赖）
scripts/
  run_train.sh           # 从头训练（Final Model pseudo 约束配置）
  run_eval_llvip_m3fd.sh # LLVIP + M3FD 外部验证
```

## 环境依赖

- Python 3.10+
- PyTorch, Transformers (OWLv2), pycocotools, scipy, numpy, matplotlib

OWLv2 权重默认路径：`/root/autodl-tmp/OWLv2/owlv2-base-patch16-finetuned`（可在 `config.py` 的 `model_id` 修改）

## 数据路径（默认）

| 用途 | 路径 |
|------|------|
| FLIR 训练 | `/root/autodl-tmp/train` |
| FLIR 验证 | `/root/autodl-tmp/val` |
| LLVIP 测试 | `/root/autodl-tmp/LLVIP` |
| M3FD 测试 | `/root/autodl-tmp/M3FD` |

## 训练

从头训练 CSMA，使用 **pseudo 正则约束**（相对默认配置的改动）：

| 参数 | 默认值 | 本代码使用 |
|------|--------|-----------|
| id_loss_weight | 0.05 | **0.005** |
| tv_loss_weight | 0.01 | **0.05** |
| logit_reg_weight | 0.01 | **0.02** |
| pseudo_clamp | 3.0 | **2.0** |
| residual_scale | 0.1 | **0.05** |

```bash
cd OWLv2
bash scripts/run_train.sh
```

每个 epoch 结束后在 FLIR val 上验证；权重保存在 `outputs_final_config_from_scratch/ckpt/`。

## 外部验证（LLVIP / M3FD）

```bash
cd OWLv2
CKPT=outputs_final_config_from_scratch/ckpt/best.pt \
  bash scripts/run_eval_llvip_m3fd.sh
```

结果 JSON 写入 checkpoint 同实验目录下的 `logs/`。

评估协议：threshold=0.2，COCO 标准指标（AP / AP50 / AP_person / AP_car 等）。
