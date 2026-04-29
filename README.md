# demo_RGBT_net — IR Translator + 冻结 Grounding DINO（MVP）

## 环境

```bash
pip install -r requirements.txt
```

需能下载 `IDEA-Research/grounding-dino-tiny`（首次运行自动缓存）。

## 数据

将约 10 张红外图与 `\_annotations.coco.json` 放在 `train/`（本仓库已包含示例）。

## 训练（过拟合 Demo）

```bash
python -m src.train_demo --epochs 200 --data-root train --out-dir outputs
```

- 首步会打印 tokenizer 分词与 **梯度检查**（translator 有梯度、DINO 无梯度）。
- 权重：`outputs/ckpt/translator_last.pt`
- 曲线：`outputs/logs/loss.png`
- 可视化：每 20 epoch 写入 `outputs/vis/epoch_XXXX.png`

## 仅推理可视化

```bash
python -m src.infer_vis --ckpt outputs/ckpt/translator_last.pt --data-root train --out outputs/vis/infer_grid.png
```

## 说明

- 使用 `GroundingDinoForObjectDetection`（非 `AutoModelForObjectDetection`）。
- 标签由 `processor.image_processor(..., annotations=...)` 生成，与 resize/pad 对齐。
- `class_labels` 为 **0=person、1=car**（与 prompt `person. car.` 从左到右一致）。
