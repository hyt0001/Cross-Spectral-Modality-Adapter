"""
推理与可视化：加载 ``ResidualTranslator`` 权重，在伪 RGB 上跑冻结 DINO，导出对比图。
"""

from __future__ import annotations

import argparse
import os
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from src.translator import ResidualTranslator


def denormalize_pixel_values(
    pixel_values: torch.Tensor,
    image_mean: Tuple[float, float, float],
    image_std: Tuple[float, float, float],
) -> np.ndarray:
    """
    将 ImageNet 归一化的 ``pixel_values`` [C,H,W] 转为 uint8 RGB HWC，便于 ``imshow``。

    Args:
        pixel_values: 单张图，形状 [3, H, W]。

    Returns:
        ``numpy.ndarray`` uint8，形状 [H, W, 3]。
    """
    pv = pixel_values.detach()
    mean = torch.tensor(image_mean, device=pv.device, dtype=pv.dtype).view(3, 1, 1)
    std = torch.tensor(image_std, device=pv.device, dtype=pv.dtype).view(3, 1, 1)
    x = pv * std + mean
    x = x.clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    return (x * 255.0).astype(np.uint8)


def cxcywh_norm_to_xyxy_pixels(
    boxes_cxcywh: torch.Tensor,
    height: int,
    width: int,
) -> np.ndarray:
    """归一化 cxcywh → 绝对像素 xyxy，形状 [N,4]。"""
    if boxes_cxcywh.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, w, h = boxes_cxcywh.unbind(dim=-1)
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    return torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()


def draw_boxes(
    ax: Any,
    img: np.ndarray,
    boxes_xyxy: np.ndarray,
    color: str,
    linewidth: float,
) -> None:
    """在轴上绘制边界框。"""
    ax.imshow(img)
    ax.axis("off")
    for b in boxes_xyxy:
        x1, y1, x2, y2 = b
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)


def save_visualization_grid(
    translator: ResidualTranslator,
    dino: GroundingDinoForObjectDetection,
    processor: Any,
    pixel_values: torch.Tensor,
    pixel_mask: torch.Tensor,
    labels: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    out_path: str,
    box_threshold: float,
    text_threshold: float,
) -> None:
    """
    单张图三联：原输入（归一化反变换）| 伪 RGB | 伪 RGB + 预测框（红）+ GT（绿）。
    """
    dino.eval()
    translator.eval()
    mean = processor.image_processor.image_mean
    std = processor.image_processor.image_std

    pv = pixel_values.unsqueeze(0).to(device)
    pm = pixel_mask.unsqueeze(0).to(device)
    pseudo = translator(pv)

    h, w = pseudo.shape[-2], pseudo.shape[-1]
    target_sizes = torch.tensor([[h, w]], dtype=torch.int64, device=device)

    with torch.no_grad():
        outputs = dino(
            pixel_values=pseudo,
            pixel_mask=pm,
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        )
        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=input_ids.to(device),
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]

    pred_boxes = results["boxes"].cpu().numpy()
    img_ir = denormalize_pixel_values(pixel_values, mean, std)
    img_pseudo = denormalize_pixel_values(pseudo[0], mean, std)

    gt_boxes = cxcywh_norm_to_xyxy_pixels(labels["boxes"], h, w)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    draw_boxes(axes[0], img_ir, np.zeros((0, 4)), "white", 1.0)
    axes[0].set_title("Input (IR→RGB normalized)")

    draw_boxes(axes[1], img_pseudo, np.zeros((0, 4)), "white", 1.0)
    axes[1].set_title("Pseudo-RGB (translator)")

    axes[2].imshow(img_pseudo)
    axes[2].axis("off")
    axes[2].set_title("Pred(red) vs GT(green)")
    for b in pred_boxes:
        x1, y1, x2, y2 = b
        axes[2].add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
        )
    for b in gt_boxes:
        x1, y1, x2, y2 = b
        axes[2].add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_multi_sample_grid(
    translator: ResidualTranslator,
    dino: GroundingDinoForObjectDetection,
    processor: Any,
    samples: List[dict[str, Any]],
    text_prompt: str,
    device: torch.device,
    out_path: str,
    box_threshold: float,
    text_threshold: float,
) -> None:
    """将前 ``len(samples)`` 张图各画一行三联，保存为单张 PNG。"""
    enc = processor.tokenizer(text_prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    mean = processor.image_processor.image_mean
    std = processor.image_processor.image_std
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    dino.eval()
    translator.eval()

    for row, sample in enumerate(samples):
        pv = sample["pixel_values"].to(device).unsqueeze(0)
        pm = sample["pixel_mask"].to(device).unsqueeze(0)
        labels = sample["labels"]
        pseudo = translator(pv)
        h, w = pseudo.shape[-2], pseudo.shape[-1]
        target_sizes = torch.tensor([[h, w]], dtype=torch.int64, device=device)

        with torch.no_grad():
            outputs = dino(
                pixel_values=pseudo,
                pixel_mask=pm,
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            results = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=input_ids.to(device),
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]

        pred_boxes = results["boxes"].cpu().numpy()
        img_ir = denormalize_pixel_values(sample["pixel_values"], mean, std)
        img_pseudo = denormalize_pixel_values(pseudo[0].cpu(), mean, std)
        gt_boxes = cxcywh_norm_to_xyxy_pixels(labels["boxes"], h, w)

        axes[row, 0].imshow(img_ir)
        axes[row, 0].axis("off")
        axes[row, 0].set_title("Input")
        axes[row, 1].imshow(img_pseudo)
        axes[row, 1].axis("off")
        axes[row, 1].set_title("Pseudo-RGB")
        axes[row, 2].imshow(img_pseudo)
        axes[row, 2].axis("off")
        axes[row, 2].set_title("Pred(red) / GT(green)")
        for b in pred_boxes:
            x1, y1, x2, y2 = b
            axes[row, 2].add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
            )
        for b in gt_boxes:
            x1, y1, x2, y2 = b
            axes[row, 2].add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="lime",
                    facecolor="none",
                )
            )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="加载 translator 权重并可视化检测与伪 RGB")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/ckpt/translator_last.pt",
        help="translator 的 state_dict 路径",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="train",
        help="含 COCO json 与图像的目录",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/vis/infer_grid.png",
        help="输出 PNG 路径",
    )
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "IDEA-Research/grounding-dino-tiny"
    text_prompt = "person. car."

    processor = AutoProcessor.from_pretrained(model_id)
    dino = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False

    from src.dataset import FlirCocoOverfitDataset, build_coco_category_to_class_index, collate_fn

    cat_map = build_coco_category_to_class_index(text_prompt)
    ds = FlirCocoOverfitDataset(args.data_root, processor, text_prompt, cat_map)
    translator = ResidualTranslator().to(device)
    state = torch.load(args.ckpt, map_location=device)
    translator.load_state_dict(state)
    translator.eval()

    samples = [ds[i] for i in range(min(3, len(ds)))]
    save_multi_sample_grid(
        translator,
        dino,
        processor,
        samples,
        text_prompt,
        device,
        args.out,
        args.box_threshold,
        args.text_threshold,
    )
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
