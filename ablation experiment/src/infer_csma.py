"""
CSMA 推理与可视化。

对应 docs/TD.md §1.6，docs/architecture.md §8。

推理阶段仅需红外图像，无需 RGB、无需 GMM、无需 L_align。
所有通用工具函数（denormalize_pixel_values、cxcywh_norm_to_xyxy_pixels、draw_boxes）
直接从 src/infer_vis.py 导入，不重复实现。
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from src.cmss_utils import CMSSScheduler, build_cmss_mask, compute_cmss
from src.config import CSMAConfig
from src.csma import CSMA
from src.infer_vis import (
    cxcywh_norm_to_xyxy_pixels,
    denormalize_pixel_values,
    draw_boxes,
)
from src.train_csma import extract_dino_backbone_features


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: 单图推理
# ──────────────────────────────────────────────────────────────────────────────

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

    推理阶段无需 RGB 配对、无需 GMM、无需 L_align。
    完整流程：读图 → processor → csma(pv) → dino → post_process。

    Args:
        csma:             已加载权重的 CSMA 适配器（eval 模式）。
        dino:             冻结的 GroundingDinoForObjectDetection（eval 模式）。
        processor:        HuggingFace AutoProcessor。
        ir_image_path:    红外图像路径。
        text_prompt:      检测文本提示，如 "person. car."。
        device:           推理设备。
        box_threshold:    检测框置信度阈值。
        text_threshold:   文本匹配阈值。

    Returns:
        字典，包含：
            boxes        [N, 4] numpy，原始像素 xyxy 坐标
            scores       [N]    numpy，置信度
            labels       List[str]，检测到的类别名称
            pseudo_rgb   [1, 3, H, W] Tensor，CSMA 输出的伪 RGB
            pseudo_rgb_np [H, W, 3] uint8 numpy，反归一化后的伪 RGB
    """
    csma.eval()
    dino.eval()

    # Phase 1.1：读图与预处理
    image = Image.open(ir_image_path).convert("RGB")
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    ir_pv = inputs["pixel_values"].to(device)                   # [1, 3, H, W]
    input_ids = inputs["input_ids"].to(device)                  # [1, T]
    attention_mask = inputs["attention_mask"].to(device)        # [1, T]

    # Phase 1.2：CSMA 前向（红外 → 伪 RGB）
    with torch.no_grad():
        pseudo_rgb = csma(ir_pv)                                # [1, 3, H, W]

    h, w = pseudo_rgb.shape[-2], pseudo_rgb.shape[-1]
    target_sizes = torch.tensor([[h, w]], dtype=torch.int64, device=device)

    # Phase 1.3：冻结 DINO 前向
    with torch.no_grad():
        outputs = dino(
            pixel_values=pseudo_rgb,
            pixel_mask=torch.ones(1, h, w, dtype=torch.long, device=device),
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # Phase 1.4：后处理
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )[0]

    mean = processor.image_processor.image_mean
    std = processor.image_processor.image_std
    pseudo_rgb_np = denormalize_pixel_values(pseudo_rgb[0].cpu(), mean, std)

    return {
        "boxes":        results["boxes"].cpu().numpy(),
        "scores":       results["scores"].cpu().numpy(),
        "labels":       results["labels"],
        "pseudo_rgb":   pseudo_rgb.cpu(),
        "pseudo_rgb_np": pseudo_rgb_np,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: CMSS 热力图可视化（论文 Figure 专用）
# ──────────────────────────────────────────────────────────────────────────────

def visualize_cmss_mask(
    ir_pv: torch.Tensor,
    csma: CSMA,
    dino: GroundingDinoForObjectDetection,
    cmss_sched: CMSSScheduler,
    stage: int,
    device: torch.device,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    processor: Any,
    rgb_pv: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    将当前阶段 GMM-CMSS 掩码叠加在红外图像上，生成可用于论文 Figure 的热力图。

    Args:
        ir_pv:            [1, 3, H, W] 红外图像（ImageNet 归一化），在 device 上。
        csma:             CSMA 适配器（eval 模式）。
        dino:             冻结 DINO（eval 模式）。
        cmss_sched:       CMSSScheduler，提供 sorted_means 与 gmm。
        stage:            课程阶段编号：0=A(Easy), 1=B(Mixed), 2=C(Hard)。
        device:           计算设备。
        input_ids:        文本 input_ids [1, T]。
        attention_mask:   文本 attention_mask [1, T]。
        processor:        HuggingFace AutoProcessor（用于反归一化显示）。
        rgb_pv:           [1, 3, H, W] RGB 配对图像（可选）。
                          若为 None，以 pseudo_rgb 自比较（CMSS≈1，热力图全亮）。
        alpha:            热力图叠加透明度（0~1）。

    Returns:
        叠加图：HWC uint8 numpy array，尺寸与 ir_pv 输入分辨率相同。
    """
    csma.eval()
    dino.eval()

    H, W = ir_pv.shape[-2], ir_pv.shape[-1]

    with torch.no_grad():
        pseudo_rgb = csma(ir_pv)

        # Phase 2.1：提取 DINO backbone 特征（encoder 入口）
        feat_ir = extract_dino_backbone_features(
            dino, pseudo_rgb, input_ids, attention_mask
        )                                                        # [1, L_total, 256]

        if rgb_pv is not None:
            feat_rgb = extract_dino_backbone_features(
                dino, rgb_pv.to(device), input_ids, attention_mask
            )
        else:
            # 自比较降级：CMSS≈1（全背景，热力图全亮）
            feat_rgb = feat_ir.clone()

        # Phase 2.2：计算 CMSS 值 [1, L_total]
        cmss_map = compute_cmss(feat_rgb, feat_ir)

        # Phase 2.3：取第一尺度 n1 = (H//8)*(W//8) tokens → 空间掩码
        H8, W8 = H // 8, W // 8
        n1 = H8 * W8
        n1 = min(n1, cmss_map.shape[1])     # 防止越界（空间尺寸近似）

        cmss_map_primary = cmss_map[:, :n1]  # [1, n1]
        mu1, mu2, mu3 = cmss_sched.sorted_means
        mask = build_cmss_mask(
            cmss_map_primary,
            stage,
            mu1, mu2, mu3,
            csma.cfg.mask_ratio,
            cmss_sched.gmm,
        )                                                        # [1, n1]

        # Phase 2.4：reshape 到 (1, 1, H8, W8) 再上采样至 (1, 1, H, W)
        mask_spatial = mask.reshape(1, 1, H8, W8).float()
        mask_full = F.interpolate(
            mask_spatial, size=(H, W), mode="nearest"
        )[0, 0].cpu().numpy()                                   # [H, W]，0 或 1

    # Phase 2.5：可视化：热力图 alpha 叠加在反归一化 IR 图上
    mean = processor.image_processor.image_mean
    std = processor.image_processor.image_std
    ir_np = denormalize_pixel_values(ir_pv[0].cpu(), mean, std)  # [H, W, 3] uint8

    colormap = cm.get_cmap("RdYlGn")                             # 0=红(掩蔽) 1=绿(保留)
    heatmap_rgba = colormap(mask_full)                           # [H, W, 4]
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    overlay = (
        ir_np.astype(np.float32) * (1 - alpha)
        + heatmap_rgb.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)

    return overlay


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: 多样本对比图（CSMA 类型安全版）
# ──────────────────────────────────────────────────────────────────────────────

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
    将多个样本各画一行三联对比图（原始 IR | 伪 RGB | 伪 RGB + 预测框 + GT 框），
    保存为单张 PNG。

    与 infer_vis.save_multi_sample_grid 逻辑完全相同，仅将
    translator: ResidualTranslator 替换为 csma: CSMA（类型安全版）。

    Args:
        csma:           CSMA 适配器（eval 模式）。
        dino:           冻结 DINO（eval 模式）。
        processor:      HuggingFace AutoProcessor。
        samples:        样本列表，每个元素为 FlirPairedDataset.__getitem__ 的输出。
        text_prompt:    检测文本提示。
        device:         推理设备。
        out_path:       输出 PNG 路径（自动创建父目录）。
        box_threshold:  检测框置信度阈值。
        text_threshold: 文本匹配阈值。
    """
    enc = processor.tokenizer(text_prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    mean = processor.image_processor.image_mean
    std = processor.image_processor.image_std

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    csma.eval()
    dino.eval()

    for row, sample in enumerate(samples):
        pv = sample["pixel_values"].to(device).unsqueeze(0)     # [1, 3, H, W]
        pm = sample["pixel_mask"].to(device).unsqueeze(0)       # [1, H, W]
        labels = sample["labels"]

        with torch.no_grad():
            pseudo = csma(pv)

        h, w = pseudo.shape[-2], pseudo.shape[-1]
        target_sizes = torch.tensor([[h, w]], dtype=torch.int64, device=device)

        with torch.no_grad():
            outputs = dino(
                pixel_values=pseudo,
                pixel_mask=pm,
                input_ids=input_ids.expand(1, -1),
                attention_mask=attention_mask.expand(1, -1),
            )
            results = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]

        pred_boxes = results["boxes"].cpu().numpy()
        img_ir = denormalize_pixel_values(sample["pixel_values"], mean, std)
        img_pseudo = denormalize_pixel_values(pseudo[0].cpu(), mean, std)
        gt_boxes = cxcywh_norm_to_xyxy_pixels(labels["boxes"], h, w)

        # 第一列：原始 IR
        axes[row, 0].imshow(img_ir)
        axes[row, 0].axis("off")
        axes[row, 0].set_title("Input (IR)")

        # 第二列：伪 RGB
        axes[row, 1].imshow(img_pseudo)
        axes[row, 1].axis("off")
        axes[row, 1].set_title("Pseudo-RGB (CSMA)")

        # 第三列：预测框（红）+ GT（绿）
        axes[row, 2].imshow(img_pseudo)
        axes[row, 2].axis("off")
        axes[row, 2].set_title("Pred(red) / GT(green)")
        for b in pred_boxes:
            x1, y1, x2, y2 = b
            axes[row, 2].add_patch(
                patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor="red", facecolor="none",
                )
            )
        for b in gt_boxes:
            x1, y1, x2, y2 = b
            axes[row, 2].add_patch(
                patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor="lime", facecolor="none",
                )
            )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: CLI 推理入口
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSMA 推理与可视化：加载训练好的 CSMA 权重，在 FLIR 红外图像上运行检测"
    )
    parser.add_argument("--ckpt",             type=str, required=True,
                        help="CSMA state_dict 路径（必填）")
    parser.add_argument("--dataset",          type=str, default="flir_v1",
                        choices=["legacy", "flir_v1", "flir_v2"],
                        help="数据集类型：flir_v1=FLIR_License（默认）；"
                             "flir_v2=FLIR_ADAS_v2；legacy=旧版 train/ 目录")
    parser.add_argument("--data-root",        type=str,
                        default="FLIR_License/val",
                        help="flir_v1: split 目录（含 thermal_annotations.json + thermal_8_bit/ + RGB/）；"
                             "flir_v2: thermal split 目录（含 coco.json + data/）；"
                             "legacy: 含 _annotations.coco.json 的 IR 目录")
    parser.add_argument("--out",              type=str,
                        default="outputs_csma/vis/infer_grid.png",
                        help="多样本对比 PNG 输出路径")
    parser.add_argument("--out-mask",         type=str, default=None,
                        help="可选：CMSS 热力图 PNG 输出路径（需要训练好的 GMM）")
    parser.add_argument("--stage",            type=int, default=0,
                        choices=[0, 1, 2],
                        help="--out-mask 时使用的课程阶段：0=A(Easy) 1=B(Mixed) 2=C(Hard)")
    parser.add_argument("--num-samples",      type=int, default=5,
                        help="可视化样本数（默认 5）")
    parser.add_argument("--box-threshold",    type=float, default=0.3)
    parser.add_argument("--text-threshold",   type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer_csma] 设备: {device}")
    print(f"[infer_csma] 数据集模式: {args.dataset}")

    # Phase 4.1：加载处理器与 DINO
    model_id = "IDEA-Research/grounding-dino-tiny"
    text_prompt = "person. car."
    processor = AutoProcessor.from_pretrained(model_id)
    dino: GroundingDinoForObjectDetection = (
        GroundingDinoForObjectDetection.from_pretrained(model_id).to(device)
    )
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False

    # Phase 4.2：加载 CSMA 权重
    cfg = CSMAConfig()
    csma = CSMA(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    csma.load_state_dict(state)
    csma.eval()
    print(f"[infer_csma] 已加载 CSMA 权重: {args.ckpt}")

    # Phase 4.3：加载数据集（支持 flir_v1 / flir_v2 / legacy 三种模式）
    if args.dataset == "flir_v1":
        from src.dataset_flir_v1 import FlirV1PairedDataset, build_flir_v1_category_map

        cat_map, valid_ids = build_flir_v1_category_map(text_prompt)
        dataset = FlirV1PairedDataset(
            root=args.data_root,
            processor=processor,
            text_prompt=text_prompt,
            category_map=cat_map,
            valid_cat_ids=valid_ids,
        )
    elif args.dataset == "flir_v2":
        from src.dataset_flir_v2 import FlirADASV2Dataset, build_flir_v2_category_map

        cat_map, valid_ids = build_flir_v2_category_map(text_prompt)
        dataset = FlirADASV2Dataset(
            root=args.data_root,
            processor=processor,
            text_prompt=text_prompt,
            category_map=cat_map,
            valid_cat_ids=valid_ids,
        )
    else:
        from src.dataset import FlirCocoOverfitDataset, build_coco_category_to_class_index

        cat_map = build_coco_category_to_class_index(text_prompt)
        dataset = FlirCocoOverfitDataset(
            args.data_root, processor, text_prompt, cat_map
        )

    samples = [dataset[i] for i in range(min(args.num_samples, len(dataset)))]
    print(f"[infer_csma] 数据集大小: {len(dataset)}，可视化 {len(samples)} 张")

    # Phase 4.4：多样本对比图
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_multi_sample_grid_csma(
        csma, dino, processor, samples,
        text_prompt, device, args.out,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    print(f"[infer_csma] 已保存多样本对比图: {args.out}")

    # Phase 4.5：CMSS 热力图（可选）
    if args.out_mask is not None:
        enc = processor.tokenizer(text_prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        cmss_sched = CMSSScheduler(cfg)
        sample0 = samples[0]
        ir_pv = sample0["pixel_values"].to(device).unsqueeze(0)

        overlay = visualize_cmss_mask(
            ir_pv=ir_pv,
            csma=csma,
            dino=dino,
            cmss_sched=cmss_sched,
            stage=args.stage,
            device=device,
            input_ids=input_ids,
            attention_mask=attention_mask,
            processor=processor,
        )

        os.makedirs(os.path.dirname(args.out_mask) or ".", exist_ok=True)
        Image.fromarray(overlay).save(args.out_mask)
        print(f"[infer_csma] 已保存 CMSS 热力图: {args.out_mask}")


if __name__ == "__main__":
    main()
