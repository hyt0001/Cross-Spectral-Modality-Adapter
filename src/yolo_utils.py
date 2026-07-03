"""
YOLOv8 冻结检测器工具（CSMA 训练 / 评估用）。

将 ``train_csma.py`` 中 Grounding DINO 相关逻辑替换为 Ultralytics YOLOv8：
  - L_det：伪 RGB → 冻结 YOLO → v8DetectionLoss
  - L_align：YOLO neck P5 特征（展平为 [B, L, D]）上的 CMSS 对齐
  - Val mAP：复用 ``eval_yolo_csma.run_yolo_eval``
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.config import CSMAConfig

# FLIR prompt class_idx（0=person, 1=car）→ COCO YOLO cls id
FLIR_CLASS_IDX_TO_YOLO: Dict[int, int] = {0: 0, 1: 2}


def load_frozen_yolo(
    weights: str,
    device: torch.device,
) -> Tuple[Any, nn.Module]:
    """
    加载 YOLOv8 权重并冻结全部参数。

    Returns:
        (yolo_wrapper, yolo_model)  — wrapper 供 val predict；model 供 loss / hook。
    """
    try:
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
    except ImportError as e:
        raise ImportError("请安装 ultralytics: pip install ultralytics") from e

    yolo_wrapper = YOLO(weights)
    yolo_model: nn.Module = yolo_wrapper.model.to(device)
    yolo_model.args = get_cfg()
    for p in yolo_model.parameters():
        p.requires_grad = False
    yolo_model.eval()
    return yolo_wrapper, yolo_model


def denormalize_for_yolo(
    pixel_values: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """ImageNet 归一化 [B,3,H,W] → YOLO 输入 [0,1]。"""
    m = torch.tensor(mean, device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    return (pixel_values * s + m).clamp(0.0, 1.0)


def labels_to_yolo_targets(
    labels: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    DINO collate labels → YOLO loss batch 字段（不含 img）。

    boxes 为归一化 cxcywh；cls 为 COCO id（person=0, car=2）。
    """
    batch_idx_list: List[int] = []
    cls_list: List[int] = []
    bbox_list: List[torch.Tensor] = []

    for i, lab in enumerate(labels):
        cl = lab["class_labels"]
        boxes = lab["boxes"]
        n = int(cl.shape[0])
        if n == 0:
            continue
        for j in range(n):
            cidx = int(cl[j].item())
            yolo_cls = FLIR_CLASS_IDX_TO_YOLO.get(cidx)
            if yolo_cls is None:
                continue
            batch_idx_list.append(i)
            cls_list.append(yolo_cls)
            bbox_list.append(boxes[j].to(device))

    if not batch_idx_list:
        return {
            "batch_idx": torch.zeros(0, device=device, dtype=torch.float32),
            "cls": torch.zeros(0, device=device, dtype=torch.float32),
            "bboxes": torch.zeros(0, 4, device=device, dtype=torch.float32),
        }

    return {
        "batch_idx": torch.tensor(batch_idx_list, device=device, dtype=torch.float32),
        "cls": torch.tensor(cls_list, device=device, dtype=torch.float32),
        "bboxes": torch.stack(bbox_list).to(dtype=torch.float32),
    }


def flatten_neck_feat(feat_list: List[torch.Tensor], level: int = -1) -> torch.Tensor:
    """取 neck 某一尺度特征并展平为 [B, L, D]（默认 P5）。"""
    f = feat_list[level]
    b, c, h, w = f.shape
    return f.permute(0, 2, 3, 1).reshape(b, h * w, c)


class YoloAlignProjector(nn.Module):
    """
    将 YOLO neck 多尺度特征（P3/P4/P5）线性投影到 proto_dim，
    修复 L_align 的两个核心问题：

      1. 维度不匹配：P5=512-D（YOLOv8m）vs DINO input_proj=256-D (proto_dim)
         → 每尺度独立 Linear(C_i→proto_dim) + 共享 LayerNorm

      2. 单尺度过粗：P5 仅 20×20=400 tokens（640px）vs DINO 多尺度 ≈5440 tokens
         → 拼接 P3+P4+P5，512px 输入时 64²+32²+16²=5376 tokens

    不修改 CSMA，只作为独立对齐模块与 CSMA 一同优化。
    """

    def __init__(self, feat_dims: List[int], proto_dim: int = 256) -> None:
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, proto_dim) for d in feat_dims])
        self.norm = nn.LayerNorm(proto_dim)

    def forward(self, neck_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            neck_feats: list of [B, C_i, H_i, W_i]（P3/P4/P5 顺序）
        Returns:
            [B, L_all, proto_dim]，L_all = ΣH_i*W_i
        """
        tokens: List[torch.Tensor] = []
        for feat, proj in zip(neck_feats, self.projs):
            b, c, h, w = feat.shape
            t = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
            tokens.append(proj(t))
        return self.norm(torch.cat(tokens, dim=1))


def build_yolo_align_projector(
    yolo_model: nn.Module,
    proto_dim: int = 256,
    device: torch.device = torch.device("cpu"),
    img_size: int = 512,
) -> YoloAlignProjector:
    """
    通过 dummy forward 自动探测各尺度 neck 通道数，构建 YoloAlignProjector。

    img_size 须与训练时 processor 输出的最短边一致（默认 512）。
    """
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
    was_training = yolo_model.training
    yolo_model.eval()
    with torch.no_grad():
        neck_feats = _forward_neck_feats(yolo_model, dummy)
    yolo_model.train(was_training)
    feat_dims = [int(f.shape[1]) for f in neck_feats]
    total_tokens = sum(f.shape[2] * f.shape[3] for f in neck_feats)
    print(
        f"[YoloAlignProjector] neck channel dims={feat_dims}  "
        f"total tokens at {img_size}px={total_tokens} (DINO≈5440)  "
        f"proto_dim={proto_dim}"
    )
    return YoloAlignProjector(feat_dims, proto_dim).to(device)


def _forward_neck_feats(yolo_model: nn.Module, img: torch.Tensor) -> List[torch.Tensor]:
    """前向至 Detect 头输入（P3/P4/P5 列表）。"""
    y: List[Any] = []
    x = img
    for m in yolo_model.model[:-1]:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in yolo_model.save else None)
    head = yolo_model.model[-1]
    return [y[j] for j in head.f]


def extract_yolo_backbone_features(
    yolo_model: nn.Module,
    img: torch.Tensor,
    projector: Optional[YoloAlignProjector] = None,
) -> torch.Tensor:
    """
    提取 YOLO neck 特征，供 L_align / CMSS 使用。

    projector=None（旧行为）：仅取 P5，展平为 [B, L_p5, C_p5]。
    projector 传入时：多尺度 P3+P4+P5 → [B, L_all, proto_dim]（推荐）。
    """
    neck_feats = _forward_neck_feats(yolo_model, img)
    if projector is not None:
        return projector(neck_feats)
    return flatten_neck_feat(neck_feats, level=-1)


def build_yolo_det_loss(
    loss_vec: torch.Tensor,
    cfg: CSMAConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    YOLO v8DetectionLoss 三分量 → 与 DINO ``_build_det_loss`` 对齐的标量字典。

    loss_vec: [box, cls, dfl]（已 × batch_size）
    """
    box, cls, dfl = loss_vec[0], loss_vec[1], loss_vec[2]
    loss_det = cls + box * cfg.det_w_bbox + dfl * cfg.det_w_giou
    scalars = {
        "loss_det": float(loss_det.detach().cpu()),
        "loss_ce": float(cls.detach().cpu()),
        "loss_bbox": float(box.detach().cpu()),
        "loss_giou": float(dfl.detach().cpu()),
        "loss_ce_enc": 0.0,
        "loss_bbox_enc": 0.0,
        "loss_giou_enc": 0.0,
    }
    return loss_det, scalars


def yolo_det_forward_with_feats(
    yolo_model: nn.Module,
    img: torch.Tensor,
    yolo_targets: Dict[str, torch.Tensor],
    cfg: CSMAConfig,
    projector: Optional[YoloAlignProjector] = None,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """
    单次前向：L_det + 捕获带梯度的 neck 特征。

    projector=None（旧行为）：feat_ir = P5 展平，[B, L_p5, C_p5]。
    projector 传入时：feat_ir = 多尺度投影，[B, L_all, proto_dim]（推荐）。

    Returns:
        (l_det, det_scalars, feat_ir)
    """
    yolo_batch = {**yolo_targets, "img": img}
    feat_cache: Dict[str, List[torch.Tensor]] = {}

    def _pre_hook(_module: nn.Module, inp: Tuple[Any, ...]) -> None:
        feat_cache["feats"] = inp[0]

    handle = yolo_model.model[-1].register_forward_pre_hook(_pre_hook)
    was_training = yolo_model.training
    yolo_model.train()
    try:
        preds = yolo_model.predict(img)
        loss_vec, _ = yolo_model.loss(yolo_batch, preds=preds)
    finally:
        handle.remove()
        yolo_model.train(was_training)

    l_det, scalars = build_yolo_det_loss(loss_vec, cfg)
    neck_feats = feat_cache["feats"]
    if projector is not None:
        feat_ir = projector(neck_feats)
    else:
        feat_ir = flatten_neck_feat(neck_feats, level=-1)
    return l_det, scalars, feat_ir


def collect_cmss_values_yolo(
    yolo_model: nn.Module,
    csma: nn.Module,
    loader: Any,
    device: torch.device,
    mean: Sequence[float],
    std: Sequence[float],
    max_batches: int = 200,
    projector: Optional[YoloAlignProjector] = None,
) -> "np.ndarray":
    """GMM 重拟合用 CMSS 采样（YOLO 特征版）。"""
    import numpy as np

    csma.eval()
    all_vals: List[np.ndarray] = []
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            if max_batches != -1 and n_batches >= max_batches:
                break
            if "rgb_pixel_values" not in batch:
                continue
            ir_pv = batch["pixel_values"].to(device)
            rgb_pv = batch["rgb_pixel_values"].to(device)

            pseudo_rgb = csma(ir_pv)
            yolo_ir = denormalize_for_yolo(pseudo_rgb, mean, std)
            yolo_rgb = denormalize_for_yolo(rgb_pv, mean, std)

            feat_rgb = extract_yolo_backbone_features(yolo_model, yolo_rgb, projector)
            feat_ir = extract_yolo_backbone_features(yolo_model, yolo_ir, projector)

            from src.cmss_utils import compute_cmss

            cmss_map = compute_cmss(feat_rgb, feat_ir)
            all_vals.append(cmss_map.cpu().numpy().astype(np.float32).flatten())
            n_batches += 1

    csma.train()
    result = np.concatenate(all_vals) if all_vals else np.array([0.5], dtype=np.float32)
    print(f"[collect_cmss_yolo] 采样 {n_batches} batch，获得 {len(result):,} 个 CMSS 值")
    return result


def run_val_map_yolo(
    csma: nn.Module,
    yolo_wrapper: Any,
    processor: Any,
    val_dataset: Any,
    valid_cat_ids: frozenset,
    device: torch.device,
    dataset_mode: str,
    batch_size: int,
    conf: float = 0.05,
) -> Dict[str, Any]:
    """Val 集 YOLO mAP（与 eval_yolo_csma 一致）。"""
    from src.eval_yolo_csma import YOLO_CLS_TO_EVAL_CAT, _build_gt_coco, compute_map, run_yolo_eval

    coco_gt = _build_gt_coco(val_dataset, valid_cat_ids)
    csma.eval()
    try:
        predictions = run_yolo_eval(
            yolo_model=yolo_wrapper,
            dataset=val_dataset,
            device=device,
            batch_size=batch_size,
            num_workers=2,
            conf=conf,
            input_mode="pseudo_rgb",
            dataset_mode=dataset_mode,
            cls_to_eval_cat=YOLO_CLS_TO_EVAL_CAT,
            csma=csma,
            processor=processor,
        )
        metrics = compute_map(coco_gt, predictions)
    finally:
        csma.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics
