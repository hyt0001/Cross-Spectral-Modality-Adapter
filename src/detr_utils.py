"""
DETR (facebook/detr-resnet-50) 冻结检测器工具（CSMA 训练 / 评估用）。

  - L_det：伪 RGB → 冻结 DETR → HF loss_ce / loss_bbox / loss_giou
  - L_align：DETR encoder 入口特征 [B, L, 256]（与 proto_dim 一致，无需 projector）
  - Val mAP：复用 ``eval_detr_baseline.run_detr_eval``
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import DetrForObjectDetection, DetrImageProcessor

from src.config import CSMAConfig

# FLIR class_idx（0=person, 1=car）→ DETR COCO label id
FLIR_CLASS_IDX_TO_DETR: Dict[int, int] = {0: 1, 1: 3}

MODEL_ID_DEFAULT = "facebook/detr-resnet-50"


def load_frozen_detr(
    model_id: str,
    device: torch.device,
) -> Tuple[DetrForObjectDetection, DetrImageProcessor]:
    """加载并冻结 DETR + processor。"""
    proc = DetrImageProcessor.from_pretrained(model_id)
    model = DetrForObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, proc


def gdino_tensor_to_detr_input(
    pixel_values: torch.Tensor,
    gdino_mean: Sequence[float],
    gdino_std: Sequence[float],
    detr_mean: Sequence[float],
    detr_std: Sequence[float],
) -> torch.Tensor:
    """GDINO 归一化 tensor → DETR 归一化（可微，保持 H×W）。"""
    m_g = torch.tensor(gdino_mean, device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    s_g = torch.tensor(gdino_std, device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    m_d = torch.tensor(detr_mean, device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    s_d = torch.tensor(detr_std, device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    img_01 = (pixel_values * s_g + m_g).clamp(0.0, 1.0)
    return (img_01 - m_d) / s_d


def labels_to_detr_labels(
    labels: List[Dict[str, Any]],
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    """GDINO collate labels → DETR HF labels（COCO class id + cxcywh）。"""
    out: List[Dict[str, torch.Tensor]] = []
    for lab in labels:
        cl = lab["class_labels"]
        boxes = lab["boxes"]
        n = int(cl.shape[0])
        if n == 0:
            out.append({
                "class_labels": torch.zeros(0, dtype=torch.long, device=device),
                "boxes": torch.zeros(0, 4, device=device, dtype=torch.float32),
            })
            continue
        detr_cls: List[int] = []
        detr_boxes: List[torch.Tensor] = []
        for j in range(n):
            cidx = int(cl[j].item())
            detr_id = FLIR_CLASS_IDX_TO_DETR.get(cidx)
            if detr_id is None:
                continue
            detr_cls.append(detr_id)
            detr_boxes.append(boxes[j].to(device))
        if not detr_cls:
            out.append({
                "class_labels": torch.zeros(0, dtype=torch.long, device=device),
                "boxes": torch.zeros(0, 4, device=device, dtype=torch.float32),
            })
        else:
            out.append({
                "class_labels": torch.tensor(detr_cls, dtype=torch.long, device=device),
                "boxes": torch.stack(detr_boxes).to(dtype=torch.float32),
            })
    return out


def build_detr_det_loss(
    outputs: Any,
    cfg: CSMAConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """DETR loss_dict → 标量 L_det（无 encoder 辅助项）。"""
    ld = outputs.loss_dict
    loss_det = (
        ld["loss_ce"]
        + ld["loss_bbox"] * cfg.det_w_bbox
        + ld["loss_giou"] * cfg.det_w_giou
    )
    scalars = {
        "loss_det": float(loss_det.detach().cpu()),
        "loss_ce": float(ld["loss_ce"].detach().cpu()),
        "loss_bbox": float(ld["loss_bbox"].detach().cpu()),
        "loss_giou": float(ld["loss_giou"].detach().cpu()),
        "loss_ce_enc": 0.0,
        "loss_bbox_enc": 0.0,
        "loss_giou_enc": 0.0,
    }
    return loss_det, scalars


def extract_detr_encoder_features(
    detr_model: DetrForObjectDetection,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """Hook DETR encoder 入口，返回 [B, L, 256]。"""
    cache: Dict[str, torch.Tensor] = {}

    def _hook(_module: nn.Module, inp: tuple, out: Any) -> None:
        if inp and isinstance(inp[0], torch.Tensor):
            cache["feat"] = inp[0]
        elif hasattr(out, "last_hidden_state"):
            cache["feat"] = out.last_hidden_state

    handle = detr_model.model.encoder.register_forward_hook(_hook)
    try:
        detr_model(pixel_values=pixel_values)
    finally:
        handle.remove()

    assert "feat" in cache, "DETR encoder hook 未触发"
    return cache["feat"]


def detr_det_forward_with_feats(
    detr_model: DetrForObjectDetection,
    pixel_values: torch.Tensor,
    detr_labels: List[Dict[str, torch.Tensor]],
    cfg: CSMAConfig,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """
    单次前向：L_det + 捕获 encoder 特征（带梯度）。

    Returns:
        (l_det, det_scalars, feat_ir)  feat_ir: [B, L, 256]
    """
    cache: Dict[str, torch.Tensor] = {}

    def _hook(_module: nn.Module, inp: tuple, out: Any) -> None:
        if inp and isinstance(inp[0], torch.Tensor):
            cache["feat"] = inp[0]
        elif hasattr(out, "last_hidden_state"):
            cache["feat"] = out.last_hidden_state

    handle = detr_model.model.encoder.register_forward_hook(_hook)
    was_training = detr_model.training
    detr_model.train()  # loss 需要 train 模式
    try:
        outputs = detr_model(pixel_values=pixel_values, labels=detr_labels)
        if outputs.loss is None:
            raise RuntimeError("DETR outputs.loss 为 None，请检查 labels 格式。")
        l_det, scalars = build_detr_det_loss(outputs, cfg)
    finally:
        handle.remove()
        detr_model.train(was_training)

    assert "feat" in cache, "DETR encoder hook 未触发"
    return l_det, scalars, cache["feat"]


def collect_cmss_values_detr(
    detr_model: DetrForObjectDetection,
    csma: nn.Module,
    loader: Any,
    device: torch.device,
    gdino_mean: Sequence[float],
    gdino_std: Sequence[float],
    detr_mean: Sequence[float],
    detr_std: Sequence[float],
    max_batches: int = 200,
) -> "np.ndarray":
    """GMM 重拟合用 CMSS 采样（DETR encoder 特征）。"""
    import numpy as np

    from src.cmss_utils import compute_cmss

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
            detr_ir = gdino_tensor_to_detr_input(
                pseudo_rgb, gdino_mean, gdino_std, detr_mean, detr_std,
            )
            detr_rgb = gdino_tensor_to_detr_input(
                rgb_pv, gdino_mean, gdino_std, detr_mean, detr_std,
            )

            feat_rgb = extract_detr_encoder_features(detr_model, detr_rgb)
            feat_ir = extract_detr_encoder_features(detr_model, detr_ir)

            cmss_map = compute_cmss(feat_rgb, feat_ir)
            all_vals.append(cmss_map.cpu().numpy().astype(np.float32).flatten())
            n_batches += 1

    csma.train()
    result = np.concatenate(all_vals) if all_vals else np.array([0.5], dtype=np.float32)
    print(f"[collect_cmss_detr] 采样 {n_batches} batch，获得 {len(result):,} 个 CMSS 值")
    return result


def run_val_map_detr(
    csma: nn.Module,
    detr_model: DetrForObjectDetection,
    detr_proc: DetrImageProcessor,
    gdino_processor: Any,
    val_dataset: Any,
    valid_cat_ids: frozenset,
    device: torch.device,
    batch_size: int,
    threshold: float = 0.05,
) -> Dict[str, Any]:
    """Val 集 DETR mAP（与 eval_detr_baseline 一致）。"""
    from src.eval_csma import _build_gt_coco, compute_map
    from src.eval_detr_baseline import run_detr_eval

    coco_gt = _build_gt_coco(val_dataset, valid_cat_ids)
    csma.eval()
    try:
        predictions = run_detr_eval(
            model=detr_model,
            processor=detr_proc,
            dataset=val_dataset,
            device=device,
            batch_size=batch_size,
            num_workers=2,
            threshold=threshold,
            input_mode="pseudo_rgb",
            csma=csma,
            gdino_processor=gdino_processor,
        )
        metrics = compute_map(coco_gt, predictions)
    finally:
        csma.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics
