"""
CSMA 训练主程序。

在 src/train_demo.py 骨架基础上新增：
  - CMSS 引导的 L_align（跨模态特征蒸馏）
  - CMSSScheduler 三阶段渐进课程
  - FlirPairedDataset + collate_paired（IR + RGB 配对）
其余框架（SwanLab 日志、梯度检查、可视化、权重保存）与 train_demo.py 保持一致。

对应 docs/TD.md §1.5，docs/architecture.md §7。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from contextlib import contextmanager

@contextmanager
def _maybe_autocast(enabled: bool):
    """兼容多版本 PyTorch 的 AMP context manager。"""
    if enabled:
        try:
            # PyTorch 2.x 推荐写法
            with torch.amp.autocast("cuda"):
                yield
        except TypeError:
            # PyTorch 1.x 兼容写法
            with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                yield
    else:
        yield
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from src.cmss_utils import CMSSScheduler, build_cmss_mask, compute_cmss
from src.config import CSMAConfig
from src.csma import CSMA
from src.dino_vision_bridge import (
    extract_vision_tokens,
    forward_dino_with_feature_adapter,
)
from src.feature_adapter import FeatureAdapter
from src.dataset import build_coco_category_to_class_index
from src.dataset_flir_v1 import FlirV1PairedDataset, build_flir_v1_category_map, collate_flir_v1
from src.dataset_flir_v2 import FlirADASV2Dataset, build_flir_v2_category_map, collate_flir_v2
from src.dataset_paired import FlirPairedDataset, collate_paired
from src.infer_vis import save_multi_sample_grid
from src.model_ema import ModelEMA
from src.pseudo_losses import identity_loss, logit_regularization, total_variation_loss

_emergency_save_requested = False


def _request_emergency_save(signum: int, frame: Any) -> None:
    global _emergency_save_requested
    _emergency_save_requested = True
    print(f"[train_csma] 收到信号 {signum}，本 epoch 结束后将写入 latest.pt / emergency.pt")


def _load_csma_state_dict(ckpt_path: str, map_location: Any) -> Dict[str, torch.Tensor]:
    raw = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    if isinstance(raw, dict) and "csma" in raw:
        return raw["csma"]
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"无法从 checkpoint 解析 state_dict: {ckpt_path}")


def _load_feature_adapter_state_dict(ckpt_path: str, map_location: Any) -> Dict[str, torch.Tensor]:
    raw = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    if isinstance(raw, dict) and "feature_adapter" in raw:
        return raw["feature_adapter"]
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"无法从 checkpoint 解析 FeatureAdapter state_dict: {ckpt_path}")


def _save_adapter_weights(path: str, module: nn.Module) -> None:
    """保存适配器 state_dict（像素/特征模式通用）。"""
    torch.save(module.state_dict(), path)


def _save_ema_weights(path: str, ema: ModelEMA) -> None:
    """保存 EMA shadow 权重。"""
    torch.save(ema.state_dict(), path)


def _save_csma_weights(path: str, csma: nn.Module) -> None:
    _save_adapter_weights(path, csma)


def _write_latest_meta(ckpt_dir: str, epoch: int) -> None:
    meta = {
        "epoch": epoch,
        "completed_epochs": epoch + 1,
        "note": "epoch 为 0-based；completed_epochs 为已完成的训练轮数",
    }
    with open(os.path.join(ckpt_dir, "latest_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _default_val_data_root(train_root: str) -> str:
    """FLIR_License/train → FLIR_License/val"""
    root = train_root.rstrip(os.sep)
    if os.path.basename(root) == "train":
        return os.path.join(os.path.dirname(root), "val")
    return os.path.join(os.path.dirname(root), "val")


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _val_metric_score(metrics: Dict[str, Any], val_metric: str) -> float:
    """
    从评测指标字典提取早停分数。

    Args:
        metrics: compute_map 返回值。
        val_metric: ``map_50`` 或 ``person_car_mean``。

    Returns:
        用于早停比较的标量（越大越好）。
    """
    if val_metric == "person_car_mean":
        if "person_car_mean" in metrics:
            return float(metrics["person_car_mean"])
        ap_p = float(metrics.get("ap_person", 0.0))
        ap_c = float(metrics.get("ap_car", 0.0))
        return (ap_p + ap_c) / 2.0
    return float(metrics["map_50"])


def _update_class_ap_bests(
    metrics: Dict[str, Any],
    best_ap_person: float,
    best_ap_car: float,
) -> Tuple[bool, bool, float, float]:
    """
    检查 person / car 单项 AP@0.5 是否创新高。

    Args:
        metrics: val 评测指标（含 ap_person、ap_car）。
        best_ap_person: 历史最佳 person AP。
        best_ap_car: 历史最佳 car AP。

    Returns:
        (person是否新高, car是否新高, 更新后best_person, 更新后best_car)
    """
    ap_p = metrics.get("ap_person")
    ap_c = metrics.get("ap_car")
    cur_p = float(ap_p) if isinstance(ap_p, (int, float)) else -1.0
    cur_c = float(ap_c) if isinstance(ap_c, (int, float)) else -1.0
    improved_p = cur_p > best_ap_person
    improved_c = cur_c > best_ap_car
    return (
        improved_p,
        improved_c,
        cur_p if improved_p else best_ap_person,
        cur_c if improved_c else best_ap_car,
    )


def _run_val_map(
    dino: nn.Module,
    processor: Any,
    val_dataset: Any,
    valid_cat_ids: frozenset,
    device: torch.device,
    dataset_mode: str,
    text_prompt: str,
    batch_size: int,
    box_threshold: float,
    text_threshold: float,
    adapter_mode: str,
    csma: Optional[nn.Module] = None,
    feature_adapter: Optional[FeatureAdapter] = None,
) -> Dict[str, Any]:
    """在 val 上跑一轮 mAP（复用 eval_csma）。"""
    from src.eval_csma import _build_gt_coco, compute_map, run_eval

    coco_gt = _build_gt_coco(val_dataset, valid_cat_ids, dataset_mode)
    trainable = feature_adapter if adapter_mode == "feature" else csma
    assert trainable is not None
    trainable.eval()
    try:
        predictions = run_eval(
            dino=dino,
            processor=processor,
            dataset=val_dataset,
            device=device,
            text_prompt=text_prompt,
            batch_size=batch_size,
            num_workers=2,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            dataset_mode=dataset_mode,
            csma=csma if adapter_mode == "pixel" else None,
            feature_adapter=feature_adapter if adapter_mode == "feature" else None,
            adapter_mode=adapter_mode,
        )
        metrics = compute_map(coco_gt, predictions, dataset_mode=dataset_mode)
    finally:
        trainable.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: 辅助函数（复用 / 适配自 train_demo.py）
# ──────────────────────────────────────────────────────────────────────────────

def _move_labels_to_device(
    labels: List[Dict[str, Any]], device: torch.device
) -> List[Dict[str, Any]]:
    """将 labels list 中所有 Tensor 移至指定设备。"""
    out: List[Dict[str, Any]] = []
    for lab in labels:
        entry: Dict[str, Any] = {}
        for k, v in lab.items():
            entry[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        out.append(entry)
    return out


def _build_swanlab_logger(
    enable: bool,
    project: str,
    run_name: str,
    config: Dict[str, Any],
) -> Optional[Any]:
    """按需初始化 SwanLab；未安装或关闭时返回 None。"""
    if not enable:
        return None
    try:
        import swanlab  # type: ignore
        return swanlab.init(project=project, experiment_name=run_name, config=config)
    except Exception as exc:
        print(f"[SwanLab] 初始化失败，继续本地训练: {exc}")
        return None


def _build_det_loss(
    outputs: Any,
    cfg: CSMAConfig,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    使用 CSMAConfig 中的权重构建检测损失 L_det，对应 docs/TD.md §2.3。

    L_det = L_ce + w_bbox*L_bbox + w_giou*L_giou
          + w_ce_enc*L_ce_enc + w_bbox_enc*L_bbox_enc + w_giou_enc*L_giou_enc
    """
    ld = outputs.loss_dict
    loss_det = (
        ld["loss_ce"]
        + ld["loss_bbox"]     * cfg.det_w_bbox
        + ld["loss_giou"]     * cfg.det_w_giou
        + ld["loss_ce_enc"]   * cfg.det_w_ce_enc
        + ld["loss_bbox_enc"] * cfg.det_w_bbox_enc
        + ld["loss_giou_enc"] * cfg.det_w_giou_enc
    )
    scalars = {
        "loss_det":       float(loss_det.detach().cpu()),
        "loss_ce":        float(ld["loss_ce"].detach().cpu()),
        "loss_bbox":      float(ld["loss_bbox"].detach().cpu()),
        "loss_giou":      float(ld["loss_giou"].detach().cpu()),
        "loss_ce_enc":    float(ld["loss_ce_enc"].detach().cpu()),
        "loss_bbox_enc":  float(ld["loss_bbox_enc"].detach().cpu()),
        "loss_giou_enc":  float(ld["loss_giou_enc"].detach().cpu()),
    }
    return loss_det, scalars


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: DINO 特征提取（forward hook）
# ──────────────────────────────────────────────────────────────────────────────

def extract_dino_backbone_features(
    dino_model: GroundingDinoForObjectDetection,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    通过 forward hook 提取 DINO input_proj 投影后、encoder 处理前的多尺度特征。

    对应 TD-06 决策：特征提取点为 model.backbone 输出后、送入 model.encoder 前的
    projected_features，形状 [B, L_total, 256]（L_total = 所有尺度 Patch 总数）。

    Args:
        dino_model:      冻结的 GroundingDinoForObjectDetection。
        pixel_values:    [B, 3, H, W]，ImageNet 归一化图像。
        input_ids:       文本 tokenizer 输出，[1, T] 扩展至 [B, T]。
        attention_mask:  文本注意力掩码，[B, T]。

    Returns:
        features: [B, L_total, 256]，encoder 入口处特征（含梯度图，调用方按需 detach）。
    """
    hook_output: Dict[str, torch.Tensor] = {}

    def _hook(module: nn.Module, inp: tuple, out: Any) -> None:
        # 优先从输入捕获（encoder 入口特征）；
        # 新版 transformers 以全关键字参数调用时 inp 为空，改从输出取。
        if inp and isinstance(inp[0], torch.Tensor):
            hook_output["feat"] = inp[0]
        else:
            # GroundingDinoEncoderOutput 的视觉特征字段
            if hasattr(out, "last_hidden_state_vision"):
                hook_output["feat"] = out.last_hidden_state_vision
            elif hasattr(out, "last_hidden_state"):
                hook_output["feat"] = out.last_hidden_state
            elif isinstance(out, (tuple, list)) and len(out) > 0:
                # 取第一个 Tensor
                for o in out:
                    if isinstance(o, torch.Tensor):
                        hook_output["feat"] = o
                        break
            else:
                hook_output["feat"] = out

    handle = dino_model.model.encoder.register_forward_hook(_hook)
    try:
        bsz = pixel_values.shape[0]
        dino_model(
            pixel_values=pixel_values,
            pixel_mask=torch.ones(
                bsz, pixel_values.shape[2], pixel_values.shape[3],
                dtype=torch.long, device=pixel_values.device
            ),
            input_ids=input_ids.expand(bsz, -1),
            attention_mask=attention_mask.expand(bsz, -1),
        )
    finally:
        handle.remove()

    assert "feat" in hook_output, "encoder hook 未触发，请检查 DINO 模型结构"
    return hook_output["feat"]


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: L_align 计算
# ──────────────────────────────────────────────────────────────────────────────

def compute_align_loss(
    feat_ir: torch.Tensor,
    feat_rgb: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    仅在未被掩蔽的 Patch（mask=0）上计算余弦相似度对齐损失。

    L_align = 1 - mean( cosine_sim(F_ir[Ω], sg(F_rgb[Ω])) )
    其中 Ω = {i | mask[i] == 0}（保留的 Patch 集合）

    改用余弦相似度而非 MSE 的原因：
    - DINO input_proj 后特征 L2 norm ≈ 1，MSE 输出 ≈ 0.004（与 L_det≈600 差 5 个数量级）
    - 余弦损失值域 [0, 2]，量级与 L_det 相近，梯度贡献不再被淹没

    Args:
        feat_ir:  [B, L, D]  有梯度，经 CSMA → DINO 路径。
        feat_rgb: [B, L, D]  须已 detach（stop_gradient）。
        mask:     [B, L]     0=保留，1=掩蔽；由 build_cmss_mask 生成。

    Returns:
        标量余弦损失 ∈ [0, 2]；若全部 Patch 被掩蔽则返回 0。
    """
    assert feat_ir.shape == feat_rgb.shape, "feat_ir 与 feat_rgb 形状不一致"
    unmasked = (mask == 0)                              # [B, L] bool
    if not unmasked.any():
        return torch.tensor(0.0, device=feat_ir.device, requires_grad=True)
    ir_sel   = F.normalize(feat_ir[unmasked], dim=-1)
    rgb_sel  = F.normalize(feat_rgb[unmasked].detach(), dim=-1)
    return 1.0 - (ir_sel * rgb_sel).sum(dim=-1).mean()


def compute_align_loss_weighted(
    feat_ir: torch.Tensor,
    feat_rgb: torch.Tensor,
    mask: torch.Tensor,
    patch_weight: torch.Tensor,
) -> torch.Tensor:
    """
    带 Patch 级权重的余弦对齐损失（GT bbox 区域加权版）。

    L_align = 1 - weighted_mean( cosine_sim(F_ir[Ω], sg(F_rgb[Ω])) )
    其中 Ω = {i | mask[i] == 0}，权重由 patch_weight 给出。

    Args:
        feat_ir:      [B, L, D]  有梯度。
        feat_rgb:     [B, L, D]  须已 detach。
        mask:         [B, L]     0=保留，1=掩蔽。
        patch_weight: [B, L]     各 Patch 的对齐权重（>0），背景=1.0，目标区域>1.0。

    Returns:
        标量余弦损失；若全部 Patch 被掩蔽则返回 0。
    """
    assert feat_ir.shape == feat_rgb.shape, "feat_ir 与 feat_rgb 形状不一致"
    assert patch_weight.shape == mask.shape, "patch_weight 与 mask 形状不一致"

    unmasked = (mask == 0)
    if not unmasked.any():
        return torch.tensor(0.0, device=feat_ir.device, requires_grad=True)

    ir_sel  = F.normalize(feat_ir[unmasked], dim=-1)
    rgb_sel = F.normalize(feat_rgb[unmasked].detach(), dim=-1)
    cos_sim = (ir_sel * rgb_sel).sum(dim=-1)            # [N_unmasked]

    w = patch_weight[unmasked].to(feat_ir.device).clamp(min=1e-6)
    return 1.0 - (cos_sim * w).sum() / w.sum()


def build_bbox_patch_weight(
    labels: List[Dict[str, Any]],
    feat_len: int,
    img_hw: tuple[int, int],
    bbox_weight: float,
) -> torch.Tensor:
    """
    将 GT bbox（cxcywh 归一化）映射到多尺度 Patch 网格，生成 [B, L] 权重矩阵。

    GroundingDINO Tiny 使用 stride=[8,16,32] 三尺度特征，L = sum(H/s * W/s)。
    bbox 区域内的 Patch 权重 = bbox_weight；其余 = 1.0。
    若 feat_len 与推算不符则退化为全 1.0 均匀权重。

    Args:
        labels:      batch 内各样本 label dict，含 "boxes" [N,4] cxcywh 归一化。
        feat_len:    L —— encoder 入口 token 总数。
        img_hw:      (H, W) —— canonical 输入图像尺寸。
        bbox_weight: bbox 区域 patch 的权重倍率（>1）。

    Returns:
        weight: [B, L] float32 CPU Tensor。
    """
    B = len(labels)
    H, W = img_hw
    strides = [8, 16, 32]
    grids = [(max(H // s, 1), max(W // s, 1)) for s in strides]
    grid_lens = [gh * gw for gh, gw in grids]
    total = sum(grid_lens)

    weight = torch.ones(B, feat_len, dtype=torch.float32)
    if total != feat_len:
        # 尺寸不匹配（Swin backbone padding 差异），退化为均匀权重
        return weight

    for b_idx, lab in enumerate(labels):
        boxes: Optional[torch.Tensor] = lab.get("boxes", None)
        if boxes is None or boxes.numel() == 0:
            continue
        boxes = boxes.float().cpu()
        # cxcywh → xyxy 归一化
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - bw * 0.5).clamp(0.0, 1.0)
        y1 = (cy - bh * 0.5).clamp(0.0, 1.0)
        x2 = (cx + bw * 0.5).clamp(0.0, 1.0)
        y2 = (cy + bh * 0.5).clamp(0.0, 1.0)

        offset = 0
        for (gh, gw), glen in zip(grids, grid_lens):
            # patch 中心坐标归一化 [gh, gw]
            pc_y = (torch.arange(gh, dtype=torch.float32) + 0.5) / gh
            pc_x = (torch.arange(gw, dtype=torch.float32) + 0.5) / gw
            py, px = torch.meshgrid(pc_y, pc_x, indexing="ij")
            py = py.reshape(-1)  # [glen]
            px = px.reshape(-1)  # [glen]

            for n in range(boxes.shape[0]):
                inside = (
                    (px >= x1[n]) & (px <= x2[n]) &
                    (py >= y1[n]) & (py <= y2[n])
                )
                weight[b_idx, offset:offset + glen] = torch.where(
                    inside,
                    torch.full((glen,), bbox_weight),
                    weight[b_idx, offset:offset + glen],
                )
            offset += glen

    return weight


def extract_dino_multilayer_features(
    dino_model: GroundingDinoForObjectDetection,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: List[int],
) -> List[torch.Tensor]:
    """
    通过 forward hook 同时提取 DINO encoder 多个中间层的输出特征。

    每个 encoder layer 的输出第一个 Tensor 即视觉 token 特征 [B, L, D]。
    该设计保持 CSMA 即插即用性：多层对齐仅在训练期加钩子，推理时无需修改。

    Args:
        dino_model:    冻结的 GroundingDinoForObjectDetection。
        pixel_values:  [B, 3, H, W]，ImageNet 归一化。
        input_ids:     [1, T] 或 [B, T]。
        attention_mask:[1, T] 或 [B, T]。
        layer_indices: encoder 层索引（0-based），如 [1, 3, 5]。

    Returns:
        feats: len(layer_indices) 个 [B, L, D] Tensor 的列表，顺序与 layer_indices 一致。
               调用方按需 detach。
    """
    cache: Dict[int, torch.Tensor] = {}

    def _make_hook(idx: int):
        def _hook(module: nn.Module, inp: tuple, out: Any) -> None:
            # GroundingDinoEncoderLayer 输出为 tuple，第一个元素是视觉 hidden state
            if isinstance(out, (tuple, list)):
                for o in out:
                    if isinstance(o, torch.Tensor) and o.dim() == 3:
                        cache[idx] = o
                        return
            elif isinstance(out, torch.Tensor):
                cache[idx] = out
        return _hook

    enc = dino_model.model.encoder
    handles = []
    for li in layer_indices:
        assert 0 <= li < len(enc.layers), (
            f"layer_indices 中的 {li} 超出范围 [0, {len(enc.layers) - 1}]"
        )
        handles.append(enc.layers[li].register_forward_hook(_make_hook(li)))

    try:
        bsz = pixel_values.shape[0]
        pm = torch.ones(
            bsz, pixel_values.shape[2], pixel_values.shape[3],
            dtype=torch.long, device=pixel_values.device,
        )
        dino_model(
            pixel_values=pixel_values,
            pixel_mask=pm,
            input_ids=input_ids.expand(bsz, -1),
            attention_mask=attention_mask.expand(bsz, -1),
        )
    finally:
        for h in handles:
            h.remove()

    feats = []
    for li in layer_indices:
        assert li in cache, f"encoder layer {li} hook 未触发"
        feats.append(cache[li])
    return feats


def extract_dino_rgb_features_all(
    dino_model: GroundingDinoForObjectDetection,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: List[int],
) -> tuple[torch.Tensor, List[torch.Tensor]]:
    """
    单次 DINO 前向同时捕获 encoder 入口特征（供 CMSS）+ 各中间层特征（供多层对齐）。

    将原本两次独立的 DINO 前向合并为一次，减少 RGB 路径内存 ~50%。

    Args:
        dino_model:    冻结的 GroundingDinoForObjectDetection。
        pixel_values:  [B, 3, H, W]。
        input_ids:     [1, T] 或 [B, T]。
        attention_mask:[1, T] 或 [B, T]。
        layer_indices: encoder 层索引（0-based）。

    Returns:
        entry_feat:  [B, L, D]，encoder 入口特征（与 extract_dino_backbone_features 一致）。
        layer_feats: len(layer_indices) 个 [B, L, D] Tensor 列表（未触发层被跳过）。
    """
    entry_cache: Dict[str, torch.Tensor] = {}
    layer_cache: Dict[int, torch.Tensor] = {}

    def _entry_hook(module: nn.Module, inp: tuple, out: Any) -> None:
        if inp and isinstance(inp[0], torch.Tensor):
            entry_cache["feat"] = inp[0]
        elif hasattr(out, "last_hidden_state_vision"):
            entry_cache["feat"] = out.last_hidden_state_vision
        elif isinstance(out, (tuple, list)):
            for o in out:
                if isinstance(o, torch.Tensor):
                    entry_cache["feat"] = o
                    break
        elif isinstance(out, torch.Tensor):
            entry_cache["feat"] = out

    def _make_layer_hook(li: int):
        def _hook(module: nn.Module, inp: tuple, out: Any) -> None:
            if isinstance(out, (tuple, list)):
                for o in out:
                    if isinstance(o, torch.Tensor) and o.dim() == 3:
                        layer_cache[li] = o
                        return
            elif isinstance(out, torch.Tensor) and out.dim() == 3:
                layer_cache[li] = out
        return _hook

    enc = dino_model.model.encoder
    _handles: List[Any] = [enc.register_forward_hook(_entry_hook)]
    for li in layer_indices:
        if 0 <= li < len(enc.layers):
            _handles.append(enc.layers[li].register_forward_hook(_make_layer_hook(li)))

    try:
        bsz = pixel_values.shape[0]
        pm = torch.ones(
            bsz, pixel_values.shape[2], pixel_values.shape[3],
            dtype=torch.long, device=pixel_values.device,
        )
        dino_model(
            pixel_values=pixel_values,
            pixel_mask=pm,
            input_ids=input_ids.expand(bsz, -1),
            attention_mask=attention_mask.expand(bsz, -1),
        )
    finally:
        for h in _handles:
            h.remove()

    assert "feat" in entry_cache, "encoder entry hook 未触发，请检查 DINO 模型结构"
    layer_feats = [layer_cache[li] for li in layer_indices if li in layer_cache]
    return entry_cache["feat"], layer_feats


def _effective_loss_weights(
    cfg: CSMAConfig,
    lambda_align: float,
    lambda_det: float,
) -> tuple[float, float]:
    """
    按 loss_mode 返回实际用于反传的 (λ_align, λ_det)。

    det_only：仅 L_det；align_only：仅 L_align；full：两者按课程权重。
    """
    if cfg.loss_mode == "det_only":
        return 0.0, max(lambda_det, 0.05)
    if cfg.loss_mode == "align_only":
        return lambda_align, 0.0
    return lambda_align, max(lambda_det, 0.05)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: GMM 全量 CMSS 收集
# ──────────────────────────────────────────────────────────────────────────────

def collect_cmss_values(
    dino_model: GroundingDinoForObjectDetection,
    csma: CSMA,
    loader: DataLoader,
    device: torch.device,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_batches: int = 200,
) -> np.ndarray:
    """
    随机采样训练集子集（torch.no_grad），收集 Patch 级 CMSS 值供 GMM 重新拟合。

    Args:
        max_batches: 最多采样的 batch 数；-1 表示遍历全量。
                     200 batch（≈1600 张）统计上足够拟合 3 分量 GMM，
                     同时将每次 GMM 拟合时间控制在 2 分钟以内。

    Returns:
        cmss_vals: 1D float32 numpy 数组，形状 [N_total_patches]。
    """
    csma.eval()
    all_vals: List[np.ndarray] = []
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            if max_batches != -1 and n_batches >= max_batches:
                break
            if "rgb_pixel_values" not in batch:
                continue
            ir_pv  = batch["pixel_values"].to(device)
            rgb_pv = batch["rgb_pixel_values"].to(device)

            pseudo_rgb = csma(ir_pv)
            feat_rgb = extract_dino_backbone_features(
                dino_model, rgb_pv, input_ids, attention_mask
            )
            feat_ir = extract_dino_backbone_features(
                dino_model, pseudo_rgb, input_ids, attention_mask
            )
            cmss_map = compute_cmss(feat_rgb, feat_ir)      # [B, L]
            all_vals.append(cmss_map.cpu().numpy().astype(np.float32).flatten())
            n_batches += 1

    csma.train()
    result = np.concatenate(all_vals) if all_vals else np.array([0.5], dtype=np.float32)
    print(f"[collect_cmss] 采样 {n_batches} batch，获得 {len(result):,} 个 CMSS 值")
    return result


def collect_cmss_values_feature(
    dino_model: GroundingDinoForObjectDetection,
    feature_adapter: FeatureAdapter,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 200,
) -> np.ndarray:
    """
    特征模式：收集 IR 适配 token 与 RGB token 的 CMSS 值。

    Returns:
        cmss_vals: 1D float32 numpy 数组。
    """
    feature_adapter.eval()
    all_vals: List[np.ndarray] = []
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            if max_batches != -1 and n_batches >= max_batches:
                break
            if "rgb_pixel_values" not in batch:
                continue
            ir_pv  = batch["pixel_values"].to(device)
            rgb_pv = batch["rgb_pixel_values"].to(device)
            pm     = batch["pixel_mask"].to(device)

            bundle_rgb = extract_vision_tokens(dino_model, rgb_pv, pm)
            bundle_ir  = extract_vision_tokens(dino_model, ir_pv, pm)
            feat_ir    = feature_adapter(bundle_ir.vision_features)
            cmss_map   = compute_cmss(bundle_rgb.vision_features, feat_ir)
            all_vals.append(cmss_map.cpu().numpy().astype(np.float32).flatten())
            n_batches += 1

    feature_adapter.train()
    result = np.concatenate(all_vals) if all_vals else np.array([0.5], dtype=np.float32)
    print(f"[collect_cmss/feature] 采样 {n_batches} batch，获得 {len(result):,} 个 CMSS 值")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5: 主训练入口
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global _emergency_save_requested

    parser = argparse.ArgumentParser(description="CSMA 训练：冻结 DINO + 可训练 CSMA + GMM-CMSS L_align")
    parser.add_argument("--dataset",       type=str, default="flir_v1",
                        choices=["legacy", "flir_v1", "flir_v2", "m3fd"],
                        help="数据集类型：flir_v1=FLIR_License 配对数据集（默认）；"
                             "flir_v2=FLIR_ADAS_v2 无配对；legacy=旧版 train/ 目录；"
                             "m3fd=M3FD 多类别配对数据集（ir/ + vi/）")
    parser.add_argument("--data-root",     type=str, default="FLIR_License/train",
                        help="flir_v1: split 目录（含 thermal_annotations.json + thermal_8_bit/ + RGB/）；"
                             "flir_v2: thermal split 目录（含 coco.json + data/）；"
                             "legacy: IR 目录（含 _annotations.coco.json）")
    parser.add_argument("--rgb-data-root", type=str, default="train/rgb",   help="RGB 配对图像目录（仅 legacy 模式）")
    parser.add_argument("--out-dir",       type=str, default="outputs_csma",help="ckpt / logs / vis 输出根目录")
    parser.add_argument("--epochs",        type=int, default=None,          help="覆盖 total_epochs")
    parser.add_argument("--batch-size",    type=int, default=None,          help="覆盖 batch_size")
    parser.add_argument("--lr",            type=float, default=None,        help="覆盖学习率")
    parser.add_argument("--loss-mode",     type=str, default=None,
                        choices=["full", "det_only", "align_only"],
                        help="覆盖 loss_mode（flir_v2 模式默认强制为 det_only）")
    parser.add_argument("--gmm-batches",   type=int, default=None,
                        help="GMM 采样 batch 数上限（-1=全量；默认读 config gmm_max_batches=200）")
    parser.add_argument("--max-steps",     type=int, default=None,
                        help="每 epoch 最大训练步数（-1=全量）；smoke test 可设为 20")
    parser.add_argument("--use-swanlab",   action="store_true",             help="启用 SwanLab 记录")
    parser.add_argument("--swanlab-project",  type=str, default="csma-training")
    parser.add_argument("--swanlab-run-name", type=str, default="csma-run")
    parser.add_argument("--init-ckpt",     type=str, default=None,
                        help="从已有 CSMA 权重继续训练（仅加载 csma，不含优化器）")
    parser.add_argument("--start-epoch",   type=int, default=0,
                        help="起始 epoch（0-based）；需与 --init-ckpt 对应，如从 epoch_0030 续训则填 31")
    parser.add_argument("--val-early-stop", action="store_true",
                        help="val 评测并保存 best_stage1.pt；配合 --val-stop-patience 可提前结束训练")
    parser.add_argument(
        "--val-stop-patience", type=int, default=0,
        help="连续 N 次 val 中 person 与 car 单项 AP 均未创新高则停止（0=不启用）",
    )
    parser.add_argument(
        "--val-stop-min-epochs", type=int, default=0,
        help="早停生效前至少训练的 epoch 数（0-based，含）",
    )
    parser.add_argument("--val-data-root", type=str, default=None,
                        help="val 目录；默认将 train 路径中的 train 替换为 val")
    parser.add_argument("--val-start",     type=int, default=25,
                        help="开始 val 评测的 epoch（0-based，含）")
    parser.add_argument("--val-end",       type=int, default=33,
                        help="结束 val 评测的 epoch（0-based，含）；对应 stage1 末段")
    parser.add_argument("--val-every",     type=int, default=1,
                        help="在 [val-start,val-end] 内每隔 N 个 epoch 评一次")
    parser.add_argument("--val-batch-size", type=int, default=4,
                        help="val 推理 batch size")
    parser.add_argument("--val-box-threshold",  type=float, default=0.05)
    parser.add_argument("--val-text-threshold", type=float, default=0.05)
    parser.add_argument("--stop-after-stage1", action="store_true",
                        help="完成 Mixed 末轮后停止训练，跳过 Hard（避免破坏伪 RGB）")
    parser.add_argument("--hard-max-epochs", type=int, default=None,
                        help="Hard 阶段最多训练 N 个 epoch（默认 2T/3 起至结束；如 5 表示仅最后 5 轮）")
    parser.add_argument("--stage-boundaries", type=str, default=None,
                        help="手动指定阶段边界 'easy_end,mixed_end'，0-based 下一 stage 起始 epoch")
    parser.add_argument("--val-manual", action="store_true",
                        help="使用 --val-start/--val-end 原值；默认按课程自动对齐 Mixed 末段")
    parser.add_argument(
        "--val-every-epoch", action="store_true",
        help="每个 epoch 结束后在 val 上评测并更新 best_stage1（M3FD 推荐）",
    )
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="学习率线性 warmup 轮数（默认 0 = 不 warmup）；"
                             "建议 2–5，保护初始化不被大梯度冲垮")
    parser.add_argument("--vis-every",    type=int, default=None,
                        help="覆盖 cfg.vis_every：每隔 N epoch 保存 ckpt + 可视化（默认 10）")
    parser.add_argument("--text-prompt",  type=str, default=None,
                        help="覆盖 cfg.text_prompt；M3FD 推荐 'person. car.'")
    parser.add_argument("--lambda-recon", type=float, default=None,
                        help="覆盖 cfg.lambda_recon；MSE(pseudo, RGB)")
    parser.add_argument("--lambda-id", type=float, default=None,
                        help="MSE(pseudo, IR) 权重；Final Model 建议 0.005")
    parser.add_argument("--lambda-tv", type=float, default=None,
                        help="pseudo TV 平滑损失权重；Final Model 建议 0.05")
    parser.add_argument("--lambda-logit-reg", type=float, default=None,
                        help="检测 logit 均值正则权重；Final Model 建议 0.02")
    parser.add_argument("--pseudo-clamp", type=float, default=None,
                        help="pseudo 像素 clamp 上限；Final Model 建议 2.0")
    parser.add_argument("--residual-scale", type=float, default=None,
                        help="残差缩放 pseudo=IR+scale*delta；Final Model 建议 0.05")
    parser.add_argument("--ema-decay", type=float, default=None,
                        help="EMA 衰减系数，如 0.999；0 表示关闭")
    parser.add_argument("--m3fd-ann-file", type=str, default=None,
                        help="M3FD COCO 标注文件路径（--dataset m3fd 时使用）；"
                             "None 时使用 {data-root}/annotations/val.json")
    parser.add_argument("--stage-weights", type=str, default=None,
                        help="手动指定三阶段损失权重，格式 'a0,d0;a1,d1;a2,d2'，"
                             "如 '1.0,0.0;0.8,0.2;0.5,0.5'（先对齐再检测）")
    parser.add_argument("--align-layer-indices", type=str, default=None,
                        help="多层 L_align 所用 encoder 层索引（逗号分隔，0-based），"
                             "如 '1,3,5'；空字符串退化为仅对齐入口特征")
    parser.add_argument("--bbox-align-weight", type=float, default=None,
                        help="GT bbox 区域 patch 的对齐权重倍率（默认 3.0）；1.0=均匀权重")
    parser.add_argument("--adapter-mode", type=str, default="pixel",
                        choices=["pixel", "feature"],
                        help="适配器模式：pixel=CSMA 像素翻译；feature=FeatureAdapter 特征级")
    parser.add_argument("--fa-zero-init", action="store_true", default=None,
                        help="FeatureAdapter 最后一层零初始化（默认 feature 模式开启）")
    parser.add_argument("--no-fa-zero-init", action="store_true",
                        help="禁用 FeatureAdapter 零初始化")
    parser.add_argument(
        "--val-metric", type=str, default=None,
        choices=["map_50", "person_car_mean"],
        help="val 早停指标：map_50 或 person_car_mean（M3FD 两类推荐后者）",
    )
    parser.add_argument(
        "--allow-hard-stage", action="store_true",
        help="M3FD 允许进入 Hard 课程阶段（默认 M3FD 跳过 Hard，仅 Easy+Mixed）",
    )
    args = parser.parse_args()

    # Phase 5.1：构建 CSMAConfig（命令行可覆盖字段）
    overrides: Dict[str, Any] = {}
    if args.epochs      is not None: overrides["total_epochs"]      = args.epochs
    if args.batch_size  is not None: overrides["batch_size"]        = args.batch_size
    if args.lr          is not None: overrides["lr"]                = args.lr
    if args.gmm_batches is not None: overrides["gmm_max_batches"]   = args.gmm_batches
    if args.max_steps   is not None: overrides["max_steps_per_epoch"] = args.max_steps
    if args.hard_max_epochs is not None:
        overrides["hard_max_epochs"] = args.hard_max_epochs
    if args.vis_every is not None:
        overrides["vis_every"] = args.vis_every
    if args.text_prompt is not None:
        overrides["text_prompt"] = args.text_prompt
    if args.val_metric is not None:
        overrides["val_metric"] = args.val_metric
    if args.lambda_recon is not None:
        overrides["lambda_recon"] = args.lambda_recon
    if args.lambda_id is not None:
        overrides["lambda_id"] = args.lambda_id
    if args.lambda_tv is not None:
        overrides["lambda_tv"] = args.lambda_tv
    if args.lambda_logit_reg is not None:
        overrides["lambda_logit_reg"] = args.lambda_logit_reg
    if args.pseudo_clamp is not None:
        overrides["pseudo_clamp"] = args.pseudo_clamp
    if args.residual_scale is not None:
        overrides["residual_scale"] = args.residual_scale
    if args.ema_decay is not None:
        overrides["ema_decay"] = args.ema_decay
    if args.stage_weights:
        try:
            parsed = [
                tuple(float(v) for v in pair.split(","))
                for pair in args.stage_weights.split(";")
            ]
            if len(parsed) != 3 or any(len(p) != 2 for p in parsed):
                raise ValueError()
            overrides["stage_loss_weights"] = [tuple(p) for p in parsed]  # type: ignore[misc]
        except Exception:
            raise ValueError(
                "--stage-weights 格式错误，示例: '1.0,0.0;0.8,0.2;0.5,0.5'"
            )
    if args.stage_boundaries:
        parts = [int(x.strip()) for x in args.stage_boundaries.split(",")]
        if len(parts) != 2:
            raise ValueError("--stage-boundaries 格式应为 'easy_end,mixed_end'")
        overrides["stage_epoch_boundaries"] = parts
    if args.align_layer_indices is not None:
        if args.align_layer_indices.strip() == "":
            overrides["align_layer_indices"] = []
        else:
            overrides["align_layer_indices"] = [
                int(x.strip()) for x in args.align_layer_indices.split(",")
            ]
    if args.bbox_align_weight is not None:
        overrides["bbox_align_weight"] = args.bbox_align_weight
    overrides["adapter_mode"] = args.adapter_mode
    if args.adapter_mode == "feature":
        # 特征模式固定：无 L_recon、单层 token 对齐
        if args.lambda_recon is None:
            overrides["lambda_recon"] = 0.0
        if args.align_layer_indices is None:
            overrides["align_layer_indices"] = []
    if args.no_fa_zero_init:
        overrides["fa_zero_init"] = False
    elif args.fa_zero_init:
        overrides["fa_zero_init"] = True
    elif args.adapter_mode == "feature" and "fa_zero_init" not in overrides:
        overrides["fa_zero_init"] = True
    overrides["ir_data_root"]  = args.data_root
    overrides["rgb_data_root"] = args.rgb_data_root
    overrides["output_dir"]    = args.out_dir
    # flir_v1/m3fd：有 RGB 配对，默认 full（L_det + L_align，完整 GMM-CMSS）
    # flir_v2：无 RGB 配对，强制 det_only（除非用户显式指定）
    # legacy：使用用户指定值
    if args.dataset in ("flir_v1", "m3fd"):
        overrides["loss_mode"] = args.loss_mode if args.loss_mode else "full"
    elif args.dataset == "flir_v2":
        overrides["loss_mode"] = args.loss_mode if args.loss_mode else "det_only"
    elif args.loss_mode is not None:
        overrides["loss_mode"] = args.loss_mode
    if args.dataset == "m3fd":
        from src.dataset_m3fd import M3FD_DEFAULT_TEXT_PROMPT
        if "text_prompt" not in overrides:
            overrides["text_prompt"] = M3FD_DEFAULT_TEXT_PROMPT
        if "val_metric" not in overrides:
            overrides["val_metric"] = "person_car_mean"
        if not args.allow_hard_stage:
            overrides["skip_hard_stage"] = True
        if not args.stage_weights:
            overrides["stage_loss_weights"] = [(0.3, 0.7), (0.1, 0.9), (0.0, 1.0)]
    cfg = CSMAConfig.from_overrides(overrides)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_csma] 设备: {device}")

    ckpt_dir = os.path.join(cfg.output_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "vis"),  exist_ok=True)

    # Phase 5.2：处理器与文本编码
    processor = AutoProcessor.from_pretrained(cfg.model_id, local_files_only=True)
    # FLIR IR 原始分辨率 640×512；Grounding DINO processor 默认 shortest_edge=800 会放大图像，
    # 导致显存激增（800×1000 vs 640×512）。将 shortest_edge/longest_edge 限制在原始尺度内。
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
        ip = processor.image_processor
        # SizeDict 可通过属性或下标赋值
        try:
            cur_se = ip.size.shortest_edge or 0
        except AttributeError:
            cur_se = ip.size.get("shortest_edge", 0) or 0
        if cur_se > cfg.img_size:
            try:
                ip.size.shortest_edge = cfg.img_size
                ip.size.longest_edge  = cfg.img_size * 2   # 保持宽高比不被截断
            except AttributeError:
                ip.size = {"shortest_edge": cfg.img_size, "longest_edge": cfg.img_size * 2}
            print(f"[train_csma] processor image size 限制到 shortest_edge={cfg.img_size}, longest_edge={cfg.img_size*2}")
    tokenizer = processor.tokenizer
    encoded = tokenizer(cfg.text_prompt, return_tensors="pt")
    input_ids_base:       torch.Tensor = encoded["input_ids"].to(device)
    attention_mask_base:  torch.Tensor = encoded["attention_mask"].to(device)
    print(f"[train_csma] 文本 prompt 分词: {tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])}")

    # Phase 5.3：加载并冻结 DINO
    dino: GroundingDinoForObjectDetection = GroundingDinoForObjectDetection.from_pretrained(
        cfg.model_id, local_files_only=True
    ).to(device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False
    # GroundingDINO 暂不支持 gradient_checkpointing_enable()，主要依赖 AMP 节省显存

    # Phase 5.4：实例化适配器、优化器、调度器、课程调度器
    csma: Optional[CSMA] = None
    feature_adapter: Optional[FeatureAdapter] = None
    start_epoch = max(0, args.start_epoch)

    if cfg.adapter_mode == "feature":
        feature_adapter = FeatureAdapter(cfg).to(device)
        if args.init_ckpt:
            fa_state = _load_feature_adapter_state_dict(args.init_ckpt, device)
            feature_adapter.load_state_dict(fa_state)
            print(f"[train_csma] 已加载 FeatureAdapter: {args.init_ckpt}，从 epoch {start_epoch} 继续")
        elif start_epoch > 0:
            raise ValueError("--start-epoch>0 时必须提供 --init-ckpt")
        else:
            print(
                f"[train_csma] FeatureAdapter 随机初始化，参数量="
                f"{feature_adapter.count_parameters():,}  "
                f"fa_zero_init={cfg.fa_zero_init}"
            )
            if cfg.fa_zero_init:
                with torch.no_grad():
                    probe = torch.zeros(1, 8, cfg.proto_dim, device=device)
                    dnorm = feature_adapter.identity_delta_norm(probe)
                print(f"[train_csma] 零初始化检查 |MLP(x)|_mean={dnorm:.6e}（应≈0）")
        feature_adapter.train()
        trainable: nn.Module = feature_adapter
    else:
        csma = CSMA(cfg).to(device)
        if args.init_ckpt:
            state = _load_csma_state_dict(args.init_ckpt, device)
            csma.load_state_dict(state)
            print(f"[train_csma] 已加载 CSMA: {args.init_ckpt}，从 epoch {start_epoch} 继续")
        elif start_epoch > 0:
            raise ValueError("--start-epoch>0 时必须提供 --init-ckpt")
        csma.train()
        trainable = csma

    model_ema: Optional[ModelEMA] = None
    if cfg.ema_decay > 0.0:
        model_ema = ModelEMA(trainable, decay=cfg.ema_decay)
        print(f"[train_csma] EMA 已启用，decay={cfg.ema_decay}")

    optimizer = AdamW(trainable.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    warmup_epochs = max(0, args.warmup_epochs)
    cosine_epochs = max(1, cfg.total_epochs - warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    if warmup_epochs > 0:
        warmup_sched = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
        print(f"[train_csma] LR warmup: {warmup_epochs} epoch  {cfg.lr*0.1:.2e} → {cfg.lr:.2e}")
    else:
        lr_scheduler = cosine_sched

    cmss_sched = CMSSScheduler(cfg)
    b_easy, b_mixed = cmss_sched.stage_boundaries
    n_hard = cfg.total_epochs - b_mixed
    print(
        f"[train_csma] 课程: Easy epoch[0,{b_easy})  "
        f"Mixed[{b_easy},{b_mixed})  Hard[{b_mixed},{cfg.total_epochs})  "
        f"（Hard 共 {n_hard} epoch）"
    )

    signal.signal(signal.SIGUSR1, _request_emergency_save)

    # AMP GradScaler（fp16 下防梯度下溢）
    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    print(f"[train_csma] AMP={'开启 fp16' if use_amp else '关闭（fp32）'}")

    # Phase 5.5：数据集与 DataLoader
    if args.dataset == "flir_v1":
        cat_map, valid_ids = build_flir_v1_category_map(cfg.text_prompt)
        dataset = FlirV1PairedDataset(
            root=cfg.ir_data_root,
            processor=processor,
            text_prompt=cfg.text_prompt,
            category_map=cat_map,
            valid_cat_ids=valid_ids,
        )
        collate_fn_use = collate_flir_v1
        print(f"[train_csma] 数据集模式: flir_v1  loss_mode={cfg.loss_mode}")
    elif args.dataset == "flir_v2":
        cat_map, valid_ids = build_flir_v2_category_map(cfg.text_prompt)
        dataset = FlirADASV2Dataset(
            root=cfg.ir_data_root,
            processor=processor,
            text_prompt=cfg.text_prompt,
            category_map=cat_map,
            valid_cat_ids=valid_ids,
        )
        collate_fn_use = collate_flir_v2
        print(f"[train_csma] 数据集模式: flir_v2  loss_mode={cfg.loss_mode}")
    elif args.dataset == "m3fd":
        from src.dataset_m3fd import (
            M3FDPairedDataset,
            build_m3fd_category_map,
            build_m3fd_category_map_for_training,
            collate_m3fd,
        )
        # 训练用 0-based class_idx（DINO loss 需要）；val eval 用 1-based eval_cat_id（COCO eval 需要）
        cat_map, valid_ids = build_m3fd_category_map_for_training(cfg.text_prompt)
        dataset = M3FDPairedDataset(
            root=cfg.ir_data_root,
            processor=processor,
            text_prompt=cfg.text_prompt,
            ann_file=args.m3fd_ann_file,
            category_map=cat_map,
            valid_cat_ids=valid_ids,
            split="train",
            canonical_size=(1024, 768),   # 统一分辨率，消除 batch padding 坐标偏移
        )
        collate_fn_use = collate_m3fd
        print(f"[train_csma] 数据集模式: m3fd  adapter_mode={cfg.adapter_mode}  "
              f"loss_mode={cfg.loss_mode}  lambda_recon={cfg.lambda_recon}  "
              f"lambda_id={cfg.lambda_id}  lambda_tv={cfg.lambda_tv}  "
              f"logit_reg={cfg.lambda_logit_reg}  clamp={cfg.pseudo_clamp}  "
              f"res_scale={cfg.residual_scale}  "
              f"prompt={cfg.text_prompt!r}  val_metric={cfg.val_metric}  "
              f"skip_hard={cfg.skip_hard_stage}")
    else:
        cat_map = build_coco_category_to_class_index(cfg.text_prompt)
        dataset = FlirPairedDataset(
            ir_root=cfg.ir_data_root,
            rgb_root=cfg.rgb_data_root,
            processor=processor,
            text_prompt=cfg.text_prompt,
            coco_category_id_to_class_idx=cat_map,
        )
        collate_fn_use = collate_paired
        print(f"[train_csma] 数据集模式: legacy  loss_mode={cfg.loss_mode}")

    print(f"[train_csma] adapter_mode={cfg.adapter_mode}  loss_mode={cfg.loss_mode}")
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn_use,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    print(f"[train_csma] 数据集大小: {len(dataset)}，每 epoch {len(loader)} 个 batch")

    # Val 早停（stage1 末段按 mAP@0.5 选 best_stage1.pt）
    val_dataset = None
    val_valid_ids: Optional[frozenset] = None
    val_jsonl = os.path.join(cfg.output_dir, "logs", "val_early_stop.jsonl")
    best_stage1_path = os.path.join(ckpt_dir, "best_stage1.pt")
    best_stage1_meta_path = os.path.join(ckpt_dir, "best_stage1_meta.json")
    best_val_map = -1.0
    best_val_epoch = -1
    best_ap_person = -1.0
    best_ap_car = -1.0
    epochs_without_class_improve = 0
    val_metric_name = cfg.val_metric
    use_val_early_stop = args.val_early_stop
    use_class_ap_stop = (
        use_val_early_stop
        and args.val_stop_patience > 0
        and args.dataset == "m3fd"
    )
    if use_val_early_stop:
        if args.dataset not in ("flir_v1", "m3fd"):
            print(f"[train_csma] 警告: --val-early-stop 仅支持 flir_v1/m3fd，已忽略（当前={args.dataset}）")
            use_val_early_stop = False
        elif args.dataset == "flir_v1":
            val_root = args.val_data_root or _default_val_data_root(cfg.ir_data_root)
            if not os.path.isdir(val_root):
                raise FileNotFoundError(f"val 目录不存在: {val_root}")
            val_dataset = FlirV1PairedDataset(
                root=val_root,
                processor=processor,
                text_prompt=cfg.text_prompt,
                category_map=cat_map,
                valid_cat_ids=valid_ids,
            )
            val_valid_ids = valid_ids
            print(
                f"[train_csma] Val 早停: epoch [{args.val_start},{args.val_end}] "
                f"every={args.val_every}  val={val_root}  → {best_stage1_path}"
            )
            if args.stop_after_stage1:
                print("[train_csma] 将在 Mixed 末轮后停止训练（跳过 Hard）")
        else:  # m3fd
            from src.dataset_m3fd import (
                M3FDPairedDataset,
                build_m3fd_category_map,
            )
            # val 用 1-based eval_cat_id，供 _build_gt_coco + COCO eval
            val_cat_map, val_valid_ids_m3fd = build_m3fd_category_map(cfg.text_prompt)
            val_dataset = M3FDPairedDataset(
                root=cfg.ir_data_root,
                processor=processor,
                text_prompt=cfg.text_prompt,
                ann_file=args.m3fd_ann_file,
                category_map=val_cat_map,
                valid_cat_ids=val_valid_ids_m3fd,
                split="val",
                canonical_size=(1024, 768),
            )
            val_valid_ids = val_valid_ids_m3fd
            if not args.val_every_epoch:
                print(
                    f"[train_csma] Val 早停 (m3fd): epoch [{args.val_start},{args.val_end}] "
                    f"every={args.val_every}  val_split=20%  → {best_stage1_path}"
                )
            if args.stop_after_stage1:
                print("[train_csma] 将在 Mixed 末轮后停止训练（跳过 Hard）")

    val_start_eff = args.val_start
    val_end_eff = args.val_end
    if use_val_early_stop and args.val_every_epoch:
        val_start_eff = start_epoch
        val_end_eff = cfg.total_epochs - 1
        print(
            f"[train_csma] Val 每 epoch 评测: epoch [{val_start_eff},{val_end_eff}]  "
            f"metric={val_metric_name}  → {best_stage1_path}"
        )
        if use_class_ap_stop:
            print(
                f"[train_csma] 单项 AP 早停: 连续 {args.val_stop_patience} 次 val "
                f"person/car 均未创新高则停止"
                + (
                    f"（epoch≥{args.val_stop_min_epochs} 后生效）"
                    if args.val_stop_min_epochs > 0
                    else ""
                )
            )
    elif use_val_early_stop and not args.val_manual:
        val_end_eff = cmss_sched.stage1_last_epoch()
        val_start_eff = max(start_epoch, val_end_eff - 8)
        print(
            f"[train_csma] Val 窗口自动对齐 Mixed 末段: epoch [{val_start_eff},{val_end_eff}]"
        )

    # Phase 5.6：SwanLab 初始化
    swan_run = _build_swanlab_logger(
        enable=args.use_swanlab,
        project=args.swanlab_project,
        run_name=args.swanlab_run_name,
        config=cfg.to_dict(),
    )

    # ── 训练状态 ────────────────────────────────────────────────────────────
    loss_history:  List[float] = []
    global_step = 0
    grad_checked = False

    # ══════════════════════════════════════════════════════════════════════════
    # 训练循环
    # ══════════════════════════════════════════════════════════════════════════
    stop_training = False
    for epoch in range(start_epoch, cfg.total_epochs):
        if stop_training:
            break

        # Phase 5.7：GMM 定期更新
        if cmss_sched.should_update_gmm(epoch) and cfg.loss_mode != "det_only":
            print(f"[train_csma] Epoch {epoch}: 重新拟合 GMM...")
            if cfg.adapter_mode == "feature":
                assert feature_adapter is not None
                cmss_vals = collect_cmss_values_feature(
                    dino, feature_adapter, loader, device,
                    max_batches=cfg.gmm_max_batches,
                )
            else:
                assert csma is not None
                cmss_vals = collect_cmss_values(
                    dino, csma, loader, device, input_ids_base, attention_mask_base,
                    max_batches=cfg.gmm_max_batches,
                )
            if len(cmss_vals) >= cfg.gmm_n_components:
                cmss_sched.update_gmm(cmss_vals)
                mu1, mu2, mu3 = cmss_sched.sorted_means
                print(f"[train_csma] GMM 均值更新: μ₁={mu1:.3f}  μ₂={mu2:.3f}  μ₃={mu3:.3f}")
            # GMM 采样后释放显存碎片，避免训练前向 OOM
            torch.cuda.empty_cache()

        stage = cmss_sched.get_stage(epoch)
        lambda_align, lambda_det = cmss_sched.get_loss_weights(epoch)
        eff_la, eff_ld = _effective_loss_weights(cfg, lambda_align, lambda_det)
        mu1, mu2, mu3 = cmss_sched.sorted_means

        # ── Epoch 统计累积 ───────────────────────────────────────────────────
        ep_loss_total:  List[float] = []
        ep_loss_det:    List[float] = []
        ep_loss_align:  List[float] = []
        ep_loss_recon:  List[float] = []
        ep_loss_id:     List[float] = []
        ep_loss_tv:     List[float] = []
        ep_loss_logit:  List[float] = []

        max_steps = cfg.max_steps_per_epoch
        ep_step = 0
        for batch in loader:
            if max_steps != -1 and ep_step >= max_steps:
                break
            ir_pv  = batch["pixel_values"].to(device)
            pm     = batch["pixel_mask"].to(device)
            labels = _move_labels_to_device(batch["labels"], device)
            bsz    = ir_pv.shape[0]

            input_ids = input_ids_base.expand(bsz, -1)
            attn_mask = attention_mask_base.expand(bsz, -1)

            # Phase 5.8 + 5.9 + 5.10：AMP 混合精度前向
            with _maybe_autocast(use_amp):
                l_align = torch.tensor(0.0, device=device)
                l_recon = torch.tensor(0.0, device=device)
                l_id = torch.tensor(0.0, device=device)
                l_tv = torch.tensor(0.0, device=device)
                l_logit = torch.tensor(0.0, device=device)

                if cfg.adapter_mode == "feature":
                    assert feature_adapter is not None
                    adapted_cache: Dict[str, torch.Tensor] = {}
                    outputs = forward_dino_with_feature_adapter(
                        dino,
                        feature_adapter,
                        ir_pv,
                        pm,
                        input_ids,
                        attn_mask,
                        labels,
                        adapted_cache,
                    )
                    if outputs.loss is None:
                        raise RuntimeError("outputs.loss 为 None")
                    l_det, det_scalars = _build_det_loss(outputs, cfg)
                    if cfg.lambda_logit_reg > 0.0:
                        l_logit = logit_regularization(outputs, device)

                    if cfg.loss_mode != "det_only" and "rgb_pixel_values" in batch:
                        rgb_pv = batch["rgb_pixel_values"].to(device)
                        with torch.no_grad():
                            bundle_rgb = extract_vision_tokens(dino, rgb_pv, pm)
                        feat_adapt = adapted_cache.get("feat")
                        if feat_adapt is None:
                            bundle_ir = extract_vision_tokens(dino, ir_pv, pm)
                            feat_adapt = feature_adapter(bundle_ir.vision_features)

                        cmss_map = compute_cmss(
                            bundle_rgb.vision_features.detach(),
                            feat_adapt.detach(),
                        )
                        mask = build_cmss_mask(
                            cmss_map, stage, mu1, mu2, mu3,
                            cfg.mask_ratio, cmss_sched.gmm,
                        )
                        feat_len = feat_adapt.shape[1]
                        ir_hw = (ir_pv.shape[2], ir_pv.shape[3])
                        p_weight = build_bbox_patch_weight(
                            batch["labels"], feat_len, ir_hw, cfg.bbox_align_weight,
                        ).to(device)
                        l_align = compute_align_loss_weighted(
                            feat_adapt,
                            bundle_rgb.vision_features,
                            mask,
                            p_weight,
                        )
                        del rgb_pv, bundle_rgb
                else:
                    assert csma is not None
                    # Phase 5.8：CSMA 前向 → 伪 RGB
                    pseudo_rgb = csma(ir_pv)

                    # Phase 5.9：L_det（冻结 DINO 前向）
                    _multilayer_ir: Dict[int, torch.Tensor] = {}
                    _entry_feat_ir: Dict[str, torch.Tensor] = {}

                    def _capture_entry_hook(module: nn.Module, inp: tuple, out: Any) -> None:
                        if inp and isinstance(inp[0], torch.Tensor):
                            _entry_feat_ir["feat"] = inp[0]
                        elif hasattr(out, "last_hidden_state_vision"):
                            _entry_feat_ir["feat"] = out.last_hidden_state_vision
                        elif isinstance(out, (tuple, list)):
                            for o in out:
                                if isinstance(o, torch.Tensor):
                                    _entry_feat_ir["feat"] = o
                                    break

                    def _make_layer_hook(li: int):
                        def _hook(module: nn.Module, inp: tuple, out: Any) -> None:
                            if isinstance(out, (tuple, list)):
                                for o in out:
                                    if isinstance(o, torch.Tensor) and o.dim() == 3:
                                        _multilayer_ir[li] = o
                                        return
                            elif isinstance(out, torch.Tensor) and out.dim() == 3:
                                _multilayer_ir[li] = out
                        return _hook

                    enc = dino.model.encoder
                    _handles: List[Any] = [enc.register_forward_hook(_capture_entry_hook)]
                    _align_layers = cfg.align_layer_indices
                    for _li in _align_layers:
                        if 0 <= _li < len(enc.layers):
                            _handles.append(
                                enc.layers[_li].register_forward_hook(_make_layer_hook(_li))
                            )

                    try:
                        outputs = dino(
                            pixel_values=pseudo_rgb,
                            pixel_mask=pm,
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                            labels=labels,
                        )
                    finally:
                        for _h in _handles:
                            _h.remove()

                    if outputs.loss is None:
                        raise RuntimeError("outputs.loss 为 None")
                    l_det, det_scalars = _build_det_loss(outputs, cfg)
                    if cfg.lambda_logit_reg > 0.0:
                        l_logit = logit_regularization(outputs, device)

                    if "rgb_pixel_values" in batch:
                        rgb_pv = batch["rgb_pixel_values"].to(device)
                        if cfg.lambda_recon > 0.0:
                            l_recon = F.mse_loss(pseudo_rgb, rgb_pv.detach())

                        if cfg.lambda_id > 0.0:
                            l_id = identity_loss(pseudo_rgb, ir_pv)
                        if cfg.lambda_tv > 0.0:
                            l_tv = total_variation_loss(pseudo_rgb)

                        _entry_ir = _entry_feat_ir.get("feat")
                        if _entry_ir is not None:
                            valid_layers = [li for li in _align_layers if li in _multilayer_ir]
                            with torch.no_grad():
                                feat_rgb_entry, _multilayer_rgb = extract_dino_rgb_features_all(
                                    dino, rgb_pv, input_ids_base, attention_mask_base, valid_layers
                                )
                            cmss_map = compute_cmss(feat_rgb_entry.detach(), _entry_ir.detach())
                            mask = build_cmss_mask(
                                cmss_map, stage, mu1, mu2, mu3,
                                cfg.mask_ratio, cmss_sched.gmm,
                            )
                            feat_len = _entry_ir.shape[1]
                            ir_hw = (ir_pv.shape[2], ir_pv.shape[3])
                            p_weight = build_bbox_patch_weight(
                                batch["labels"], feat_len, ir_hw, cfg.bbox_align_weight,
                            ).to(device)
                            layer_losses: List[torch.Tensor] = []
                            for i, li in enumerate(valid_layers):
                                f_ir = _multilayer_ir[li]
                                f_rgb = _multilayer_rgb[i] if i < len(_multilayer_rgb) else None
                                if f_rgb is None or f_ir.shape != f_rgb.shape:
                                    continue
                                layer_losses.append(
                                    compute_align_loss_weighted(f_ir, f_rgb.detach(), mask, p_weight)
                                )
                            if layer_losses:
                                l_align = torch.stack(layer_losses).mean()
                            else:
                                l_align = compute_align_loss_weighted(
                                    _entry_ir, feat_rgb_entry.detach(), mask, p_weight
                                )
                            del feat_rgb_entry
                        del rgb_pv
                    else:
                        if cfg.lambda_id > 0.0:
                            l_id = identity_loss(pseudo_rgb, ir_pv)
                        if cfg.lambda_tv > 0.0:
                            l_tv = total_variation_loss(pseudo_rgb)

                loss = (
                    eff_la * l_align
                    + eff_ld * l_det
                    + cfg.lambda_recon * l_recon
                    + cfg.lambda_id * l_id
                    + cfg.lambda_tv * l_tv
                    + cfg.lambda_logit_reg * l_logit
                )

            # Phase 5.11：AMP 反传 + 梯度裁剪
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Phase 5.12：梯度检查（第一个有效 step）— 在 backward 后、step 前执行
            if not grad_checked:
                adapter_grad = sum(
                    p.grad.abs().sum().item()
                    for p in trainable.parameters() if p.grad is not None
                )
                dino_grads = [p.grad for p in dino.parameters()]
                if math.isnan(adapter_grad) or math.isinf(adapter_grad):
                    print(f"[梯度检查] 跳过（梯度={adapter_grad:.6g}，等待 scaler 稳定）")
                else:
                    assert adapter_grad > 0.0, "适配器梯度为 0，计算图可能断裂"
                    assert all(g is None for g in dino_grads), "冻结的 DINO 不应有梯度"
                    tag = "FeatureAdapter" if cfg.adapter_mode == "feature" else "CSMA"
                    print(f"[梯度检查] {tag} grad L1 = {adapter_grad:.6f}；DINO grad 均为 None: OK")
                    grad_checked = True

            nn.utils.clip_grad_norm_(trainable.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if model_ema is not None:
                model_ema.update(trainable)

            # ── 统计累积 ─────────────────────────────────────────────────────
            loss_val   = float(loss.detach().cpu())
            align_val  = float(l_align.detach().cpu())
            recon_val  = float(l_recon.detach().cpu())
            id_val     = float(l_id.detach().cpu())
            tv_val     = float(l_tv.detach().cpu())
            logit_val  = float(l_logit.detach().cpu())
            ep_loss_total.append(loss_val)
            ep_loss_det.append(det_scalars["loss_det"])
            ep_loss_align.append(align_val)
            ep_loss_recon.append(recon_val)
            ep_loss_id.append(id_val)
            ep_loss_tv.append(tv_val)
            ep_loss_logit.append(logit_val)
            global_step += 1
            ep_step     += 1

            # 每 10 步打印一次进度
            if ep_step % 10 == 0 or ep_step == 1:
                print(
                    f"  Ep{epoch} Step{ep_step:5d}  "
                    f"loss={loss_val:.4f}  det={det_scalars['loss_det']:.4f}  "
                    f"align={align_val:.4f}  recon={recon_val:.4f}  "
                    f"id={id_val:.4f}  tv={tv_val:.4f}  logit={logit_val:.4f}  "
                    f"scale={scaler.get_scale():.0f}"
                )

            if swan_run is not None:
                swan_run.log({
                    "train/loss_step":        loss_val,
                    "train/loss_det_step":    det_scalars["loss_det"],
                    "train/loss_align_step":  align_val,
                    "train/loss_recon_step":  recon_val,
                    "train/loss_ce":          det_scalars["loss_ce"],
                    "train/loss_bbox":        det_scalars["loss_bbox"],
                    "train/loss_giou":        det_scalars["loss_giou"],
                    "train/global_step":      global_step,
                    "train/stage":            stage,
                })

        # ── Epoch 结束：日志 ─────────────────────────────────────────────────
        lr_scheduler.step()

        mean_total = float(np.mean(ep_loss_total))
        mean_det   = float(np.mean(ep_loss_det))
        mean_align = float(np.mean(ep_loss_align))
        mean_recon = float(np.mean(ep_loss_recon)) if ep_loss_recon else 0.0
        mean_id    = float(np.mean(ep_loss_id)) if ep_loss_id else 0.0
        mean_tv    = float(np.mean(ep_loss_tv)) if ep_loss_tv else 0.0
        mean_logit = float(np.mean(ep_loss_logit)) if ep_loss_logit else 0.0
        loss_history.append(mean_total)

        print(
            f"Epoch {epoch + 1:4d}/{cfg.total_epochs}  "
            f"stage={stage}  loss={mean_total:.5f}  "
            f"(det={mean_det:.4f}  align={mean_align:.4f}  recon={mean_recon:.4f}  "
            f"id={mean_id:.4f}  tv={mean_tv:.4f}  logit={mean_logit:.4f})  "
            f"λ_align={eff_la}  λ_det={eff_ld}  λ_recon={cfg.lambda_recon}  "
            f"λ_id={cfg.lambda_id}  λ_tv={cfg.lambda_tv}  "
            f"loss_mode={cfg.loss_mode}"
        )

        if swan_run is not None:
            swan_run.log({
                "train/loss_epoch":        mean_total,
                "train/loss_det_epoch":    mean_det,
                "train/loss_align_epoch":  mean_align,
                "train/loss_recon_epoch":  mean_recon,
                "train/lambda_align":      lambda_align,
                "train/lambda_det":        lambda_det,
                "train/lambda_recon":      cfg.lambda_recon,
                "train/stage":             stage,
                "train/epoch":             epoch + 1,
            })

        # 每 epoch 写入 latest（便于暂停评测 / 断点续训）
        latest_path = os.path.join(ckpt_dir, "latest.pt")
        _save_adapter_weights(latest_path, trainable)
        _write_latest_meta(ckpt_dir, epoch)
        if model_ema is not None:
            ema_path = os.path.join(ckpt_dir, f"ema_epoch_{epoch:04d}.pt")
            _save_ema_weights(ema_path, model_ema)
            print(f"  [ckpt] EMA 快照: {ema_path}")
        if _emergency_save_requested:
            emerg_path = os.path.join(ckpt_dir, f"emergency_epoch_{epoch:04d}.pt")
            _save_adapter_weights(emerg_path, trainable)
            print(f"  [ckpt] 紧急快照: {emerg_path}")
            _emergency_save_requested = False

        # Phase 5.13：定期保存权重 + 可视化
        if cfg.adapter_mode == "pixel" and (
            epoch % cfg.vis_every == 0 or epoch == cfg.total_epochs - 1
        ):
            assert csma is not None
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
            _save_adapter_weights(ckpt_path, trainable)

            vis_path = os.path.join(cfg.output_dir, "vis", f"epoch_{epoch:04d}.png")
            samples = [dataset[i] for i in range(min(3, len(dataset)))]
            try:
                save_multi_sample_grid(
                    csma, dino, processor, samples,
                    cfg.text_prompt, device, vis_path,
                    box_threshold=0.3,
                    text_threshold=0.25,
                )
                print(f"  [vis] 已保存: {vis_path}")
            except Exception as exc:
                print(f"  [vis] 可视化失败（不影响训练）: {exc}")
        elif cfg.adapter_mode == "feature" and (
            epoch % cfg.vis_every == 0 or epoch == cfg.total_epochs - 1
        ):
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
            _save_adapter_weights(ckpt_path, trainable)

        # Val 早停：stage1 末段（默认 epoch 25–33）按 mAP@0.5 更新 best_stage1.pt
        if (
            use_val_early_stop
            and val_dataset is not None
            and val_valid_ids is not None
            and val_start_eff <= epoch <= val_end_eff
            and (epoch - val_start_eff) % max(1, args.val_every) == 0
        ):
            print(f"[train_csma] Val 评测 epoch {epoch} ...")
            metrics = _run_val_map(
                dino=dino,
                processor=processor,
                val_dataset=val_dataset,
                valid_cat_ids=val_valid_ids,
                device=device,
                dataset_mode=args.dataset,
                text_prompt=cfg.text_prompt,
                batch_size=args.val_batch_size,
                box_threshold=args.val_box_threshold,
                text_threshold=args.val_text_threshold,
                adapter_mode=cfg.adapter_mode,
                csma=csma,
                feature_adapter=feature_adapter,
            )
            metrics["epoch"] = epoch
            metrics["stage"] = stage
            _append_jsonl(val_jsonl, metrics)
            # 打印所有 ap_* 指标（person/car/bus/... 均自动包含）
            ap_strs = "  ".join(
                f"{k.replace('ap_', '')}={v:.4f}"
                for k, v in metrics.items()
                if k.startswith("ap_") and isinstance(v, float)
            )
            score = _val_metric_score(metrics, val_metric_name)
            pcm = metrics.get("person_car_mean")
            pcm_str = f"  person_car_mean={pcm:.4f}" if pcm is not None else ""
            print(
                f"  [val] epoch={epoch}  {val_metric_name}={score:.4f}  "
                f"mAP@0.5={metrics['map_50']:.4f}{pcm_str}  {ap_strs}"
            )
            if score > best_val_map:
                best_val_map = score
                best_val_epoch = epoch
                _save_adapter_weights(best_stage1_path, trainable)
                snap = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
                _save_adapter_weights(snap, trainable)
                meta_dict: Dict[str, Any] = {
                    "best_epoch": epoch,
                    "val_metric": val_metric_name,
                    "val_score": best_val_map,
                    "map_50": metrics["map_50"],
                    "map_50_95": metrics["map_50_95"],
                    "val_data_root": (
                        args.val_data_root
                        or (cfg.ir_data_root if args.dataset == "m3fd"
                            else _default_val_data_root(cfg.ir_data_root))
                    ),
                }
                # 写入所有 per-class AP（兼容 flir_v1 两类和 m3fd 六类）
                meta_dict.update(
                    {k: v for k, v in metrics.items()
                     if (k.startswith("ap_") or k == "person_car_mean")
                     and isinstance(v, float)}
                )
                with open(best_stage1_meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta_dict, f, indent=2)
                print(f"  [val] ★ 新最佳 → {best_stage1_path}  (epoch {epoch})")

            if use_class_ap_stop:
                imp_p, imp_c, best_ap_person, best_ap_car = _update_class_ap_bests(
                    metrics, best_ap_person, best_ap_car,
                )
                if imp_p or imp_c:
                    epochs_without_class_improve = 0
                    parts: List[str] = []
                    if imp_p:
                        parts.append(f"person={metrics['ap_person']:.4f}")
                    if imp_c:
                        parts.append(f"car={metrics['ap_car']:.4f}")
                    print(f"  [val] 单项 AP 新高: {', '.join(parts)}")
                else:
                    epochs_without_class_improve += 1
                    print(
                        f"  [val] person/car 均未创新高 "
                        f"({epochs_without_class_improve}/{args.val_stop_patience})  "
                        f"best person={best_ap_person:.4f}  car={best_ap_car:.4f}"
                    )
                if (
                    epoch >= args.val_stop_min_epochs
                    and epochs_without_class_improve >= args.val_stop_patience
                ):
                    print(
                        f"[train_csma] 早停: 连续 {args.val_stop_patience} 次 val "
                        f"person/car 单项 AP 无提升  "
                        f"best person={best_ap_person:.4f}  car={best_ap_car:.4f}  "
                        f"(epoch {epoch})"
                    )
                    stop_training = True

            if swan_run is not None:
                swan_run.log({
                    "val/map_50": metrics["map_50"],
                    "val/map_50_95": metrics["map_50_95"],
                    "val/best_map_50": best_val_map,
                    "val/best_epoch": best_val_epoch,
                    "val/epoch": epoch + 1,
                })

        if args.stop_after_stage1 and epoch >= cmss_sched.stage1_last_epoch():
            msg = (
                f"[train_csma] 已完成 Mixed 末轮 epoch {cmss_sched.stage1_last_epoch()}，"
                f"stop-after-stage1：跳过 Hard（共 {n_hard} epoch 未训）"
            )
            if best_val_epoch >= 0:
                msg += (
                    f"  最佳 val epoch={best_val_epoch} "
                    f"{val_metric_name}={best_val_map:.4f}"
                )
            print(msg)
            stop_training = True

    # ── 训练结束 ─────────────────────────────────────────────────────────────
    final_name = "fa_last.pt" if cfg.adapter_mode == "feature" else "csma_last.pt"
    final_ckpt = os.path.join(cfg.output_dir, "ckpt", final_name)
    _save_adapter_weights(final_ckpt, trainable)
    print(f"[train_csma] 训练完成，最终权重: {final_ckpt}")
    if best_val_epoch >= 0:
        print(
            f"[train_csma] Stage1 最佳: epoch {best_val_epoch}  "
            f"{val_metric_name}={best_val_map:.4f}  → {best_stage1_path}"
        )

    # 绘制 loss 曲线
    loss_png = os.path.join(cfg.output_dir, "logs", "loss.png")
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label="total loss / epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("CSMA Training — Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_png, dpi=120)
    plt.close()
    print(f"[train_csma] Loss 曲线: {loss_png}")

    if loss_history:
        drop = (loss_history[0] - loss_history[-1]) / max(loss_history[0], 1e-8)
        print(f"[train_csma] Loss 初值≈{loss_history[0]:.4f}  末值≈{loss_history[-1]:.4f}  相对下降≈{drop*100:.1f}%")
        if swan_run is not None:
            swan_run.log({"train/loss_drop_ratio": float(drop)})

    if swan_run is not None:
        try:
            swan_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
