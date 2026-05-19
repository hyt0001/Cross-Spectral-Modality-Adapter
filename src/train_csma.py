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
import math
import os
from typing import Any, Dict, List, Optional

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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from src.cmss_utils import CMSSScheduler, build_cmss_mask, compute_cmss
from src.config import CSMAConfig
from src.csma import CSMA
from src.dataset import build_coco_category_to_class_index
from src.dataset_flir_v1 import FlirV1PairedDataset, build_flir_v1_category_map, collate_flir_v1
from src.dataset_flir_v2 import FlirADASV2Dataset, build_flir_v2_category_map, collate_flir_v2
from src.dataset_paired import FlirPairedDataset, collate_paired
from src.infer_vis import save_multi_sample_grid


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
    仅在未被掩蔽的 Patch（mask=0）上计算 MSE 对齐损失。

    L_align = mean( || F_ir[Ω] - sg(F_rgb[Ω]) ||² )
    其中 Ω = {i | mask[i] == 0}（保留的 Patch 集合）

    Args:
        feat_ir:  [B, L, D]  有梯度，经 CSMA → DINO 路径。
        feat_rgb: [B, L, D]  须已 detach（stop_gradient）。
        mask:     [B, L]     0=保留，1=掩蔽；由 build_cmss_mask 生成。

    Returns:
        标量 MSE 损失；若全部 Patch 被掩蔽则返回 0。
    """
    assert feat_ir.shape == feat_rgb.shape, "feat_ir 与 feat_rgb 形状不一致"
    unmasked = (mask == 0)                              # [B, L] bool
    if not unmasked.any():
        return torch.tensor(0.0, device=feat_ir.device, requires_grad=True)
    return F.mse_loss(feat_ir[unmasked], feat_rgb[unmasked].detach())


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


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5: 主训练入口
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CSMA 训练：冻结 DINO + 可训练 CSMA + GMM-CMSS L_align")
    parser.add_argument("--dataset",       type=str, default="flir_v1",
                        choices=["legacy", "flir_v1", "flir_v2"],
                        help="数据集类型：flir_v1=FLIR_License 配对数据集（默认）；"
                             "flir_v2=FLIR_ADAS_v2 无配对；legacy=旧版 train/ 目录")
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
    args = parser.parse_args()

    # Phase 5.1：构建 CSMAConfig（命令行可覆盖字段）
    overrides: Dict[str, Any] = {}
    if args.epochs      is not None: overrides["total_epochs"]      = args.epochs
    if args.batch_size  is not None: overrides["batch_size"]        = args.batch_size
    if args.lr          is not None: overrides["lr"]                = args.lr
    if args.gmm_batches is not None: overrides["gmm_max_batches"]   = args.gmm_batches
    if args.max_steps   is not None: overrides["max_steps_per_epoch"] = args.max_steps
    overrides["ir_data_root"]  = args.data_root
    overrides["rgb_data_root"] = args.rgb_data_root
    overrides["output_dir"]    = args.out_dir
    # flir_v1：有 RGB 配对，默认 full（L_det + L_align，完整 GMM-CMSS）
    # flir_v2：无 RGB 配对，强制 det_only（除非用户显式指定）
    # legacy：使用用户指定值
    if args.dataset == "flir_v1":
        overrides["loss_mode"] = args.loss_mode if args.loss_mode else "full"
    elif args.dataset == "flir_v2":
        overrides["loss_mode"] = args.loss_mode if args.loss_mode else "det_only"
    elif args.loss_mode is not None:
        overrides["loss_mode"] = args.loss_mode
    cfg = CSMAConfig.from_overrides(overrides)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_csma] 设备: {device}")

    os.makedirs(os.path.join(cfg.output_dir, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "vis"),  exist_ok=True)

    # Phase 5.2：处理器与文本编码
    processor = AutoProcessor.from_pretrained(cfg.model_id)
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
        cfg.model_id
    ).to(device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False
    # GroundingDINO 暂不支持 gradient_checkpointing_enable()，主要依赖 AMP 节省显存

    # Phase 5.4：实例化 CSMA、优化器、调度器、课程调度器
    csma = CSMA(cfg).to(device)
    csma.train()
    optimizer = AdamW(csma.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.total_epochs)
    cmss_sched = CMSSScheduler(cfg)

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

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn_use,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    print(f"[train_csma] 数据集大小: {len(dataset)}，每 epoch {len(loader)} 个 batch")

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
    for epoch in range(cfg.total_epochs):

        # Phase 5.7：GMM 定期更新
        if cmss_sched.should_update_gmm(epoch):
            print(f"[train_csma] Epoch {epoch}: 重新拟合 GMM...")
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
        mu1, mu2, mu3 = cmss_sched.sorted_means

        # ── Epoch 统计累积 ───────────────────────────────────────────────────
        ep_loss_total:  List[float] = []
        ep_loss_det:    List[float] = []
        ep_loss_align:  List[float] = []

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
                # Phase 5.8：CSMA 前向 → 伪 RGB
                pseudo_rgb = csma(ir_pv)

                # Phase 5.9：L_det（冻结 DINO 前向）
                # 同步通过 hook 捕获 encoder 特征，供 L_align 复用（省去第三次 DINO 前向）
                _feat_ir_cache: Dict[str, torch.Tensor] = {}

                def _capture_hook(module: nn.Module, inp: tuple, out: Any) -> None:
                    if inp and isinstance(inp[0], torch.Tensor):
                        _feat_ir_cache["feat"] = inp[0]
                    elif hasattr(out, "last_hidden_state_vision"):
                        _feat_ir_cache["feat"] = out.last_hidden_state_vision
                    elif hasattr(out, "last_hidden_state"):
                        _feat_ir_cache["feat"] = out.last_hidden_state
                    elif isinstance(out, (tuple, list)):
                        for o in out:
                            if isinstance(o, torch.Tensor):
                                _feat_ir_cache["feat"] = o
                                break

                _hook_handle = dino.model.encoder.register_forward_hook(_capture_hook)
                try:
                    outputs = dino(
                        pixel_values=pseudo_rgb,
                        pixel_mask=pm,
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        labels=labels,
                    )
                finally:
                    _hook_handle.remove()

                if outputs.loss is None:
                    raise RuntimeError("outputs.loss 为 None，请确认 transformers>=4.40 且 labels 非空。")
                l_det, det_scalars = _build_det_loss(outputs, cfg)

                # Phase 5.10：L_align（需 RGB 配对）
                # feat_ir 复用 L_det 前向捕获的特征；仅额外做一次 RGB no_grad 前向
                l_align = torch.tensor(0.0, device=device)
                if "rgb_pixel_values" in batch and "feat" in _feat_ir_cache:
                    rgb_pv = batch["rgb_pixel_values"].to(device)

                    with torch.no_grad():
                        feat_rgb = extract_dino_backbone_features(
                            dino, rgb_pv, input_ids_base, attention_mask_base
                        )
                    feat_ir = _feat_ir_cache["feat"]

                    cmss_map = compute_cmss(feat_rgb.detach(), feat_ir.detach())
                    mask = build_cmss_mask(
                        cmss_map, stage,
                        mu1, mu2, mu3,
                        cfg.mask_ratio,
                        cmss_sched.gmm,
                    )
                    l_align = compute_align_loss(feat_ir, feat_rgb, mask)
                    del rgb_pv, feat_rgb

                loss = lambda_align * l_align + lambda_det * l_det

            # Phase 5.11：AMP 反传 + 梯度裁剪
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Phase 5.12：梯度检查（第一个有效 step）— 在 backward 后、step 前执行
            if not grad_checked:
                csma_grad = sum(
                    p.grad.abs().sum().item()
                    for p in csma.parameters() if p.grad is not None
                )
                dino_grads = [p.grad for p in dino.parameters()]
                # NaN 来自 fp16 初始 overflow（scaler 会自动调整 scale）；
                # 等到梯度合法（有限 & 非零）再置 checked
                if math.isnan(csma_grad) or math.isinf(csma_grad):
                    print(f"[梯度检查] 跳过（梯度={csma_grad:.6g}，等待 scaler 稳定）")
                else:
                    assert csma_grad > 0.0, "CSMA 梯度为 0，计算图可能断裂"
                    assert all(g is None for g in dino_grads), "冻结的 DINO 不应有梯度"
                    print(f"[梯度检查] CSMA grad L1 = {csma_grad:.6f}；DINO 参数 grad 均为 None: OK")
                    grad_checked = True

            nn.utils.clip_grad_norm_(csma.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # ── 统计累积 ─────────────────────────────────────────────────────
            loss_val   = float(loss.detach().cpu())
            align_val  = float(l_align.detach().cpu())
            ep_loss_total.append(loss_val)
            ep_loss_det.append(det_scalars["loss_det"])
            ep_loss_align.append(align_val)
            global_step += 1
            ep_step     += 1

            # 每 10 步打印一次进度
            if ep_step % 10 == 0 or ep_step == 1:
                print(
                    f"  Ep{epoch} Step{ep_step:5d}  "
                    f"loss={loss_val:.4f}  det={det_scalars['loss_det']:.4f}  "
                    f"align={align_val:.5f}  scale={scaler.get_scale():.0f}"
                )

            if swan_run is not None:
                swan_run.log({
                    "train/loss_step":       loss_val,
                    "train/loss_det_step":   det_scalars["loss_det"],
                    "train/loss_align_step": align_val,
                    "train/loss_ce":         det_scalars["loss_ce"],
                    "train/loss_bbox":       det_scalars["loss_bbox"],
                    "train/loss_giou":       det_scalars["loss_giou"],
                    "train/global_step":     global_step,
                    "train/stage":           stage,
                })

        # ── Epoch 结束：日志 ─────────────────────────────────────────────────
        lr_scheduler.step()

        mean_total = float(np.mean(ep_loss_total))
        mean_det   = float(np.mean(ep_loss_det))
        mean_align = float(np.mean(ep_loss_align))
        loss_history.append(mean_total)

        print(
            f"Epoch {epoch + 1:4d}/{cfg.total_epochs}  "
            f"stage={stage}  loss={mean_total:.5f}  "
            f"(det={mean_det:.4f}  align={mean_align:.5f})  "
            f"λ_align={lambda_align}  λ_det={lambda_det}"
        )

        if swan_run is not None:
            swan_run.log({
                "train/loss_epoch":       mean_total,
                "train/loss_det_epoch":   mean_det,
                "train/loss_align_epoch": mean_align,
                "train/lambda_align":     lambda_align,
                "train/lambda_det":       lambda_det,
                "train/stage":            stage,
                "train/epoch":            epoch + 1,
            })

        # Phase 5.13：定期保存权重 + 可视化
        if epoch % cfg.vis_every == 0 or epoch == cfg.total_epochs - 1:
            ckpt_path = os.path.join(cfg.output_dir, "ckpt", f"epoch_{epoch:04d}.pt")
            torch.save(csma.state_dict(), ckpt_path)

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

    # ── 训练结束 ─────────────────────────────────────────────────────────────
    final_ckpt = os.path.join(cfg.output_dir, "ckpt", "csma_last.pt")
    torch.save(csma.state_dict(), final_ckpt)
    print(f"[train_csma] 训练完成，最终权重: {final_ckpt}")

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
