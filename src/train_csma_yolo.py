"""
CSMA + 冻结 YOLOv8-m 训练主程序。

结构与 ``train_csma.py`` 一致，将 Grounding DINO 替换为 YOLOv8-m：
  - L_det：伪 RGB → v8DetectionLoss
  - L_align：YOLO neck P5 特征 + GMM-CMSS
  - Val mAP：``eval_yolo_csma`` 协议

用法：
    source /root/miniconda3/etc/profile.d/conda.sh
    bash scripts/01_train_yolo.sh
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from src.cmss_utils import CMSSScheduler, build_cmss_mask, compute_cmss
from src.config import CSMAConfig
from src.csma import CSMA
from src.ir_augment import IRAugment
from src.dataset import build_coco_category_to_class_index
from src.dataset_flir_v1 import FlirV1PairedDataset, build_flir_v1_category_map, collate_flir_v1
from src.dataset_flir_v2 import FlirADASV2Dataset, build_flir_v2_category_map, collate_flir_v2
from src.dataset_paired import FlirPairedDataset, collate_paired
from src.infer_vis import denormalize_pixel_values
from src.train_csma import (
    _append_jsonl,
    _build_swanlab_logger,
    _default_val_data_root,
    _load_csma_state_dict,
    _maybe_autocast,
    _move_labels_to_device,
    _write_latest_meta,
    compute_align_loss,
)
from src.yolo_utils import (
    YoloAlignProjector,
    build_yolo_align_projector,
    collect_cmss_values_yolo,
    denormalize_for_yolo,
    extract_yolo_backbone_features,
    labels_to_yolo_targets,
    load_frozen_yolo,
    run_val_map_yolo,
    yolo_det_forward_with_feats,
)

_emergency_save_requested = False
_LOG = "[train_csma_yolo]"


def _save_csma_weights(
    path: str,
    csma: nn.Module,
    projector: Optional[YoloAlignProjector] = None,
) -> None:
    """保存 CSMA（+ 可选 align_projector）权重。格式与 eval_yolo_csma 的 {'csma':...} 兼容。"""
    payload: Dict[str, Any] = {"csma": csma.state_dict()}
    if projector is not None:
        payload["align_projector"] = projector.state_dict()
    torch.save(payload, path)


def _load_projector_state_dict(
    ckpt_path: str,
    map_location: Any,
) -> Optional[Dict[str, Any]]:
    """从 ckpt 读 align_projector 权重；旧格式不含此字段时返回 None。"""
    raw = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    if isinstance(raw, dict) and "align_projector" in raw:
        return raw["align_projector"]
    return None


def _request_emergency_save(signum: int, frame: Any) -> None:
    global _emergency_save_requested
    _emergency_save_requested = True
    print(f"{_LOG} 收到信号 {signum}，本 epoch 结束后将写入 latest.pt / emergency.pt")


def _processor_mean_std(processor: Any) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    ip = processor.image_processor
    mean = tuple(ip.image_mean)
    std = tuple(ip.image_std)
    return mean, std


def _save_pseudo_rgb_grid(
    csma: CSMA,
    samples: List[Dict[str, Any]],
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    device: torch.device,
    out_path: str,
) -> None:
    """IR | 伪 RGB | GT RGB 三联图（无检测框）。"""
    csma.eval()
    n = min(3, len(samples))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = np.array([axes])
    with torch.no_grad():
        for row, sample in enumerate(samples[:n]):
            ir = sample["pixel_values"].unsqueeze(0).to(device)
            pseudo = csma(ir)[0]
            ir_img = denormalize_pixel_values(sample["pixel_values"], mean, std)
            pseudo_img = denormalize_pixel_values(pseudo.cpu(), mean, std)
            axes[row, 0].imshow(ir_img)
            axes[row, 0].set_title("IR")
            axes[row, 0].axis("off")
            axes[row, 1].imshow(pseudo_img)
            axes[row, 1].set_title("Pseudo RGB")
            axes[row, 1].axis("off")
            rgb_pv = sample.get("rgb_pixel_values")
            if rgb_pv is not None:
                rgb_img = denormalize_pixel_values(rgb_pv, mean, std)
                axes[row, 2].imshow(rgb_img)
            axes[row, 2].set_title("GT RGB")
            axes[row, 2].axis("off")
    csma.train()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSMA 训练：冻结 YOLOv8-m + 可训练 CSMA + GMM-CMSS L_align"
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="/root/autodl-tmp/yolov8m.pt",
        help="YOLOv8-m 权重路径",
    )
    parser.add_argument("--dataset", type=str, default="flir_v1",
                        choices=["legacy", "flir_v1", "flir_v2"])
    parser.add_argument("--data-root", type=str, default="FLIR_License/train")
    parser.add_argument("--rgb-data-root", type=str, default="train/rgb")
    parser.add_argument("--out-dir", type=str, default="outputs_csma_yolo")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss-mode", type=str, default=None,
                        choices=["full", "det_only", "align_only"])
    parser.add_argument("--gmm-batches", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", type=str, default="csma-yolo-training")
    parser.add_argument("--swanlab-run-name", type=str, default="csma-yolo-run")
    parser.add_argument("--init-ckpt", type=str, default=None)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--val-early-stop", action="store_true")
    parser.add_argument("--val-data-root", type=str, default=None)
    parser.add_argument("--val-start", type=int, default=25)
    parser.add_argument("--val-end", type=int, default=33)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--val-conf", type=float, default=0.05)
    parser.add_argument("--stop-after-stage1", action="store_true")
    parser.add_argument("--hard-max-epochs", type=int, default=None)
    parser.add_argument("--stage-boundaries", type=str, default=None)
    parser.add_argument("--val-manual", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--stage-weights", type=str, default=None)
    parser.add_argument(
        "--ir-aug",
        action="store_true",
        help="训练时对 IR 图做随机域增强（亮度/对比度/gamma/噪声/直方图均衡），改善跨域泛化",
    )
    parser.add_argument("--ir-aug-prob", type=float, default=0.8,
                        help="IR 增强触发概率（默认 0.8）")
    args = parser.parse_args()

    if not os.path.isfile(args.yolo_weights):
        raise FileNotFoundError(f"YOLO 权重不存在: {args.yolo_weights}")

    overrides: Dict[str, Any] = {}
    if args.epochs is not None:
        overrides["total_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.gmm_batches is not None:
        overrides["gmm_max_batches"] = args.gmm_batches
    if args.max_steps is not None:
        overrides["max_steps_per_epoch"] = args.max_steps
    if args.hard_max_epochs is not None:
        overrides["hard_max_epochs"] = args.hard_max_epochs
    if args.stage_weights:
        try:
            parsed = [
                tuple(float(v) for v in pair.split(","))
                for pair in args.stage_weights.split(";")
            ]
            if len(parsed) != 3 or any(len(p) != 2 for p in parsed):
                raise ValueError()
            overrides["stage_loss_weights"] = [tuple(p) for p in parsed]  # type: ignore[misc]
        except Exception as exc:
            raise ValueError(
                "--stage-weights 格式错误，示例: '1.0,0.0;0.8,0.2;0.5,0.5'"
            ) from exc
    if args.stage_boundaries:
        parts = [int(x.strip()) for x in args.stage_boundaries.split(",")]
        if len(parts) != 2:
            raise ValueError("--stage-boundaries 格式应为 'easy_end,mixed_end'")
        overrides["stage_epoch_boundaries"] = parts
    overrides["ir_data_root"] = args.data_root
    overrides["rgb_data_root"] = args.rgb_data_root
    overrides["output_dir"] = args.out_dir
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
    print(f"{_LOG} 设备: {device}")
    print(f"{_LOG} YOLO: {args.yolo_weights}")

    ckpt_dir = os.path.join(cfg.output_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "vis"), exist_ok=True)

    processor = AutoProcessor.from_pretrained(cfg.model_id)
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
        ip = processor.image_processor
        try:
            cur_se = ip.size.shortest_edge or 0
        except AttributeError:
            cur_se = ip.size.get("shortest_edge", 0) or 0
        if cur_se > cfg.img_size:
            try:
                ip.size.shortest_edge = cfg.img_size
                ip.size.longest_edge = cfg.img_size * 2
            except AttributeError:
                ip.size = {"shortest_edge": cfg.img_size, "longest_edge": cfg.img_size * 2}
            print(f"{_LOG} processor shortest_edge={cfg.img_size}")

    img_mean, img_std = _processor_mean_std(processor)

    yolo_wrapper, yolo_model = load_frozen_yolo(args.yolo_weights, device)
    print(f"{_LOG} YOLO 已加载并冻结（nc={yolo_model.model[-1].nc}）")

    # YoloAlignProjector：多尺度 P3+P4+P5 → proto_dim，修复维度/尺度不匹配
    projector = build_yolo_align_projector(
        yolo_model, proto_dim=cfg.proto_dim, device=device, img_size=cfg.img_size
    )

    csma = CSMA(cfg).to(device)
    start_epoch = max(0, args.start_epoch)
    if args.init_ckpt:
        state = _load_csma_state_dict(args.init_ckpt, device)
        csma.load_state_dict(state)
        proj_state = _load_projector_state_dict(args.init_ckpt, device)
        if proj_state is not None:
            projector.load_state_dict(proj_state)
            print(f"{_LOG} 已加载 projector 权重: {args.init_ckpt}")
        print(f"{_LOG} 已加载 CSMA 权重: {args.init_ckpt}，从 epoch {start_epoch} 继续")
    elif start_epoch > 0:
        raise ValueError("--start-epoch>0 时必须提供 --init-ckpt")
    csma.train()
    projector.train()

    # CSMA + projector 一起优化；YOLO 全程冻结
    optimizer = AdamW(
        list(csma.parameters()) + list(projector.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    warmup_epochs = max(0, args.warmup_epochs)
    cosine_epochs = max(1, cfg.total_epochs - warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    if warmup_epochs > 0:
        warmup_sched = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        lr_scheduler: Any = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
        print(f"{_LOG} LR warmup: {warmup_epochs} epoch")
    else:
        lr_scheduler = cosine_sched

    cmss_sched = CMSSScheduler(cfg)
    b_easy, b_mixed = cmss_sched.stage_boundaries
    n_hard = cfg.total_epochs - b_mixed
    print(
        f"{_LOG} 课程: Easy[0,{b_easy})  Mixed[{b_easy},{b_mixed})  "
        f"Hard[{b_mixed},{cfg.total_epochs})"
    )

    signal.signal(signal.SIGUSR1, _request_emergency_save)
    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    print(f"{_LOG} AMP={'fp16' if use_amp else 'fp32'}")

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
    print(f"{_LOG} 数据集: {args.dataset}  loss_mode={cfg.loss_mode}  n={len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn_use,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_dataset = None
    val_valid_ids: Optional[frozenset] = None
    val_jsonl = os.path.join(cfg.output_dir, "logs", "val_early_stop.jsonl")
    best_stage1_path = os.path.join(ckpt_dir, "best_stage1.pt")
    best_stage1_meta_path = os.path.join(ckpt_dir, "best_stage1_meta.json")
    best_val_map = -1.0
    best_val_epoch = -1
    use_val_early_stop = args.val_early_stop
    if use_val_early_stop:
        if args.dataset != "flir_v1":
            print(f"{_LOG} 警告: val 早停仅支持 flir_v1，已忽略")
            use_val_early_stop = False
        else:
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
            print(f"{_LOG} Val 早停 → {best_stage1_path}  val={val_root}")

    val_start_eff = args.val_start
    val_end_eff = args.val_end
    if use_val_early_stop and not args.val_manual:
        val_end_eff = cmss_sched.stage1_last_epoch()
        val_start_eff = max(start_epoch, val_end_eff - 8)
        print(f"{_LOG} Val 窗口: epoch [{val_start_eff},{val_end_eff}]")

    swan_run = _build_swanlab_logger(
        enable=args.use_swanlab,
        project=args.swanlab_project,
        run_name=args.swanlab_run_name,
        config={**cfg.to_dict(), "yolo_weights": args.yolo_weights},
    )

    loss_history: List[float] = []
    global_step = 0
    grad_checked = False
    stop_training = False

    for epoch in range(start_epoch, cfg.total_epochs):
        if stop_training:
            break

        if cmss_sched.should_update_gmm(epoch):
            print(f"{_LOG} Epoch {epoch}: 重新拟合 GMM...")
            cmss_vals = collect_cmss_values_yolo(
                yolo_model, csma, loader, device, img_mean, img_std,
                max_batches=cfg.gmm_max_batches,
                projector=projector,
            )
            if len(cmss_vals) >= cfg.gmm_n_components:
                cmss_sched.update_gmm(cmss_vals)
                mu1, mu2, mu3 = cmss_sched.sorted_means
                print(f"{_LOG} GMM: μ₁={mu1:.3f} μ₂={mu2:.3f} μ₃={mu3:.3f}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

        stage = cmss_sched.get_stage(epoch)
        lambda_align, lambda_det = cmss_sched.get_loss_weights(epoch)
        lambda_det = max(lambda_det, 0.05)
        mu1, mu2, mu3 = cmss_sched.sorted_means

        ep_loss_total: List[float] = []
        ep_loss_det: List[float] = []
        ep_loss_align: List[float] = []
        max_steps = cfg.max_steps_per_epoch
        ep_step = 0

        for batch in loader:
            if max_steps != -1 and ep_step >= max_steps:
                break
            ir_pv = batch["pixel_values"].to(device)
            labels = _move_labels_to_device(batch["labels"], device)

            with _maybe_autocast(use_amp):
                pseudo_rgb = csma(ir_pv)
                yolo_img = denormalize_for_yolo(pseudo_rgb, img_mean, img_std)
                yolo_targets = labels_to_yolo_targets(labels, device)

                l_det, det_scalars, feat_ir = yolo_det_forward_with_feats(
                    yolo_model, yolo_img, yolo_targets, cfg, projector=projector,
                )

                l_align = torch.tensor(0.0, device=device)
                if cfg.loss_mode != "det_only" and "rgb_pixel_values" in batch:
                    rgb_pv = batch["rgb_pixel_values"].to(device)
                    yolo_rgb = denormalize_for_yolo(rgb_pv, img_mean, img_std)
                    with torch.no_grad():
                        feat_rgb = extract_yolo_backbone_features(yolo_model, yolo_rgb, projector)
                    cmss_map = compute_cmss(feat_rgb.detach(), feat_ir.detach())
                    mask = build_cmss_mask(
                        cmss_map, stage, mu1, mu2, mu3, cfg.mask_ratio, cmss_sched.gmm,
                    )
                    l_align = compute_align_loss(feat_ir, feat_rgb, mask)

                # ── Pseudo-RGB 正则（id + tv；YOLO 无 logit_reg）───────────
                l_id = torch.tensor(0.0, device=device)
                l_tv = torch.tensor(0.0, device=device)

                if cfg.id_loss_weight > 0:
                    l_id = F.l1_loss(pseudo_rgb, ir_pv) * cfg.id_loss_weight

                if cfg.tv_loss_weight > 0:
                    diff_h = pseudo_rgb[:, :, 1:, :] - pseudo_rgb[:, :, :-1, :]
                    diff_w = pseudo_rgb[:, :, :, 1:] - pseudo_rgb[:, :, :, :-1]
                    l_tv = (diff_h.abs().mean() + diff_w.abs().mean()) * cfg.tv_loss_weight

                l_pseudo_reg = l_id + l_tv

                if cfg.loss_mode == "align_only":
                    loss = lambda_align * l_align + l_pseudo_reg
                elif cfg.loss_mode == "det_only":
                    loss = lambda_det * l_det + l_pseudo_reg
                else:
                    loss = lambda_align * l_align + lambda_det * l_det + l_pseudo_reg

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if not grad_checked:
                csma_grad = sum(
                    p.grad.abs().sum().item()
                    for p in csma.parameters() if p.grad is not None
                )
                proj_grad = sum(
                    p.grad.abs().sum().item()
                    for p in projector.parameters() if p.grad is not None
                )
                yolo_grads = [p.grad for p in yolo_model.parameters()]
                if math.isnan(csma_grad) or math.isinf(csma_grad):
                    print(f"[梯度检查] 跳过（csma_grad={csma_grad:.6g}）")
                else:
                    assert csma_grad > 0.0, "CSMA 梯度为 0"
                    assert all(g is None for g in yolo_grads), "冻结 YOLO 不应有梯度"
                    print(
                        f"[梯度检查] CSMA grad L1={csma_grad:.6f}  "
                        f"projector grad L1={proj_grad:.6f}  YOLO grad=None: OK"
                    )
                    grad_checked = True

            trainable_params = list(csma.parameters()) + list(projector.parameters())
            nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_val  = float(loss.detach().cpu())
            align_val = float(l_align.detach().cpu())
            id_val    = float(l_id.detach().cpu())
            tv_val    = float(l_tv.detach().cpu())
            ep_loss_total.append(loss_val)
            ep_loss_det.append(det_scalars["loss_det"])
            ep_loss_align.append(align_val)
            global_step += 1
            ep_step += 1

            if ep_step % 10 == 0 or ep_step == 1:
                print(
                    f"  Ep{epoch} Step{ep_step:5d}  loss={loss_val:.4f}  "
                    f"det={det_scalars['loss_det']:.4f}  align={align_val:.5f}  "
                    f"id={id_val:.4f}  tv={tv_val:.4f}  scale={scaler.get_scale():.0f}"
                )

            if swan_run is not None:
                swan_run.log({
                    "train/loss_step":       loss_val,
                    "train/loss_det_step":   det_scalars["loss_det"],
                    "train/loss_align_step": align_val,
                    "train/loss_id_step":    id_val,
                    "train/loss_tv_step":    tv_val,
                    "train/global_step":     global_step,
                    "train/stage":           stage,
                })

        lr_scheduler.step()
        mean_total = float(np.mean(ep_loss_total)) if ep_loss_total else 0.0
        mean_det = float(np.mean(ep_loss_det)) if ep_loss_det else 0.0
        mean_align = float(np.mean(ep_loss_align)) if ep_loss_align else 0.0
        loss_history.append(mean_total)
        print(
            f"Epoch {epoch + 1:4d}/{cfg.total_epochs}  stage={stage}  "
            f"loss={mean_total:.5f}  det={mean_det:.4f}  align={mean_align:.5f}  "
            f"λ_align={lambda_align}  λ_det={lambda_det}"
        )

        latest_path = os.path.join(ckpt_dir, "latest.pt")
        _save_csma_weights(latest_path, csma, projector)
        _write_latest_meta(ckpt_dir, epoch)
        if _emergency_save_requested:
            emerg_path = os.path.join(ckpt_dir, f"emergency_epoch_{epoch:04d}.pt")
            _save_csma_weights(emerg_path, csma, projector)
            globals()["_emergency_save_requested"] = False

        # 每个 epoch 都存一份快照（MD 建议：别只留 best，保留每轮供事后挑选）
        epoch_ckpt = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
        _save_csma_weights(epoch_ckpt, csma, projector)

        if epoch % cfg.vis_every == 0 or epoch == cfg.total_epochs - 1:
            vis_path = os.path.join(cfg.output_dir, "vis", f"epoch_{epoch:04d}.png")
            try:
                samples = [dataset[i] for i in range(min(3, len(dataset)))]
                _save_pseudo_rgb_grid(csma, samples, img_mean, img_std, device, vis_path)
                print(f"  [vis] {vis_path}")
            except Exception as exc:
                print(f"  [vis] 失败: {exc}")

        if (
            use_val_early_stop
            and val_dataset is not None
            and val_valid_ids is not None
            and val_start_eff <= epoch <= val_end_eff
            and (epoch - val_start_eff) % max(1, args.val_every) == 0
        ):
            print(f"{_LOG} Val epoch {epoch} ...")
            metrics = run_val_map_yolo(
                csma, yolo_wrapper, processor, val_dataset, val_valid_ids,
                device, args.dataset, args.val_batch_size, args.val_conf,
            )
            metrics["epoch"] = epoch
            metrics["stage"] = stage
            metrics["yolo_weights"] = args.yolo_weights
            _append_jsonl(val_jsonl, metrics)
            print(
                f"  [val] mAP@0.5={metrics['map_50']:.4f}  "
                f"person={metrics['ap_person']:.4f}  car={metrics['ap_car']:.4f}"
            )
            if metrics["map_50"] > best_val_map:
                best_val_map = metrics["map_50"]
                best_val_epoch = epoch
                _save_csma_weights(best_stage1_path, csma, projector)
                with open(best_stage1_meta_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "best_epoch": epoch,
                            "map_50": best_val_map,
                            "map_50_95": metrics["map_50_95"],
                            "ap_person": metrics["ap_person"],
                            "ap_car": metrics["ap_car"],
                            "yolo_weights": args.yolo_weights,
                            "val_data_root": args.val_data_root or _default_val_data_root(cfg.ir_data_root),
                        },
                        f,
                        indent=2,
                    )
                print(f"  [val] ★ best → {best_stage1_path}")

        if args.stop_after_stage1 and epoch >= cmss_sched.stage1_last_epoch():
            print(f"{_LOG} stop-after-stage1，跳过 Hard")
            stop_training = True

    final_ckpt = os.path.join(cfg.output_dir, "ckpt", "csma_last.pt")
    _save_csma_weights(final_ckpt, csma, projector)
    print(f"{_LOG} 完成: {final_ckpt}")
    if best_val_epoch >= 0:
        print(f"{_LOG} 最佳 epoch {best_val_epoch}  mAP@0.5={best_val_map:.4f}")

    loss_png = os.path.join(cfg.output_dir, "logs", "loss.png")
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label="total loss / epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("CSMA + YOLOv8 Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_png, dpi=120)
    plt.close()

    if swan_run is not None:
        try:
            swan_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
