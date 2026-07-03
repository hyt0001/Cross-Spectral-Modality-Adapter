"""OWLv2-CSMA training (teammate-style defaults: 40 epochs, 3-stage λ schedule)."""
from __future__ import annotations

import argparse
import io
import json
import math
import os

# DataLoader workers × OpenBLAS threads can exceed OpenBLAS's 128-region limit and segfault.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import signal
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from src.cmss_utils import CMSSScheduler, build_cmss_mask, compute_cmss
from src.config import CSMAConfig
from src.csma import CSMA
from src.dataset_flir_v1 import FlirV1PairedDataset, build_flir_v1_category_map, collate_flir_v1

_emergency_save = False
FLIR_TO_EVAL_CAT = {1: 1, 3: 2}
EVAL_CATEGORIES = [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}]
LABEL_TO_EVAL_CAT = {"person": 1, "car": 2}


class ModelEMA:
    """Exponential Moving Average of model weights.

    Call ``update(model)`` after each successful optimizer step.
    Call ``copy_to(model)`` before validation, ``restore(model)`` after.
    Set ``decay=0`` to disable (shadow tracks live weights exactly).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone().float()
        for name, buf in model.named_buffers():
            self.shadow["buf:" + name] = buf.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, param in model.named_parameters():
            self.shadow[name].mul_(d).add_(param.data.float(), alpha=1.0 - d)
        for name, buf in model.named_buffers():
            self.shadow["buf:" + name].copy_(buf.data)

    def copy_to(self, model: nn.Module) -> None:
        """Swap EMA weights into model; saves originals for restore()."""
        self._backup = {}
        for name, param in model.named_parameters():
            self._backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name].to(param.data.dtype))
        for name, buf in model.named_buffers():
            self._backup["buf:" + name] = buf.data.clone()
            buf.data.copy_(self.shadow["buf:" + name].to(buf.data.dtype))

    def restore(self, model: nn.Module) -> None:
        """Restore training weights after evaluation."""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        for name, buf in model.named_buffers():
            key = "buf:" + name
            if key in self._backup:
                buf.data.copy_(self._backup[key])
        self._backup = {}


def _sigusr1_handler(signum: int, frame: Any) -> None:
    global _emergency_save
    _emergency_save = True


@contextmanager
def _autocast(enabled: bool):
    if enabled:
        with torch.amp.autocast("cuda"):
            yield
    else:
        yield


def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dim=-1)


def _generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    lt_enc = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_enc = (rb_enc - lt_enc).clamp(min=0)
    area_enc = wh_enc[:, :, 0] * wh_enc[:, :, 1]
    return iou - (area_enc - union) / area_enc.clamp(min=1e-6)


def _sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2.0, query_weight=None):
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    per_query = loss.mean(-1)  # [N]
    if query_weight is not None:
        per_query = per_query * query_weight
    return per_query.sum() / max(num_boxes, 1)


def _sigmoid_cost_for_matching(logits: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    """Scenic/DETR-style focal classification cost for Hungarian matching."""
    p = logits.sigmoid()
    eps = 1e-8
    neg = -((1 - alpha) * (p ** gamma) * torch.log((1 - p).clamp(min=eps)))
    pos = -(alpha * ((1 - p) ** gamma) * torch.log(p.clamp(min=eps)))
    return pos - neg


def compute_det_loss(pred_boxes, logits, gt_boxes_list, gt_labels_list, cfg: CSMAConfig):
    device = logits.device
    target_cls = torch.zeros_like(logits)
    matched_pred, matched_gt, matched_labels = [], [], []
    num_gt = sum(len(g) for g in gt_boxes_list)
    for b in range(logits.shape[0]):
        gt_b, gt_l = gt_boxes_list[b].to(device), gt_labels_list[b].to(device)
        if len(gt_b) == 0:
            continue
        pb, lb = pred_boxes[b], logits[b]
        with torch.no_grad():
            class_cost = _sigmoid_cost_for_matching(lb, cfg.det_focal_alpha, cfg.det_focal_gamma)
            cost = class_cost[:, gt_l] + cfg.det_w_l1 * torch.cdist(pb, gt_b, p=1)
            cost += cfg.det_w_giou * (-_generalized_box_iou(
                _box_cxcywh_to_xyxy(pb.clamp(0, 1)), _box_cxcywh_to_xyxy(gt_b.clamp(0, 1))))
            pi, gi = linear_sum_assignment(cost.cpu().float().numpy())
        pi, gi = torch.from_numpy(pi).long(), torch.from_numpy(gi).long()
        target_cls[b, pi, gt_l[gi]] = 1.0
        matched_pred.append(pb[pi])
        matched_gt.append(gt_b[gi])
        matched_labels.append(gt_l[gi])
    plw, clw = cfg.person_loss_weight, cfg.car_loss_weight
    if plw != 1.0 or clw != 1.0:
        qw = torch.ones(logits.shape[0] * logits.shape[1], device=device)
        qw[(target_cls[:, :, 0] > 0.5).reshape(-1)] = plw
        qw[(target_cls[:, :, 1] > 0.5).reshape(-1)] = clw
    else:
        qw = None
    focal = _sigmoid_focal_loss(logits.reshape(-1, logits.shape[-1]), target_cls.reshape(-1, logits.shape[-1]),
                                num_gt, cfg.det_focal_alpha, cfg.det_focal_gamma, query_weight=qw)
    if matched_pred:
        all_pred, all_gt = torch.cat(matched_pred), torch.cat(matched_gt)
        all_labels = torch.cat(matched_labels)
        match_w = torch.where(all_labels == 1, clw, plw).to(device)
        l1_per = F.l1_loss(all_pred, all_gt, reduction="none").sum(-1)
        giou = _generalized_box_iou(_box_cxcywh_to_xyxy(all_pred.clamp(0, 1)), _box_cxcywh_to_xyxy(all_gt.clamp(0, 1)))
        giou_per = 1 - torch.diag(giou)
        wsum = match_w.sum().clamp(min=1)
        l1 = (l1_per * match_w).sum() / wsum
        giou_loss = (giou_per * match_w).sum() / wsum
    else:
        l1 = giou_loss = torch.tensor(0.0, device=device)
    loss = focal + cfg.det_w_l1 * l1 + cfg.det_w_giou * giou_loss
    return loss, {"loss_det": float(loss.detach()), "loss_focal": float(focal.detach()),
                  "loss_l1": float(l1.detach()), "loss_giou": float(giou_loss.detach())}


def compute_align_loss(feat_ir, feat_rgb, mask):
    keep = mask == 0
    if not keep.any():
        return torch.tensor(0.0, device=feat_ir.device, requires_grad=True)
    return F.mse_loss(feat_ir[keep], feat_rgb[keep].detach())


def _total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def collect_cmss_values(owlv2, csma, loader, device, input_ids, attn_mask, max_batches=100):
    csma.eval()
    vals = []
    n = 0
    with torch.no_grad():
        for batch in loader:
            if max_batches != -1 and n >= max_batches:
                break
            if "rgb_pixel_values" not in batch:
                continue
            B = batch["pixel_values"].shape[0]
            rgb_idx = batch.get("rgb_indices", list(range(B)))
            n_rgb = len(rgb_idx)
            ids_full = input_ids.repeat(B, 1)
            atm_full = attn_mask.repeat(B, 1)
            ids_sub = input_ids.repeat(n_rgb, 1)
            atm_sub = attn_mask.repeat(n_rgb, 1)
            pseudo = csma(batch["pixel_values"].to(device))
            rgb = batch["rgb_pixel_values"].to(device)
            f_ir = owlv2(input_ids=ids_full, pixel_values=pseudo, attention_mask=atm_full).image_embeds.flatten(1, 2)[rgb_idx]
            f_rgb = owlv2(input_ids=ids_sub, pixel_values=rgb, attention_mask=atm_sub).image_embeds.flatten(1, 2)
            vals.append(compute_cmss(f_rgb, f_ir).cpu().numpy().astype(np.float32).ravel())
            n += 1
    csma.train()
    return np.concatenate(vals) if vals else np.array([0.5], dtype=np.float32)


def _build_gt_coco(dataset, valid_ids):
    images, anns, aid = [], [], 1
    for img in dataset._images:
        iid = int(img["id"])
        images.append({"id": iid, "width": int(img["width"]), "height": int(img["height"]), "file_name": img["file_name"]})
        for ann in dataset._id_to_anns.get(iid, []):
            cid = int(ann["category_id"])
            if cid not in valid_ids:
                continue
            x, y, w, h = [float(v) for v in ann["bbox"]]
            anns.append({"id": aid, "image_id": iid, "category_id": FLIR_TO_EVAL_CAT[cid],
                         "bbox": [x, y, w, h], "area": float(ann["area"]), "iscrowd": int(ann.get("iscrowd", 0))})
            aid += 1
    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns, "categories": EVAL_CATEGORIES}
    coco.createIndex()
    return coco


def run_validation(csma, owlv2, processor, val_dataset, valid_ids, device, text_labels, batch_size, threshold):
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_flir_v1, num_workers=2)
    with torch.no_grad():
        enc = processor(text=[text_labels], images=None, return_tensors="pt", padding=True)
    ids = enc["input_ids"].to(device)
    atm = enc["attention_mask"].to(device)
    path_to_id = {os.path.join(val_dataset._root, img["file_name"]): img["id"] for img in val_dataset._images}
    preds = []
    csma.eval()
    with torch.no_grad():
        for batch in loader:
            B = batch["pixel_values"].shape[0]
            pseudo = csma(batch["pixel_values"].to(device))
            out = owlv2(input_ids=ids.repeat(B, 1), pixel_values=pseudo, attention_mask=atm.repeat(B, 1))
            results = processor.post_process_grounded_object_detection(
                out, target_sizes=torch.tensor(batch["orig_sizes"], device=device),
                threshold=threshold, text_labels=[text_labels] * B,
            )
            for res, p in zip(results, batch["image_paths"]):
                iid = path_to_id.get(p, 0)
                for box, score, label in zip(res["boxes"].cpu(), res["scores"].cpu(), res["text_labels"]):
                    cid = LABEL_TO_EVAL_CAT.get(label.strip().lower())
                    if cid is None:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    preds.append({"image_id": iid, "category_id": cid, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    csma.train()
    coco_gt = _build_gt_coco(val_dataset, valid_ids)
    if not preds:
        return {k: 0.0 for k in ["AP", "AP50", "AP75", "APS", "APM", "APL", "AR1", "AR10", "AR100", "ARS", "ARM", "ARL", "AP_person", "AP_car", "n_preds", "n_gt"]}
    coco_dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate(); ev.accumulate()
    old = sys.stdout; sys.stdout = io.StringIO(); ev.summarize(); sys.stdout = old
    s = ev.stats
    m = {"AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]), "APS": float(s[3]), "APM": float(s[4]), "APL": float(s[5]),
         "AR1": float(s[6]), "AR10": float(s[7]), "AR100": float(s[8]), "ARS": float(s[9]), "ARM": float(s[10]), "ARL": float(s[11]),
         "n_preds": float(len(preds)), "n_gt": float(len(coco_gt.anns))}
    for cid, name in [(1, "person"), (2, "car")]:
        ev2 = COCOeval(coco_gt, coco_dt, "bbox")
        ev2.params.catIds = [cid]; ev2.params.iouThrs = np.array([0.5])
        old = sys.stdout; sys.stdout = io.StringIO(); ev2.evaluate(); ev2.accumulate(); ev2.summarize(); sys.stdout = old
        m[f"AP_{name}"] = max(float(ev2.stats[0]), 0.0)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/root/autodl-tmp/train")
    parser.add_argument("--val-root", default="/root/autodl-tmp/val")
    parser.add_argument("--out-dir", default="outputs_teammate")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers; use 0 if OpenBLAS segfaults")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--init-ckpt", type=str, default=None)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val-threshold", type=float, default=0.2)
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--pseudo-clamp", type=float, default=None)
    parser.add_argument("--id-loss-weight", type=float, default=None)
    parser.add_argument("--tv-loss-weight", type=float, default=None)
    parser.add_argument("--logit-reg-weight", type=float, default=None)
    parser.add_argument("--logit-cap", type=float, default=None)
    parser.add_argument("--car-loss-weight", type=float, default=None,
                        help="det loss weight for car-matched queries (default 1.0; try 2.0)")
    parser.add_argument("--person-loss-weight", type=float, default=None,
                        help="det loss weight for person-matched queries (default 1.0; try 0.5)")
    parser.add_argument("--grad-skip-threshold", type=float, default=None)
    parser.add_argument("--disable-group-norm", action="store_true")
    parser.add_argument("--force-amp", action="store_true")
    parser.add_argument("--stage-boundaries", type=str, default=None)
    parser.add_argument("--stage-weights", type=str, default=None,
                        help="三阶段 λ 权重, 格式 'a0,d0;a1,d1;a2,d2', 例如 '1.0,0.1;0.8,0.2;0.5,0.5'")
    parser.add_argument("--gmm-update-every", type=int, default=None,
                        help="每隔几轮更新一次 GMM，默认用 config 里的值（1）")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay (0.999 recommended); set 0 to disable EMA")
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", type=str, default="owlv2-csma")
    parser.add_argument("--swanlab-run-name", type=str, default="run")
    args = parser.parse_args()

    overrides = {"ir_data_root": args.data_root, "output_dir": args.out_dir, "total_epochs": args.epochs}
    if args.batch_size is not None: overrides["batch_size"] = args.batch_size
    if args.num_workers is not None: overrides["num_workers"] = args.num_workers
    if args.lr is not None: overrides["lr"] = args.lr
    if args.residual_scale is not None: overrides["residual_scale"] = args.residual_scale
    if args.pseudo_clamp is not None: overrides["pseudo_rgb_clamp"] = args.pseudo_clamp
    if args.id_loss_weight is not None: overrides["id_loss_weight"] = args.id_loss_weight
    if args.tv_loss_weight is not None: overrides["tv_loss_weight"] = args.tv_loss_weight
    if args.logit_reg_weight is not None: overrides["logit_reg_weight"] = args.logit_reg_weight
    if args.logit_cap is not None: overrides["logit_cap"] = args.logit_cap
    if args.car_loss_weight is not None: overrides["car_loss_weight"] = args.car_loss_weight
    if args.person_loss_weight is not None: overrides["person_loss_weight"] = args.person_loss_weight
    if args.grad_skip_threshold is not None: overrides["grad_skip_threshold"] = args.grad_skip_threshold
    if args.disable_group_norm:
        overrides["use_group_norm"] = False
    if args.force_amp:
        overrides["use_amp"] = True
    if args.max_steps is not None: overrides["max_steps_per_epoch"] = args.max_steps
    if args.stage_boundaries:
        overrides["stage_epoch_boundaries"] = [int(x) for x in args.stage_boundaries.split(",")]
    if args.stage_weights:
        try:
            parsed = [[float(v) for v in s.split(",")] for s in args.stage_weights.split(";")]
            if len(parsed) != 3 or any(len(p) != 2 for p in parsed):
                raise ValueError()
            overrides["stage_loss_weights"] = [tuple(p) for p in parsed]
        except Exception:
            raise ValueError("--stage-weights 格式错误，应为 'a0,d0;a1,d1;a2,d2'")
    if args.gmm_update_every is not None:
        overrides["gmm_update_every"] = args.gmm_update_every
    cfg = CSMAConfig.from_overrides(overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(cfg.output_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "logs"), exist_ok=True)

    processor = Owlv2Processor.from_pretrained(cfg.model_id)
    owlv2 = Owlv2ForObjectDetection.from_pretrained(cfg.model_id).to(device).eval()
    for p in owlv2.parameters():
        p.requires_grad = False

    with torch.no_grad():
        enc = processor(text=[cfg.text_labels], images=None, return_tensors="pt", padding=True)
    base_ids = enc["input_ids"].to(device)
    base_atm = enc["attention_mask"].to(device)

    csma = CSMA(cfg).to(device)
    if args.init_ckpt:
        raw = torch.load(args.init_ckpt, map_location=device, weights_only=True)
        csma.load_state_dict(raw["csma"] if isinstance(raw, dict) and "csma" in raw else raw)
    csma.train()

    opt = AdamW(csma.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = max(0, args.warmup_epochs)
    if warmup > 0:
        sched = SequentialLR(opt, [LinearLR(opt, 0.1, 1.0, warmup), CosineAnnealingLR(opt, max(1, cfg.total_epochs - warmup))], [warmup])
    else:
        sched = CosineAnnealingLR(opt, T_max=cfg.total_epochs)

    cmss_sched = CMSSScheduler(cfg)
    b0, b1 = cmss_sched.stage_boundaries
    print(f"[train] epochs={cfg.total_epochs} lr={cfg.lr} stages Easy[0,{b0}) Mixed[{b0},{b1}) Hard[{b1},{cfg.total_epochs})")

    cat_map, valid_ids = build_flir_v1_category_map(cfg.text_labels)
    train_ds = FlirV1PairedDataset(
        cfg.ir_data_root, processor, cfg.text_labels, cat_map, valid_ids, require_rgb=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_flir_v1,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_ds = (
        FlirV1PairedDataset(args.val_root, processor, cfg.text_labels, cat_map, valid_ids, require_rgb=True)
        if os.path.isdir(args.val_root)
        else None
    )

    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    signal.signal(signal.SIGUSR1, _sigusr1_handler)

    ema = ModelEMA(csma, decay=args.ema_decay)
    print(f"[train] EMA decay={args.ema_decay}" + (" (disabled)" if args.ema_decay == 0 else ""))

    # SwanLab
    swan_run = None
    if args.use_swanlab:
        try:
            import swanlab  # type: ignore
            swan_run = swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_run_name,
                config=cfg.to_dict(),
            )
            print(f"[SwanLab] 已初始化: project={args.swanlab_project} run={args.swanlab_run_name}")
        except Exception as e:
            print(f"[SwanLab] 初始化失败，继续本地训练: {e}")

    best_ap50, no_improve, loss_hist, val_hist = 0.0, 0, [], []
    grad_checked = False

    for epoch in range(args.start_epoch, cfg.total_epochs):
        if cmss_sched.should_update_gmm(epoch):
            vals = collect_cmss_values(owlv2, csma, train_loader, device, base_ids, base_atm, cfg.gmm_max_batches)
            if len(vals) >= cfg.gmm_n_components:
                cmss_sched.update_gmm(vals)
                print(f"[train] GMM means: {cmss_sched.sorted_means}")
        stage = cmss_sched.get_stage(epoch)
        la, ld = cmss_sched.get_loss_weights(epoch)
        ld = max(ld, 0.05)  # teammate default
        mu1, mu2, mu3 = cmss_sched.sorted_means
        ep_loss = []

        for step, batch in enumerate(train_loader, 1):
            if cfg.max_steps_per_epoch != -1 and step > cfg.max_steps_per_epoch:
                break
            B = batch["pixel_values"].shape[0]
            ids_b, atm_b = base_ids.repeat(B, 1), base_atm.repeat(B, 1)
            ir_in = batch["pixel_values"].to(device)
            gt_boxes = [g.to(device) for g in batch["gt_boxes"]]
            gt_labels = [g.to(device) for g in batch["gt_labels"]]
            with _autocast(use_amp):
                pseudo = csma(ir_in)
                out = owlv2(input_ids=ids_b, pixel_values=pseudo, attention_mask=atm_b)
                l_det, det_sc = compute_det_loss(out.pred_boxes, out.logits, gt_boxes, gt_labels, cfg)
                l_align = torch.tensor(0.0, device=device)
                l_id = F.l1_loss(pseudo, ir_in)
                l_tv = _total_variation_loss(pseudo)
                l_logit_reg = torch.relu(out.logits.abs() - cfg.logit_cap).mean()
                if "rgb_pixel_values" in batch:
                    rgb = batch["rgb_pixel_values"].to(device)
                    rgb_idx = batch.get("rgb_indices", list(range(B)))
                    f_ir_full = out.image_embeds.flatten(1, 2)
                    # 只取 batch 内有 RGB 的样本对应的 IR 特征
                    f_ir_sub = f_ir_full[rgb_idx]
                    ids_sub = base_ids.repeat(len(rgb_idx), 1)
                    atm_sub = base_atm.repeat(len(rgb_idx), 1)
                    with torch.no_grad():
                        f_rgb = owlv2(input_ids=ids_sub, pixel_values=rgb, attention_mask=atm_sub).image_embeds.flatten(1, 2).detach()
                    mask = build_cmss_mask(compute_cmss(f_rgb, f_ir_sub.detach()), stage, mu1, mu2, mu3, cfg.mask_ratio, cmss_sched.gmm)
                    l_align = compute_align_loss(f_ir_sub, f_rgb, mask)
                loss = (
                    la * l_align + ld * l_det +
                    cfg.id_loss_weight * l_id +
                    cfg.tv_loss_weight * l_tv +
                    cfg.logit_reg_weight * l_logit_reg
                )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if not grad_checked:
                g = sum(p.grad.abs().sum().item() for p in csma.parameters() if p.grad is not None)
                if not (math.isnan(g) or math.isinf(g)):
                    print(f"[grad check] CSMA grad L1={g:.4f} OK")
                    grad_checked = True
            grad_norm = nn.utils.clip_grad_norm_(csma.parameters(), cfg.grad_clip)
            grad_norm_val = float(grad_norm.detach()) if torch.is_tensor(grad_norm) else float(grad_norm)
            if (not math.isfinite(grad_norm_val)) or grad_norm_val > cfg.grad_skip_threshold:
                opt.zero_grad(set_to_none=True)
                scaler.update()
                print(f"  [skip step] grad_norm={grad_norm_val:.4f} > {cfg.grad_skip_threshold:.4f}")
                continue
            scaler.step(opt); scaler.update()
            ema.update(csma)
            ep_loss.append(float(loss.detach()))
            n_steps_epoch = len(train_loader) if cfg.max_steps_per_epoch == -1 else min(cfg.max_steps_per_epoch, len(train_loader))
            if step == 1 or step % 50 == 0:
                print(
                    f"  Ep{epoch+1:02d}/{cfg.total_epochs} Step{step:4d}/{n_steps_epoch}"
                    f" loss={loss.item():.4f} det={det_sc['loss_det']:.4f}"
                    f" align={float(l_align):.4f} logit_reg={float(l_logit_reg):.4f}"
                    f" la={la:.2f} ld={ld:.2f}",
                    flush=True,
                )
            if swan_run is not None:
                swan_run.log({
                    "train/loss": float(loss.detach()),
                    "train/loss_det": det_sc["loss_det"],
                    "train/loss_align": float(l_align),
                    "train/loss_id": float(l_id),
                    "train/loss_tv": float(l_tv),
                    "train/loss_logit_reg": float(l_logit_reg),
                    "train/grad_norm": grad_norm_val,
                    "train/stage": stage,
                })

        sched.step()
        mean_loss = float(np.mean(ep_loss)) if ep_loss else 0.0
        loss_hist.append(mean_loss)
        print(f"Epoch {epoch+1}/{cfg.total_epochs} stage={stage} loss={mean_loss:.5f} la={la} ld={ld}")
        if swan_run is not None:
            swan_run.log({"train/loss_epoch": mean_loss, "train/epoch": epoch + 1})

        if val_ds and (epoch + 1) % args.val_every == 0:
            ema.copy_to(csma)
            metrics = run_validation(csma, owlv2, processor, val_ds, valid_ids, device, cfg.text_labels, cfg.batch_size, args.val_threshold)
            ema.restore(csma)
            metrics["epoch"] = epoch + 1
            val_hist.append(metrics)
            with open(os.path.join(cfg.output_dir, "logs", "val_metrics.jsonl"), "a") as f:
                f.write(json.dumps(metrics) + "\n")
            print(
                f"  [val Ep{epoch+1:02d}] AP50={metrics['AP50']:.4f}  AP={metrics['AP']:.4f}"
                f"  AP_person={metrics['AP_person']:.4f}  AP_car={metrics['AP_car']:.4f}"
                f"  n_preds={int(metrics['n_preds'])}  (EMA)",
                flush=True,
            )
            if swan_run is not None:
                swan_run.log({f"val/{k}": v for k, v in metrics.items()})
            if metrics["AP50"] > best_ap50:
                best_ap50 = metrics["AP50"]; no_improve = 0
                ema.copy_to(csma)
                ckpt_payload = {"csma": csma.state_dict(), "config_overrides": cfg.to_dict()}
                torch.save(ckpt_payload, os.path.join(ckpt_dir, "best.pt"))
                # 保留每次刷新 best 时的独立副本，避免后续 epoch 覆盖
                ep_tag = f"best_ap50_ep{epoch + 1:02d}.pt"
                torch.save(ckpt_payload, os.path.join(ckpt_dir, ep_tag))
                ema.restore(csma)
                print(f"  [ckpt] best.pt (EMA) AP50={best_ap50:.4f}  + {ep_tag}", flush=True)
            else:
                no_improve += 1
                if args.patience > 0 and no_improve >= args.patience:
                    print("[train] early stop"); break

            # 每 epoch 保存 EMA 权重，便于按 ep 回溯
            ema.copy_to(csma)
            torch.save(
                {"csma": csma.state_dict(), "config_overrides": cfg.to_dict()},
                os.path.join(ckpt_dir, f"epoch_{epoch + 1:02d}_ema.pt"),
            )
            ema.restore(csma)

        torch.save(csma.state_dict(), os.path.join(ckpt_dir, "latest.pt"))
        with open(os.path.join(ckpt_dir, "latest_meta.json"), "w") as f:
            json.dump({"epoch": epoch, "completed_epochs": epoch + 1, "best_ap50": best_ap50}, f, indent=2)
        if epoch % cfg.vis_every == 0:
            torch.save(csma.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"))

    final = os.path.join(ckpt_dir, "csma_last.pt")
    torch.save(csma.state_dict(), final)
    print(f"[train] done best AP50={best_ap50:.4f} final={final}")
    if swan_run is not None:
        try:
            swan_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    main()
