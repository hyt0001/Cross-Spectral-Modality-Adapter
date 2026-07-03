"""
LLVIP 数据集上 DETR 检测评估（ir_raw 基线 + CSMA 伪 RGB）。

LLVIP：1280×1024 红外人员检测，只有 person 类。
用于验证 CSMA 的跨数据集泛化能力（训练在 FLIR，测试在 LLVIP）。

Teacher 匹配规则（重要）：
  本脚本只评估 DETR-CSMA → DETR。必须使用 DETR teacher 训练的权重
  （默认 outputs_csma_detr_base/ckpt/best_stage1.pt）。
  GDINO-CSMA 请用 src.eval_llvip_gdino，不要混用。

坐标处理：
  ir_raw     — 原始 1280×1024 直接送 DETR
  ir_pipeline — 诊断：IR 走 512 链路但不过 CSMA
  pseudo_rgb — processor 512 → CSMA → denorm → DETR（native 坐标映射）

CLI 示例：
    # ir_raw 基线
    conda run -n RGBtest python -m src.eval_llvip_detr \\
        --out-json outputs_csma/logs/llvip/eval_llvip_detr_ir_raw.json

    # DETR-CSMA 伪 RGB（teacher 匹配）
    conda run -n RGBtest python -m src.eval_llvip_detr \\
        --input-mode pseudo_rgb \\
        --ckpt outputs_csma_detr_base/ckpt/best_stage1.pt \\
        --out-json outputs_csma/logs/llvip/eval_llvip_detr_csma.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, DetrForObjectDetection, DetrImageProcessor

import numpy as np
from src.config import CSMAConfig
from src.csma import CSMA
from src.eval_csma import compute_map
from src.eval_yolo_csma import _configure_processor
from src.infer_vis import denormalize_pixel_values

# LLVIP 只有 person，DETR COCO label 1 = person → LLVIP eval cat 1
DETR_LABEL_TO_LLVIP_CAT: Dict[int, int] = {1: 1}

LLVIP_IR_DIR  = "/root/autodl-tmp/LLVIP/LLVIP/infrared/test"
LLVIP_GT_JSON = "/root/autodl-tmp/LLVIP/LLVIP/llvip_test.json"

# DETR teacher 训练的 CSMA 默认权重（不要用 GDINO/YOLO 线权重）
DETR_CSMA_CKPT_DEFAULT = "outputs_csma_detr_base/ckpt/best_stage1.pt"

# pseudo_rgb 送入检测器前的尺寸策略（与 FLIR eval_yolo_csma 对齐）
#   native  — CSMA 输出分辨率直接检测，预测框再映射回 orig_w×orig_h（推荐）
#   upscale — 双线性放大到原图尺寸再检测（旧行为，易模糊劣化）
PseudoResizeMode = Literal["native", "upscale"]


def valid_hw_from_mask(pixel_mask: torch.Tensor) -> Tuple[int, int]:
    oh = int(pixel_mask[:, 0].sum().item())
    ow = int(pixel_mask[0, :].sum().item())
    return oh, ow


def pseudo_rgb_uint8_from_csma(
    pseudo_chw: torch.Tensor,
    pixel_mask: torch.Tensor,
    gdino_mean: Tuple[float, ...],
    gdino_std: Tuple[float, ...],
) -> np.ndarray:
    oh, ow = valid_hw_from_mask(pixel_mask)
    arr = denormalize_pixel_values(pseudo_chw.cpu(), gdino_mean, gdino_std)
    return arr[:oh, :ow]


def prepare_pseudo_rgb_for_detector(
    arr: np.ndarray,
    orig_w: int,
    orig_h: int,
    resize_mode: PseudoResizeMode,
) -> Tuple[np.ndarray, float, float]:
    """返回 (检测器输入图, box_scale_x, box_scale_y)。"""
    oh, ow = arr.shape[:2]
    if resize_mode == "upscale":
        pil = Image.fromarray(arr).resize((orig_w, orig_h), Image.BILINEAR)
        return np.asarray(pil, dtype=np.uint8), 1.0, 1.0
    if resize_mode == "native":
        return arr, orig_w / ow, orig_h / oh
    raise ValueError(f"未知 resize_mode: {resize_mode}")


def scale_xyxy_boxes(xyxy: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    if scale_x == 1.0 and scale_y == 1.0:
        return xyxy
    out = xyxy.copy()
    out[:, [0, 2]] *= scale_x
    out[:, [1, 3]] *= scale_y
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class LLVIPIRDataset(Dataset):
    """LLVIP 红外 test 集，返回原始 PIL 及元信息。"""

    def __init__(
        self,
        ir_dir: str,
        gt_json: str,
        gdino_processor: Any,
        cfg: CSMAConfig,
    ) -> None:
        self._ir_dir = ir_dir
        self._cfg = cfg
        self._gdino_proc = gdino_processor

        with open(gt_json, encoding="utf-8") as f:
            coco_data = json.load(f)

        ir_files = set(os.listdir(ir_dir))
        self._images = [
            img for img in coco_data["images"]
            if img["file_name"] in ir_files
        ]
        self._images.sort(key=lambda x: x["id"])

        img_ids = {img["id"] for img in self._images}
        self._annotations = [
            ann for ann in coco_data["annotations"]
            if ann["image_id"] in img_ids
        ]

        print(f"[LLVIPIRDataset] images={len(self._images)}  annotations={len(self._annotations)}")

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self._images[idx]
        ir_path = os.path.join(self._ir_dir, info["file_name"])
        pil_ir = Image.open(ir_path).convert("RGB")
        orig_w, orig_h = pil_ir.size

        # GDINO processor：缩放到 shortest_edge=cfg.img_size，供 CSMA 使用
        enc = self._gdino_proc(images=pil_ir, return_tensors="pt")
        pixel_values = enc["pixel_values"][0]   # [3, H', W']
        pixel_mask   = enc["pixel_mask"][0]     # [H', W']  1=有效

        return {
            "image_id":    info["id"],
            "file_name":   info["file_name"],
            "orig_w":      orig_w,
            "orig_h":      orig_h,
            "pixel_values": pixel_values,
            "pixel_mask":   pixel_mask,
        }


def _collate_llvip(batch: List[Dict]) -> Dict[str, Any]:
    pv_list = [b["pixel_values"] for b in batch]
    pm_list = [b["pixel_mask"]   for b in batch]
    max_h = max(pv.shape[1] for pv in pv_list)
    max_w = max(pv.shape[2] for pv in pv_list)
    padded_pv, padded_pm = [], []
    for pv, pm in zip(pv_list, pm_list):
        _, h, w = pv.shape
        pv_pad = torch.zeros(pv.shape[0], max_h, max_w, dtype=pv.dtype)
        pv_pad[:, :h, :w] = pv
        pm_pad = torch.zeros(max_h, max_w, dtype=pm.dtype)
        pm_pad[:h, :w] = pm
        padded_pv.append(pv_pad)
        padded_pm.append(pm_pad)
    return {
        "image_ids":    [b["image_id"]  for b in batch],
        "file_names":   [b["file_name"] for b in batch],
        "orig_ws":      [b["orig_w"]    for b in batch],
        "orig_hs":      [b["orig_h"]    for b in batch],
        "pixel_values": torch.stack(padded_pv),
        "pixel_masks":  torch.stack(padded_pm),
    }


# ──────────────────────────────────────────────────────────────────────────────
# COCO GT
# ──────────────────────────────────────────────────────────────────────────────

def _build_llvip_gt(dataset: LLVIPIRDataset) -> COCO:
    coco_dict: Dict[str, Any] = {
        "images":      dataset._images,
        "annotations": dataset._annotations,
        "categories":  [{"id": 1, "name": "person"}],
    }
    coco = COCO()
    coco.dataset = coco_dict
    coco.createIndex()
    return coco


# ──────────────────────────────────────────────────────────────────────────────
# Eval loop
# ──────────────────────────────────────────────────────────────────────────────

def _assert_detr_teacher_ckpt(ckpt: str, allow_cross_teacher: bool = False) -> None:
    """拒绝把 GDINO/YOLO 线 CSMA 权重用于 DETR 评估（--allow-cross-teacher 可显式放行做对照实验）。"""
    if allow_cross_teacher:
        return
    norm = ckpt.replace("\\", "/")
    if "outputs_csma_detr" in norm:
        return
    non_detr_markers = (
        "outputs_csma/ckpt/",
        "outputs_csma_yolo",
        "outputs_csma_v3tiny",
        "outputs_csma_yolov8",
    )
    if any(m in norm for m in non_detr_markers):
        raise ValueError(
            f"eval_llvip_detr 必须使用 DETR teacher 训练的 CSMA 权重，"
            f"当前 ckpt={ckpt!r} 属于 GDINO/YOLO 线。"
            f"请改用 {DETR_CSMA_CKPT_DEFAULT}，"
            f"GDINO-CSMA 请运行 python -m src.eval_llvip_gdino。"
        )


def _load_csma(ckpt: str, cfg: CSMAConfig, device: torch.device) -> CSMA:
    csma = CSMA(cfg).to(device)
    raw = torch.load(ckpt, map_location=device, weights_only=True)
    state = raw["csma"] if isinstance(raw, dict) and "csma" in raw else raw
    csma.load_state_dict(state)
    csma.eval()
    return csma


def adapt_csma_bn(
    csma: CSMA,
    dataset: LLVIPIRDataset,
    device: torch.device,
    adapt_steps: int,
    batch_size: int = 8,
) -> None:
    """
    Test-Time BatchNorm Adaptation (AdaBN).

    在目标域图像上做纯 forward（无梯度、无权重更新），以 train 模式让 BatchNorm
    把 running_mean/var 更新到目标域分布，再切回 eval 模式。
    解决 CSMA 在 FLIR 上训练积累的 BN 统计量与 LLVIP 分布不匹配的问题。
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=_collate_llvip, num_workers=2,
    )
    csma.train()
    done = 0
    with torch.no_grad():
        for batch in loader:
            ir_pv = batch["pixel_values"].to(device)
            csma(ir_pv)
            done += 1
            if done >= adapt_steps:
                break
    csma.eval()
    print(f"[AdaBN] 已在目标域更新 BN 统计量（{done} 步 / {done * batch_size} 张图）")


class _LLVIPRawIRDataset(Dataset):
    """LLVIP 红外目录的纯图像 Dataset，无需 GT JSON，用于无监督 TTA。"""

    def __init__(self, ir_dir: str, gdino_processor: Any, cfg: CSMAConfig) -> None:
        self._ir_dir = ir_dir
        self._proc = gdino_processor
        self._cfg = cfg
        self._files = sorted(
            f for f in os.listdir(ir_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        print(f"[LLVIPRawIRDataset] {ir_dir}: {len(self._files)} 张图")

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pil = Image.open(os.path.join(self._ir_dir, self._files[idx])).convert("RGB")
        enc = self._proc(images=pil, return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"][0],
            "pixel_mask":   enc["pixel_mask"][0],
        }


def _collate_raw(batch: List[Dict]) -> Dict[str, Any]:
    pv_list = [b["pixel_values"] for b in batch]
    pm_list = [b["pixel_mask"]   for b in batch]
    max_h = max(pv.shape[1] for pv in pv_list)
    max_w = max(pv.shape[2] for pv in pv_list)
    padded_pv, padded_pm = [], []
    for pv, pm in zip(pv_list, pm_list):
        _, h, w = pv.shape
        pv_pad = torch.zeros(pv.shape[0], max_h, max_w, dtype=pv.dtype)
        pv_pad[:, :h, :w] = pv
        pm_pad = torch.zeros(max_h, max_w, dtype=pm.dtype)
        pm_pad[:h, :w] = pm
        padded_pv.append(pv_pad)
        padded_pm.append(pm_pad)
    return {
        "pixel_values": torch.stack(padded_pv),
        "pixel_masks":  torch.stack(padded_pm),
    }


def unsupervised_tta(
    csma: CSMA,
    tta_ir_dir: str,
    gdino_processor: Any,
    device: torch.device,
    tta_steps: int,
    batch_size: int = 8,
    lr: float = 2e-5,
    id_w: float = 0.005,
    tv_w: float = 0.05,
) -> None:
    """
    无监督 Test-Time Adaptation（TTA）。

    在目标域 IR 图上用 L_id + L_tv 做 tta_steps 步梯度更新，让 CSMA
    学会为 LLVIP 产生结构保留、平滑的 pseudo_rgb。不需要任何标注。

    L_id：L1(pseudo_rgb, input)  — 结构保留
    L_tv：total variation         — 抑制噪点
    """
    cfg = csma.cfg
    tta_ds = _LLVIPRawIRDataset(tta_ir_dir, gdino_processor, cfg)
    loader = DataLoader(
        tta_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate_raw, num_workers=2,
    )
    optimizer = AdamW(csma.parameters(), lr=lr, weight_decay=1e-4)
    csma.train()
    done = 0
    loss_acc = 0.0
    for batch in loader:
        ir_pv = batch["pixel_values"].to(device)
        pseudo = csma(ir_pv)

        l_id = F.l1_loss(pseudo, ir_pv) * id_w
        diff_h = pseudo[:, :, 1:, :] - pseudo[:, :, :-1, :]
        diff_w = pseudo[:, :, :, 1:] - pseudo[:, :, :, :-1]
        l_tv = (diff_h.abs().mean() + diff_w.abs().mean()) * tv_w
        loss = l_id + l_tv

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(csma.parameters(), 1.0)
        optimizer.step()

        loss_acc += float(loss.detach())
        done += 1
        if done >= tta_steps:
            break
    csma.eval()
    print(f"[TTA] {done} 步，avg loss={loss_acc/done:.5f}  (id_w={id_w}, tv_w={tv_w})")


def run_eval(
    detr_model: DetrForObjectDetection,
    detr_processor: DetrImageProcessor,
    dataset: LLVIPIRDataset,
    device: torch.device,
    batch_size: int,
    threshold: float,
    input_mode: str,
    csma: Optional[CSMA],
    gdino_mean: Tuple,
    gdino_std: Tuple,
    pseudo_resize: PseudoResizeMode = "native",
) -> List[Dict]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_llvip, num_workers=2,
    )
    detr_model.eval()
    predictions: List[Dict] = []
    total = len(dataset)
    processed = 0

    with torch.no_grad():
        for batch in loader:
            image_ids = batch["image_ids"]
            orig_ws   = batch["orig_ws"]
            orig_hs   = batch["orig_hs"]
            file_names = batch["file_names"]
            bsz = len(image_ids)

            box_scales: List[Tuple[float, float]] = [(1.0, 1.0)] * bsz

            if input_mode in ("pseudo_rgb", "ir_pipeline"):
                ir_pv = batch["pixel_values"].to(device)
                pm = batch["pixel_masks"]
                if input_mode == "pseudo_rgb":
                    assert csma is not None
                    feat_batch = csma(ir_pv)
                else:
                    # 诊断：CSMA=恒等，走与 pseudo_rgb 相同的 512 链路
                    feat_batch = ir_pv

                pil_images = []
                box_scales = []
                for i in range(bsz):
                    arr = pseudo_rgb_uint8_from_csma(
                        feat_batch[i], pm[i], gdino_mean, gdino_std,
                    )
                    det_arr, sx, sy = prepare_pseudo_rgb_for_detector(
                        arr, orig_ws[i], orig_hs[i], pseudo_resize,
                    )
                    pil_images.append(Image.fromarray(det_arr))
                    box_scales.append((sx, sy))
            else:
                # ir_raw：直接读原始红外图
                pil_images = [
                    Image.open(os.path.join(dataset._ir_dir, fn)).convert("RGB")
                    for fn in file_names
                ]

            inputs = detr_processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = detr_model(**inputs)

            target_sizes = torch.tensor(
                [[img.size[1], img.size[0]] for img in pil_images],  # (H, W)
                dtype=torch.float32, device=device,
            )
            results = detr_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=threshold,
            )

            for res, img_id, (sx, sy) in zip(results, image_ids, box_scales):
                boxes = scale_xyxy_boxes(res["boxes"].cpu().numpy(), sx, sy)
                scores = res["scores"].cpu().numpy()
                labels = res["labels"].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    cat_id = DETR_LABEL_TO_LLVIP_CAT.get(int(label))
                    if cat_id is None:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    predictions.append({
                        "image_id":    img_id,
                        "category_id": cat_id,
                        "bbox":        [x1, y1, x2 - x1, y2 - y1],
                        "score":       float(score),
                    })

            processed += bsz
            print(f"\r  DETR 推理进度: {processed}/{total}", end="", flush=True)

    print()
    return predictions


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLVIP IR 上 DETR 评估（ir_raw 基线 + CSMA 伪 RGB 泛化测试）"
    )
    parser.add_argument("--ir-dir",    type=str, default=LLVIP_IR_DIR)
    parser.add_argument("--gt-json",   type=str, default=LLVIP_GT_JSON)
    parser.add_argument("--model-id",  type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--input-mode", type=str, default="ir_raw",
                        choices=["ir_raw", "pseudo_rgb", "ir_pipeline"])
    parser.add_argument("--ckpt",      type=str, default=DETR_CSMA_CKPT_DEFAULT,
                        help="DETR-CSMA 权重（须为 outputs_csma_detr* 目录）")
    parser.add_argument("--out-json",  type=str,
                        default="outputs_csma/logs/eval_llvip_detr_ir_raw.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument(
        "--pseudo-resize",
        type=str,
        default="native",
        choices=["native", "upscale"],
        help="pseudo_rgb 尺寸策略：native=CSMA 分辨率检测后映射坐标；upscale=放大到原图再检测",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=None,
        help="覆盖 CSMAConfig.residual_scale（默认 1.0）。0.0=纯 tanh，0.5=半残差",
    )
    parser.add_argument(
        "--pseudo-clamp",
        type=float,
        default=None,
        help="覆盖 CSMAConfig.pseudo_clamp（默认 2.0）。0.0=不截断",
    )
    parser.add_argument(
        "--adapt-bn",
        type=int,
        default=0,
        metavar="N",
        help="AdaBN：在目标域做 N 步 forward（train 模式）更新 BN 统计量后再切 eval",
    )
    parser.add_argument(
        "--allow-cross-teacher",
        action="store_true",
        help="允许加载非 DETR teacher 的 CSMA 权重（跨 teacher 对照实验）",
    )
    args = parser.parse_args()

    if args.input_mode == "pseudo_rgb":
        _assert_detr_teacher_ckpt(args.ckpt, allow_cross_teacher=args.allow_cross_teacher)
        if not os.path.isfile(args.ckpt):
            raise FileNotFoundError(f"DETR-CSMA 权重不存在: {args.ckpt}")
    if args.input_mode == "ir_pipeline" and args.ckpt:
        print("[eval_llvip_detr] 警告: ir_pipeline 忽略 --ckpt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_llvip_detr] 设备:       {device}")
    print(f"[eval_llvip_detr] input_mode: {args.input_mode}")
    if args.input_mode == "pseudo_rgb":
        print(f"[eval_llvip_detr] pseudo_resize: {args.pseudo_resize}")
    elif args.input_mode == "ir_pipeline":
        print(f"[eval_llvip_detr] ir_pipeline pseudo_resize: {args.pseudo_resize}")
    print(f"[eval_llvip_detr] threshold:  {args.threshold}")

    cfg = CSMAConfig()
    if args.residual_scale is not None:
        cfg.residual_scale = args.residual_scale
    if args.pseudo_clamp is not None:
        cfg.pseudo_clamp = args.pseudo_clamp
    dino_proc = AutoProcessor.from_pretrained(cfg.model_id)
    _configure_processor(dino_proc, cfg.img_size)   # 与 FLIR eval 一致
    ip = dino_proc.image_processor
    gdino_mean = tuple(ip.image_mean)
    gdino_std  = tuple(ip.image_std)

    print("[eval_llvip_detr] 加载 DETR...")
    detr_proc  = DetrImageProcessor.from_pretrained(args.model_id)
    detr_model = DetrForObjectDetection.from_pretrained(args.model_id).to(device)
    detr_model.eval()
    for p in detr_model.parameters():
        p.requires_grad = False

    csma: Optional[CSMA] = None
    if args.input_mode == "pseudo_rgb":
        print(f"[eval_llvip_detr] 加载 CSMA: {args.ckpt}")
        if args.residual_scale is not None:
            print(f"[eval_llvip_detr] residual_scale 覆盖: {args.residual_scale}")
        if args.pseudo_clamp is not None:
            print(f"[eval_llvip_detr] pseudo_clamp 覆盖: {args.pseudo_clamp}")
        csma = _load_csma(args.ckpt, cfg, device)

    dataset = LLVIPIRDataset(args.ir_dir, args.gt_json, dino_proc, cfg)
    coco_gt = _build_llvip_gt(dataset)
    print(f"[eval_llvip_detr] GT 框数: {len(coco_gt.anns)}")

    if csma is not None and args.adapt_bn > 0:
        print(f"[eval_llvip_detr] AdaBN: {args.adapt_bn} 步 ...")
        adapt_csma_bn(csma, dataset, device, adapt_steps=args.adapt_bn, batch_size=args.batch_size)

    predictions = run_eval(
        detr_model=detr_model,
        detr_processor=detr_proc,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        threshold=args.threshold,
        input_mode=args.input_mode,
        csma=csma,
        gdino_mean=gdino_mean,
        gdino_std=gdino_std,
        pseudo_resize=args.pseudo_resize,  # type: ignore[arg-type]
    )
    print(f"[eval_llvip_detr] 预测框数: {len(predictions)}")

    results = compute_map(coco_gt, predictions)
    results.update({
        "model_id":   args.model_id,
        "input_mode": args.input_mode,
        "ckpt":       args.ckpt or None,
        "dataset":    "LLVIP",
        "threshold":   args.threshold,
        "pseudo_resize": args.pseudo_resize if args.input_mode in ("pseudo_rgb", "ir_pipeline") else None,
        "residual_scale": args.residual_scale,
        "pseudo_clamp":   args.pseudo_clamp,
        "adapt_bn":       args.adapt_bn,
    })

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 55)
    print(f"  dataset       : LLVIP (person only)")
    print(f"  input_mode    : {args.input_mode}")
    print(f"  mAP@0.5       : {results['map_50']:.4f}")
    print(f"  mAP@0.5:0.95  : {results['map_50_95']:.4f}")
    print(f"  AP_person@0.5 : {results['ap_person']:.4f}")
    print(f"  (LLVIP 无 car 类别，ap_car 字段忽略)")
    print(f"  预测框 / GT框  : {results['n_preds']} / {results['n_gt']}")
    print("=" * 55)
    print(f"[eval_llvip_detr] 结果: {args.out_json}")


if __name__ == "__main__":
    main()
