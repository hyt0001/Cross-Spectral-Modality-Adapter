"""
MVP：在 10 张 FLIR 图上过拟合 ``ResidualTranslator``，冻结 ``GroundingDinoForObjectDetection``，
使用 HF 内建 ``outputs.loss`` 反传梯度至翻译网络。
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from src.dataset import FlirCocoOverfitDataset, build_coco_category_to_class_index, collate_fn
from src.infer_vis import save_multi_sample_grid
from src.translator import ResidualTranslator

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "person. car."
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0
VIS_EVERY = 50


def _move_labels_to_device(labels: List[Dict[str, Any]], device: torch.device) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for lab in labels:
        entry: Dict[str, Any] = {}
        for k, v in lab.items():
            if isinstance(v, torch.Tensor):
                entry[k] = v.to(device)
            else:
                entry[k] = v
        out.append(entry)
    return out


def _build_swanlab_logger(
    enable: bool,
    project: str,
    run_name: str,
    config: Dict[str, Any],
):
    """按需初始化 SwanLab；未安装或关闭时返回 None。"""
    if not enable:
        return None
    try:
        import swanlab  # type: ignore

        run = swanlab.init(project=project, experiment_name=run_name, config=config)
        return run
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[SwanLab] 初始化失败，继续本地训练: {exc}")
        return None


def _get_weighted_loss(outputs: Any) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    使用 loss_dict 重构加权损失，降低 encoder 分支权重。
    """
    loss_dict = outputs.loss_dict
    loss_ce = loss_dict["loss_ce"]
    loss_bbox = loss_dict["loss_bbox"]
    loss_giou = loss_dict["loss_giou"]
    loss_ce_enc = loss_dict["loss_ce_enc"]
    loss_bbox_enc = loss_dict["loss_bbox_enc"]
    loss_giou_enc = loss_dict["loss_giou_enc"]

    weighted_loss = (
        loss_ce
        + loss_bbox * 5.0
        + loss_giou * 2.0
        + loss_ce_enc * 0.1
        + loss_bbox_enc * 0.5
        + loss_giou_enc * 0.5
    )
    scalars = {
        "loss_total_raw": float(outputs.loss.detach().cpu()),
        "loss_weighted": float(weighted_loss.detach().cpu()),
        "loss_ce": float(loss_ce.detach().cpu()),
        "loss_bbox": float(loss_bbox.detach().cpu()),
        "loss_giou": float(loss_giou.detach().cpu()),
        "loss_ce_enc": float(loss_ce_enc.detach().cpu()),
        "loss_bbox_enc": float(loss_bbox_enc.detach().cpu()),
        "loss_giou_enc": float(loss_giou_enc.detach().cpu()),
    }
    return weighted_loss, scalars


def _build_train_loss(
    outputs: Any,
    mode: str,
    enc_weight: float,
    enc_bbox_weight: float,
    enc_giou_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    依据训练模式构建用于反传的 loss。
    - main_only: 仅 decoder 主损失
    - full: 主损失 + encoder 分支（可调权重）
    """
    loss_dict = outputs.loss_dict
    loss_ce = loss_dict["loss_ce"]
    loss_bbox = loss_dict["loss_bbox"]
    loss_giou = loss_dict["loss_giou"]
    main_loss = loss_ce + loss_bbox * 5.0 + loss_giou * 2.0

    if mode == "main_only":
        train_loss = main_loss
    elif mode == "full":
        train_loss = (
            main_loss
            + loss_dict["loss_ce_enc"] * enc_weight
            + loss_dict["loss_bbox_enc"] * enc_bbox_weight
            + loss_dict["loss_giou_enc"] * enc_giou_weight
        )
    else:
        raise ValueError(f"未知 loss 模式: {mode}")

    return train_loss, {
        "loss_total_raw": float(outputs.loss.detach().cpu()),
        "loss_main": float(main_loss.detach().cpu()),
        "loss_train": float(train_loss.detach().cpu()),
        "loss_ce": float(loss_ce.detach().cpu()),
        "loss_bbox": float(loss_bbox.detach().cpu()),
        "loss_giou": float(loss_giou.detach().cpu()),
        "loss_ce_enc": float(loss_dict["loss_ce_enc"].detach().cpu()),
        "loss_bbox_enc": float(loss_dict["loss_bbox_enc"].detach().cpu()),
        "loss_giou_enc": float(loss_dict["loss_giou_enc"].detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP：冻结 DINO + 训练 ResidualTranslator")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数（默认 200 过拟合）")
    parser.add_argument("--data-root", type=str, default="train", help="COCO 图像与 json 目录")
    parser.add_argument("--out-dir", type=str, default="outputs", help="ckpt / logs / vis 输出根目录")
    parser.add_argument("--use-swanlab", action="store_true", help="启用 SwanLab 记录 loss")
    parser.add_argument("--swanlab-project", type=str, default="ir-translator-dino-mvp", help="SwanLab 项目名")
    parser.add_argument("--swanlab-run-name", type=str, default="overfit-10imgs", help="SwanLab 实验名")
    parser.add_argument(
        "--loss-mode",
        type=str,
        default="full",
        choices=["full", "main_only"],
        help="训练损失模式：full=主损失+encoder分支；main_only=仅主损失",
    )
    parser.add_argument("--enc-weight", type=float, default=0.1, help="encoder 分类损失权重")
    parser.add_argument("--enc-bbox-weight", type=float, default=0.5, help="encoder bbox 损失权重")
    parser.add_argument("--enc-giou-weight", type=float, default=0.5, help="encoder giou 损失权重")
    args = parser.parse_args()
    epochs = args.epochs
    data_root = args.data_root
    out_dir = args.out_dir

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("未检测到 CUDA，当前为 CPU 训练。")

    os.makedirs(os.path.join(out_dir, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "vis"), exist_ok=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer

    encoded = tokenizer(TEXT_PROMPT, return_tensors="pt")
    input_ids_base = encoded["input_ids"]
    attention_mask_base = encoded["attention_mask"]
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    print("Tokenizer 分词 (供核对):", tokens)

    dino = GroundingDinoForObjectDetection.from_pretrained(MODEL_ID).to(device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False

    cat_map = build_coco_category_to_class_index(TEXT_PROMPT)
    print("COCO category → 类别下标 (0=person, 1=car):", cat_map)

    dataset = FlirCocoOverfitDataset(data_root, processor, TEXT_PROMPT, cat_map)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    translator = ResidualTranslator().to(device)
    translator.train()
    optim = torch.optim.AdamW(translator.parameters(), lr=LEARNING_RATE)
    swan_run = _build_swanlab_logger(
        enable=args.use_swanlab,
        project=args.swanlab_project,
        run_name=args.swanlab_run_name,
        config={
            "model_id": MODEL_ID,
            "text_prompt": TEXT_PROMPT,
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "grad_clip": GRAD_CLIP,
            "data_root": data_root,
            "device": str(device),
            "loss_mode": args.loss_mode,
            "enc_weight": args.enc_weight,
            "enc_bbox_weight": args.enc_bbox_weight,
            "enc_giou_weight": args.enc_giou_weight,
        },
    )

    loss_history: List[float] = []
    global_step = 0
    grad_checked = False

    for epoch in range(epochs):
        epoch_losses: List[float] = []
        epoch_raw_total: List[float] = []
        epoch_main: List[float] = []
        epoch_ce: List[float] = []
        epoch_bbox: List[float] = []
        epoch_giou: List[float] = []
        epoch_ce_enc: List[float] = []
        epoch_bbox_enc: List[float] = []
        epoch_giou_enc: List[float] = []
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = _move_labels_to_device(batch["labels"], device)

            pseudo_rgb = translator(pixel_values)
            bsz = pixel_values.shape[0]
            input_ids = input_ids_base.expand(bsz, -1).to(device)
            attention_mask = attention_mask_base.expand(bsz, -1).to(device)

            outputs = dino(
                pixel_values=pseudo_rgb,
                pixel_mask=pixel_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            raw_loss = outputs.loss
            if raw_loss is None:
                raise RuntimeError("outputs.loss 为 None，请确认 transformers>=4.40 且 labels 非空。")
            train_loss, loss_scalars = _build_train_loss(
                outputs=outputs,
                mode=args.loss_mode,
                enc_weight=args.enc_weight,
                enc_bbox_weight=args.enc_bbox_weight,
                enc_giou_weight=args.enc_giou_weight,
            )

            optim.zero_grad(set_to_none=True)
            train_loss.backward()
            nn.utils.clip_grad_norm_(translator.parameters(), GRAD_CLIP)
            optim.step()

            if not grad_checked:
                tr_grad = sum(p.grad.abs().sum().item() for p in translator.parameters() if p.grad is not None)
                dino_grads = [p.grad for p in dino.parameters()]
                assert tr_grad > 0.0, "translator 梯度为 0，计算图可能断裂"
                assert all(g is None for g in dino_grads), "冻结的 DINO 不应有梯度"
                print(f"[梯度检查] translator grad L1 和: {tr_grad:.6f}；DINO 参数 grad 均为 None: OK")
                grad_checked = True

            epoch_losses.append(loss_scalars["loss_train"])
            epoch_raw_total.append(loss_scalars["loss_total_raw"])
            epoch_main.append(loss_scalars["loss_main"])
            epoch_ce.append(loss_scalars["loss_ce"])
            epoch_bbox.append(loss_scalars["loss_bbox"])
            epoch_giou.append(loss_scalars["loss_giou"])
            epoch_ce_enc.append(loss_scalars["loss_ce_enc"])
            epoch_bbox_enc.append(loss_scalars["loss_bbox_enc"])
            epoch_giou_enc.append(loss_scalars["loss_giou_enc"])
            global_step += 1
            if swan_run is not None:
                swan_run.log(
                    {
                        "train/loss_step_train": loss_scalars["loss_train"],
                        "train/loss_step_main": loss_scalars["loss_main"],
                        "train/loss_step_raw": loss_scalars["loss_total_raw"],
                        "train/loss_ce": loss_scalars["loss_ce"],
                        "train/loss_bbox": loss_scalars["loss_bbox"],
                        "train/loss_giou": loss_scalars["loss_giou"],
                        "train/loss_ce_enc": loss_scalars["loss_ce_enc"],
                        "train/loss_bbox_enc": loss_scalars["loss_bbox_enc"],
                        "train/loss_giou_enc": loss_scalars["loss_giou_enc"],
                        "train/global_step": global_step,
                    }
                )

        mean_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        mean_raw_total = float(sum(epoch_raw_total) / max(len(epoch_raw_total), 1))
        mean_main = float(sum(epoch_main) / max(len(epoch_main), 1))
        mean_ce = float(sum(epoch_ce) / max(len(epoch_ce), 1))
        mean_bbox = float(sum(epoch_bbox) / max(len(epoch_bbox), 1))
        mean_giou = float(sum(epoch_giou) / max(len(epoch_giou), 1))
        mean_ce_enc = float(sum(epoch_ce_enc) / max(len(epoch_ce_enc), 1))
        mean_bbox_enc = float(sum(epoch_bbox_enc) / max(len(epoch_bbox_enc), 1))
        mean_giou_enc = float(sum(epoch_giou_enc) / max(len(epoch_giou_enc), 1))
        loss_history.append(mean_loss)
        print(
            f"Epoch {epoch + 1}/{epochs}  mean_train_loss={mean_loss:.6f} "
            f"(main={mean_main:.4f}, ce={mean_ce:.4f}, bbox={mean_bbox:.4f}, "
            f"giou={mean_giou:.4f}, ce_enc={mean_ce_enc:.2f})"
        )
        if swan_run is not None:
            swan_run.log(
                {
                    "train/loss_epoch_train": mean_loss,
                    "train/loss_epoch_raw": mean_raw_total,
                    "train/loss_epoch_main": mean_main,
                    "train/loss_epoch_ce": mean_ce,
                    "train/loss_epoch_bbox": mean_bbox,
                    "train/loss_epoch_giou": mean_giou,
                    "train/loss_epoch_ce_enc": mean_ce_enc,
                    "train/loss_epoch_bbox_enc": mean_bbox_enc,
                    "train/loss_epoch_giou_enc": mean_giou_enc,
                    "train/epoch": epoch + 1,
                }
            )

        if epoch % VIS_EVERY == 0 or epoch == epochs - 1:
            # 保存中间权重
            mid_ckpt = os.path.join(out_dir, "ckpt", f"epoch_{epoch:04d}.pt")
            torch.save(translator.state_dict(), mid_ckpt)
            
            vis_path = os.path.join(out_dir, "vis", f"epoch_{epoch:04d}.png")
            samples = [dataset[i] for i in range(min(3, len(dataset)))]
            save_multi_sample_grid(
                translator,
                dino,
                processor,
                samples,
                TEXT_PROMPT,
                device,
                vis_path,
                box_threshold=0.3,
                text_threshold=0.25,
            )
            print(f"  已保存权重与可视化: {vis_path}")

    ckpt_path = os.path.join(out_dir, "ckpt", "translator_last.pt")
    torch.save(translator.state_dict(), ckpt_path)
    print(f"已保存权重: {ckpt_path}")

    loss_png = os.path.join(out_dir, "logs", "loss.png")
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label="mean loss / epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Overfit demo — loss curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_png, dpi=120)
    plt.close()
    print(f"已保存曲线: {loss_png}")

    if loss_history:
        init_l = loss_history[0]
        final_l = loss_history[-1]
        drop = (init_l - final_l) / max(init_l, 1e-8)
        print(f"Loss 初值≈{init_l:.4f} 末值≈{final_l:.4f} 相对下降≈{drop*100:.1f}%")
        if swan_run is not None:
            swan_run.log({"train/loss_drop_ratio": float(drop), "train/loss_final": float(final_l)})

    if swan_run is not None:
        try:
            swan_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
