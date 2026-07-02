#!/usr/bin/env python3
"""
CSMA 生产 CLI 入口。

子命令：build-knowledge | train | eval | answer
用法：python main.py [--config path] [--device dev] [--override key=value ...] <子命令> [选项]

对应 AGENTS.md §2.3。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# 保证从项目根目录可 import src
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _parse_override(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"override 需为 key=value 格式: {s!r}")
    key, raw = s.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    if raw.lower() in ("true", "false"):
        return key, raw.lower() == "true"
    try:
        if "." in raw:
            return key, float(raw)
        return key, int(raw)
    except ValueError:
        return key, raw


def _load_yaml_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError("使用 --config 需安装 PyYAML: pip install pyyaml") from e
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML 根节点必须是 mapping")
    return data


def _apply_device(device: str | None) -> None:
    if not device:
        return
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return
    if device.startswith("cuda"):
        idx = device.split(":")[-1] if ":" in device else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = idx


def _merge_overrides(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out.update(extra)
    return out


def _train_preset_argv(phase: int, overrides: dict[str, Any]) -> list[str]:
    """按训练阶段返回传给 src.train_csma 的 CLI 参数。"""
    data_root = overrides.get("ir_data_root", "FLIR_License/train")
    out_dir = overrides.get("output_dir", "outputs_csma")
    argv = [
        "--dataset",
        str(overrides.get("dataset", "flir_v1")),
        "--data-root",
        str(data_root),
        "--out-dir",
        str(out_dir),
        "--gmm-batches",
        str(overrides.get("gmm_max_batches", 100)),
    ]
    if phase == 0:
        argv += [
            "--epochs",
            str(overrides.get("total_epochs", 35)),
            "--batch-size",
            str(overrides.get("batch_size", 2)),
            "--loss-mode",
            str(overrides.get("loss_mode", "full")),
            "--hard-max-epochs",
            str(overrides.get("hard_max_epochs", 5)),
        ]
    elif phase == 1:
        argv += [
            "--epochs",
            str(overrides.get("total_epochs", 30)),
            "--batch-size",
            str(overrides.get("batch_size", 2)),
            "--loss-mode",
            "full",
            "--warmup-epochs",
            str(overrides.get("warmup_epochs", 3)),
            "--stage-weights",
            str(
                overrides.get(
                    "stage_weights",
                    "1.0,0.1;0.8,0.2;0.5,0.5",
                )
            ),
            "--hard-max-epochs",
            "0",
            "--stop-after-stage1",
            "--val-early-stop",
            "--val-data-root",
            str(overrides.get("val_data_root", "FLIR_License/val")),
        ]
    elif phase == 2:
        ckpt = overrides.get("init_ckpt")
        if not ckpt:
            raise ValueError("train --phase 2 续训需在 --override 中提供 init_ckpt=路径")
        start = overrides.get("start_epoch", 0)
        argv += [
            "--init-ckpt",
            str(ckpt),
            "--start-epoch",
            str(start),
            "--epochs",
            str(overrides.get("total_epochs", 35)),
            "--batch-size",
            str(overrides.get("batch_size", 2)),
            "--loss-mode",
            str(overrides.get("loss_mode", "full")),
        ]
    elif phase == 3:
        argv += [
            "--epochs",
            str(overrides.get("total_epochs", 20)),
            "--batch-size",
            str(overrides.get("batch_size", 2)),
            "--loss-mode",
            "det_only",
        ]
    else:
        raise ValueError(f"未知训练阶段 phase={phase}，可选 0–3")
    return argv


def _run_module_main(module_argv: list[str], module: str) -> None:
    sys.argv = [module] + module_argv
    if module == "src.train_csma":
        from src.train_csma import main as entry
    elif module == "src.eval_csma":
        from src.eval_csma import main as entry
    elif module == "src.infer_csma":
        from src.infer_csma import main as entry
    else:
        raise ValueError(module)
    entry()


def cmd_build_knowledge(overrides: dict[str, Any], gmm_batches: int) -> None:
    """在训练前拟合初始 GMM（CMSS 统计），写入 output_dir/knowledge/。"""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    from src.cmss_utils import CMSSScheduler
    from src.config import CSMAConfig
    from src.csma import CSMA
    from src.dataset_flir_v1 import FlirV1PairedDataset, build_flir_v1_category_map, collate_flir_v1
    from src.train_csma import collect_cmss_values

    cfg = CSMAConfig.from_overrides(overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_knowledge = Path(cfg.output_dir) / "knowledge"
    out_knowledge.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(cfg.model_id, local_files_only=True)
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
        ip = processor.image_processor
        try:
            ip.size.shortest_edge = cfg.img_size
            ip.size.longest_edge = cfg.img_size * 2
        except AttributeError:
            ip.size = {"shortest_edge": cfg.img_size, "longest_edge": cfg.img_size * 2}

    dino = GroundingDinoForObjectDetection.from_pretrained(cfg.model_id, local_files_only=True).to(device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False

    csma = CSMA(cfg).to(device)
    init_ckpt = overrides.get("init_ckpt")
    if init_ckpt:
        raw = torch.load(str(init_ckpt), map_location=device, weights_only=True)
        state = raw["csma"] if isinstance(raw, dict) and "csma" in raw else raw
        csma.load_state_dict(state)
        print(f"[build-knowledge] 已加载 CSMA: {init_ckpt}")

    encoded = processor.tokenizer(cfg.text_prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    cat_map, valid_ids = build_flir_v1_category_map(cfg.text_prompt)
    dataset = FlirV1PairedDataset(
        root=cfg.ir_data_root,
        processor=processor,
        text_prompt=cfg.text_prompt,
        category_map=cat_map,
        valid_cat_ids=valid_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_flir_v1,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"[build-knowledge] 数据集 {len(dataset)} 张，采样至多 {gmm_batches} batch")

    cmss_vals = collect_cmss_values(
        dino, csma, loader, device, input_ids, attention_mask, max_batches=gmm_batches
    )
    sched = CMSSScheduler(cfg)
    sched.update_gmm(cmss_vals)
    mu1, mu2, mu3 = sched.sorted_means
    payload = {
        "sorted_means": [mu1, mu2, mu3],
        "n_samples": int(len(cmss_vals)),
        "gmm_batches": gmm_batches,
        "data_root": cfg.ir_data_root,
    }
    out_path = out_knowledge / "gmm_means.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[build-knowledge] GMM 均值 μ₁={mu1:.4f} μ₂={mu2:.4f} μ₃={mu3:.4f}")
    print(f"[build-knowledge] 已保存: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-Spectral Modality Adapter 生产 CLI",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="可选 YAML 配置（键与 CSMAConfig 字段一致）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备，如 cuda:0 或 cpu；会设置 CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖 CSMAConfig 字段，可重复",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_bk = sub.add_parser("build-knowledge", help="Phase 0：拟合初始 GMM-CMSS 统计")
    p_bk.add_argument(
        "--gmm-batches",
        type=int,
        default=100,
        help="CMSS 采样 batch 上限（-1=全量）",
    )

    p_train = sub.add_parser("train", help="训练 CSMA + 冻结 DINO")
    p_train.add_argument(
        "--phase",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="0=标准全量 1=优化课程+val早停 2=续训 3=det_only消融",
    )
    p_train.add_argument("train_extra", nargs=argparse.REMAINDER, help="透传给 train_csma 的额外参数")

    p_eval = sub.add_parser("eval", help="mAP 评测")
    p_eval.add_argument("--ckpt", type=str, required=True)
    p_eval.add_argument("--data-root", type=str, default="FLIR_License/val")
    p_eval.add_argument("--dataset", type=str, default="flir_v1", choices=["flir_v1", "flir_v2", "llvip"])
    p_eval.add_argument("--out-json", type=str, default="outputs_csma/logs/eval_result.json")
    p_eval.add_argument("--batch-size", type=int, default=4)

    p_ans = sub.add_parser("answer", help="端到端推理（检测 + 可视化）")
    p_ans.add_argument("--ckpt", type=str, required=True)
    p_ans.add_argument("--data-root", type=str, default="FLIR_License/val")
    p_ans.add_argument("--dataset", type=str, default="flir_v1", choices=["legacy", "flir_v1", "flir_v2"])
    p_ans.add_argument("--out", type=str, default="outputs_csma/vis/infer_grid.png")
    p_ans.add_argument("--num-samples", type=int, default=5)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    _apply_device(args.device)

    cfg_overrides: dict[str, Any] = {}
    if args.config:
        cfg_overrides.update(_load_yaml_config(args.config))
    for item in args.override:
        k, v = _parse_override(item)
        cfg_overrides[k] = v

    if args.command == "build-knowledge":
        cmd_build_knowledge(cfg_overrides, args.gmm_batches)
        return

    if args.command == "train":
        train_argv = _train_preset_argv(args.phase, cfg_overrides)
        train_argv.extend(args.train_extra)
        _run_module_main(train_argv, "src.train_csma")
        return

    if args.command == "eval":
        eval_argv = [
            "--ckpt",
            args.ckpt,
            "--dataset",
            args.dataset,
            "--data-root",
            args.data_root,
            "--out-json",
            args.out_json,
            "--batch-size",
            str(args.batch_size),
        ]
        _run_module_main(eval_argv, "src.eval_csma")
        return

    if args.command == "answer":
        infer_argv = [
            "--ckpt",
            args.ckpt,
            "--dataset",
            args.dataset,
            "--data-root",
            args.data_root,
            "--out",
            args.out,
            "--num-samples",
            str(args.num_samples),
        ]
        _run_module_main(infer_argv, "src.infer_csma")
        return

    parser.error(f"未知子命令: {args.command}")


if __name__ == "__main__":
    main()
