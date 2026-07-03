"""
消融实验 Smoke Test。

验证 Ablation B-2 / C-1 / C-2 / C-3 四个变体在 2 epoch × 20 step 内能正常启动、
不报错、不产生 NaN loss，并生成 ckpt/latest.pt。

测试规范（对应 docs/实验实施细节.md §3.2）：
  - 运行命令: CUDA_VISIBLE_DEVICES=0 conda run -n RGBtest python -m pytest \
                tests/integration/test_ablation_smoke.py -v
  - 每个测试结果写入 tests/outputs/ablation/<name>_<timestamp>.md

前置条件：
  - FLIR_License/train 数据集已就位。
  - 对 B-2 变体，需要先生成均值原型（本测试会在 smoke 前自动生成一个随机 mock .pt）。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pytest
import torch

# 项目根目录（tests/integration 的两层上级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "tests" / "outputs" / "ablation"
DATA_ROOT    = PROJECT_ROOT / "FLIR_License" / "train"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _write_report(name: str, cmd: List[str], returncode: int, stdout: str, stderr: str) -> Path:
    """将 smoke test 结果写入 Markdown 报告文件。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUTS_DIR / f"{name}_{timestamp}.md"
    status = "PASS" if returncode == 0 else "FAIL"
    content = (
        f"# Ablation Smoke Test: {name}\n\n"
        f"## 任务\n"
        f"验证消融变体 `{name}` 在 2 epoch × 20 step 内正常运行\n\n"
        f"## 运行命令\n"
        f"```bash\n{' '.join(cmd)}\n```\n\n"
        f"## 结果\n"
        f"**状态**: {status}  \n"
        f"**退出码**: {returncode}\n\n"
        f"## stdout\n"
        f"```\n{stdout[-3000:] if len(stdout) > 3000 else stdout}\n```\n\n"
        f"## stderr\n"
        f"```\n{stderr[-2000:] if len(stderr) > 2000 else stderr}\n```\n"
    )
    report_path.write_text(content, encoding="utf-8")
    return report_path


def _assert_no_gmm_fallback(stdout: str, name: str) -> None:
    """
    确认 gmm_single_b 变体在训练日志中没有触发 GMM fallback。

    全量 C-3 训练完成后也应执行相同检查：
      grep "gmm_single_b fallback" outputs_ablation_c/c3_gmm_single_b/logs/train.log
    若出现任何一行，需停下调查——表明 GMM 未正常拟合，C-3 实际跑的是 fixed threshold。
    """
    fallback_keyword = "gmm_single_b fallback to fixed threshold"
    count = stdout.count(fallback_keyword)
    assert count == 0, (
        f"[{name}] 检测到 {count} 次 GMM fallback！\n"
        f"C-3 本应全程使用 GMM Stage B，fallback 表明 GMM 未正常拟合。\n"
        f"请检查 collect_cmss_values 是否收集到足够 patch（>= gmm_n_components=3）。"
    )


def _assert_no_nan_in_log(stdout: str, name: str) -> None:
    """
    在训练日志中显式检测 NaN / Inf。

    训练脚本在每个 epoch 末打印 loss=<value>，提取并验证。
    同时扫描整个 stdout 确认无 'nan' / 'inf' 字符串（不区分大小写）。
    """
    import re
    # 提取 loss= 后的浮点数（格式如 loss=285.0333）
    loss_vals = [float(m) for m in re.findall(r"loss=([0-9.eE+\-]+)", stdout)]
    for val in loss_vals:
        assert not (val != val), f"[{name}] 检测到 NaN loss: {val}"     # NaN != NaN
        assert val != float("inf"), f"[{name}] 检测到 Inf loss: {val}"
        assert val != float("-inf"), f"[{name}] 检测到 -Inf loss: {val}"

    # 全文扫描（不区分大小写），排除正常的浮点数格式如 2.3e-4
    lower = stdout.lower()
    # 匹配独立的 nan/inf 单词（前后非数字/字母，避免误匹配 "nan" 在其他单词中）
    nan_hits = re.findall(r"(?<![a-z0-9])nan(?![a-z0-9])", lower)
    inf_hits = re.findall(r"(?<![a-z0-9])inf(?![a-z0-9])", lower)
    assert not nan_hits, f"[{name}] stdout 中检测到 'nan'（{len(nan_hits)} 处）"
    assert not inf_hits, f"[{name}] stdout 中检测到 'inf'（{len(inf_hits)} 处）"

    if loss_vals:
        print(f"  [NaN检查] {name}: {len(loss_vals)} 个 loss 值，均有限。"
              f" 范围=[{min(loss_vals):.4f}, {max(loss_vals):.4f}]")
    else:
        print(f"  [NaN检查] {name}: 未找到 loss= 数值，已做全文扫描无 nan/inf")


def _run_smoke(
    name: str,
    extra_args: List[str],
    out_dir: Path,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """
    执行 2 epoch × 20 step 的 smoke train，返回 CompletedProcess。

    Args:
        name:       消融变体名称（仅用于日志/报告）。
        extra_args: 消融变体专属的额外 CLI 参数列表。
        out_dir:    临时输出目录（自动创建）。
        timeout:    超时秒数（默认 600s = 10 分钟）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        sys.executable, "-m", "src.train_csma",
        "--dataset",    "flir_v1",
        "--data-root",  str(DATA_ROOT),
        "--out-dir",    str(out_dir),
        "--epochs",     "2",
        "--max-steps",  "20",
        "--batch-size", "2",
    ]
    cmd = base_cmd + extra_args

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # 指向本地 HuggingFace 缓存目录（避免联网下载）
    env.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return result, cmd


# ──────────────────────────────────────────────────────────────────────────────
# 测试用例
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not DATA_ROOT.exists(), reason="FLIR_License/train 数据集不存在，跳过 smoke test")
def test_ablation_c1_random_mask(tmp_path: Path) -> None:
    """
    Ablation C-1: 随机掩码 + 固定 λ=0.5/0.5。

    验证 --cmss-ablation-mode random_mask 分支可正常运行 2 epoch × 20 step。
    """
    name = "c1_random_mask"
    out_dir = tmp_path / name
    result, cmd = _run_smoke(
        name=name,
        extra_args=["--cmss-ablation-mode", "random_mask"],
        out_dir=out_dir,
    )
    report_path = _write_report(name, cmd, result.returncode, result.stdout, result.stderr)
    print(f"\n[smoke] 报告: {report_path}")

    assert result.returncode == 0, (
        f"Ablation C-1 smoke test 失败（exit={result.returncode}）\n"
        f"stderr:\n{result.stderr[-1500:]}"
    )
    assert (out_dir / "ckpt" / "latest.pt").exists(), "ckpt/latest.pt 不存在"
    _assert_no_nan_in_log(result.stdout, name)


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="FLIR_License/train 数据集不存在，跳过 smoke test")
def test_ablation_c2_fixed_threshold(tmp_path: Path) -> None:
    """
    Ablation C-2: 固定 CMSS 阈值 μ₂=0.5 + 固定 λ=0.5/0.5。

    验证 --cmss-ablation-mode fixed_threshold 分支可正常运行 2 epoch × 20 step。
    """
    name = "c2_fixed_threshold"
    out_dir = tmp_path / name
    result, cmd = _run_smoke(
        name=name,
        extra_args=["--cmss-ablation-mode", "fixed_threshold"],
        out_dir=out_dir,
    )
    report_path = _write_report(name, cmd, result.returncode, result.stdout, result.stderr)
    print(f"\n[smoke] 报告: {report_path}")

    assert result.returncode == 0, (
        f"Ablation C-2 smoke test 失败（exit={result.returncode}）\n"
        f"stderr:\n{result.stderr[-1500:]}"
    )
    assert (out_dir / "ckpt" / "latest.pt").exists(), "ckpt/latest.pt 不存在"
    _assert_no_nan_in_log(result.stdout, name)


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="FLIR_License/train 数据集不存在，跳过 smoke test")
def test_ablation_c3_gmm_single_b(tmp_path: Path) -> None:
    """
    Ablation C-3: GMM 动态 + 固定 Stage B + 固定 λ=0.5/0.5。

    验证 --cmss-ablation-mode gmm_single_b 分支可正常运行 2 epoch × 20 step。
    注意：Stage B 需要 GMM，但 2 epoch 的 smoke test 会在 epoch 0 触发 GMM 拟合。
    """
    name = "c3_gmm_single_b"
    out_dir = tmp_path / name
    result, cmd = _run_smoke(
        name=name,
        extra_args=["--cmss-ablation-mode", "gmm_single_b"],
        out_dir=out_dir,
    )
    report_path = _write_report(name, cmd, result.returncode, result.stdout, result.stderr)
    print(f"\n[smoke] 报告: {report_path}")

    assert result.returncode == 0, (
        f"Ablation C-3 smoke test 失败（exit={result.returncode}）\n"
        f"stderr:\n{result.stderr[-1500:]}"
    )
    assert (out_dir / "ckpt" / "latest.pt").exists(), "ckpt/latest.pt 不存在"
    _assert_no_nan_in_log(result.stdout, name)
    _assert_no_gmm_fallback(result.stdout, name)


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="FLIR_License/train 数据集不存在，跳过 smoke test")
def test_ablation_b2_mean_proto(tmp_path: Path) -> None:
    """
    Ablation B-2: 均值原型（Frozen Mean）替代可学习原型库。

    流程：
      1. 生成随机 mock mean_proto.pt（避免运行完整 compute_mean_proto.py）
      2. 验证 --variant mean_proto 分支可正常运行 2 epoch × 20 step
    """
    name = "b2_mean_proto"
    out_dir = tmp_path / name

    # 生成随机 mock mean_proto（proto_dim=256，与 CSMAConfig 默认值一致）
    mock_proto_path = tmp_path / "mean_proto_mock.pt"
    mock_proto = torch.randn(256)
    torch.save(mock_proto, str(mock_proto_path))

    result, cmd = _run_smoke(
        name=name,
        extra_args=[
            "--variant", "mean_proto",
            "--mean-proto-path", str(mock_proto_path),
        ],
        out_dir=out_dir,
    )
    report_path = _write_report(name, cmd, result.returncode, result.stdout, result.stderr)
    print(f"\n[smoke] 报告: {report_path}")

    assert result.returncode == 0, (
        f"Ablation B-2 smoke test 失败（exit={result.returncode}）\n"
        f"stderr:\n{result.stderr[-1500:]}"
    )
    assert (out_dir / "ckpt" / "latest.pt").exists(), "ckpt/latest.pt 不存在"
    _assert_no_nan_in_log(result.stdout, name)


# ──────────────────────────────────────────────────────────────────────────────
# 新数据集 smoke test（M3FD / LLVIP）
# ──────────────────────────────────────────────────────────────────────────────

# 数据集路径常量
LLVIP_ROOT      = PROJECT_ROOT / "LLVIP"
LLVIP_TRAIN_ANN = LLVIP_ROOT / "infrared" / "train"   # 原生 XML，不再依赖 JSON

# M3FD 候选路径（按优先级尝试）
_M3FD_CANDIDATES = [
    Path("/root/autodl-tmp/M3FD/train"),                                       # 当前实际位置
    PROJECT_ROOT / "M3FD" / "train",                                           # 项目内备用
    Path("/root/autodl-tmp/Cross-Spectral-Modality-Adapter/M3FD/train"),       # 旧副本备用
]
M3FD_ROOT = next((p for p in _M3FD_CANDIDATES if p.exists()), None)


def _run_dataset_smoke(
    name: str,
    dataset: str,
    data_root: "Path",
    out_dir: "Path",
    extra_args: "List[str] | None" = None,
    timeout: int = 600,
) -> "tuple":
    """
    针对新数据集的灵活 smoke runner（数据集类型和路径可自由指定）。

    2 epoch × 20 step，环境变量与 _run_smoke 完全一致。

    Args:
        name:       测试名称（仅用于日志）。
        dataset:    --dataset 参数值（如 "llvip" / "m3fd"）。
        data_root:  --data-root 路径。
        out_dir:    输出目录。
        extra_args: 额外 CLI 参数（如 --llvip-split train）。
        timeout:    超时秒数。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.train_csma",
        "--dataset",    dataset,
        "--data-root",  str(data_root),
        "--out-dir",    str(out_dir),
        "--epochs",     "2",
        "--max-steps",  "20",
        "--batch-size", "2",
    ] + (extra_args or [])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]   = "0"
    env["HF_HUB_OFFLINE"]         = "1"
    env["TRANSFORMERS_OFFLINE"]   = "1"
    env.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
    # 新数据集图像量大（LLVIP 12k / M3FD 多分辨率），DataLoader 多进程在容器内
    # 容易触发 SIGSEGV；禁用 OpenBLAS/OMP 多线程并强制单进程 DataLoader
    env["OPENBLAS_NUM_THREADS"]   = "1"
    env["OMP_NUM_THREADS"]        = "1"

    result = subprocess.run(
        cmd + ["--num-workers", "0"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return result, cmd


@pytest.mark.skipif(
    not LLVIP_TRAIN_ANN.exists(),
    reason="LLVIP/infrared/train/ 不存在，请确认 LLVIP 数据集已就位",
)
def test_dataset_llvip(tmp_path: Path) -> None:
    """
    LLVIP 数据集 smoke test：2 epoch × 20 step。

    验证：
      - llvip 数据集分支（infrared/train + visible/train + train_coco.json）正常加载
      - loss 无 NaN / Inf
      - ckpt/latest.pt 生成
    """
    name    = "llvip_smoke"
    out_dir = tmp_path / name

    result, cmd = _run_dataset_smoke(
        name=name,
        dataset="llvip",
        data_root=LLVIP_ROOT,
        out_dir=out_dir,
        extra_args=["--llvip-split", "train"],
    )
    report_path = _write_report(name, cmd, result.returncode, result.stdout, result.stderr)
    print(f"\n[smoke] 报告: {report_path}")

    assert result.returncode == 0, (
        f"LLVIP smoke test 失败（exit={result.returncode}）\n"
        f"stderr:\n{result.stderr[-1500:]}"
    )
    assert (out_dir / "ckpt" / "latest.pt").exists(), "ckpt/latest.pt 不存在"
    _assert_no_nan_in_log(result.stdout, name)


@pytest.mark.skipif(
    M3FD_ROOT is None,
    reason="M3FD/train 目录不存在（已检查所有候选路径），请确认数据已就绪",
)
def test_dataset_m3fd(tmp_path: Path) -> None:
    """
    M3FD 数据集 smoke test：2 epoch × 20 step。

    验证：
      - m3fd 数据集分支（ir/ + vi/ + annotations_coco.json）正常加载
      - loss 无 NaN / Inf
      - ckpt/latest.pt 生成

    前置条件：
      bash scripts/08_prepare_m3fd.sh   （解压 + YOLO→COCO 转换）
    """
    name    = "m3fd_smoke"
    out_dir = tmp_path / name

    result, cmd = _run_dataset_smoke(
        name=name,
        dataset="m3fd",
        data_root=M3FD_ROOT,  # type: ignore[arg-type]
        out_dir=out_dir,
    )
    report_path = _write_report(name, cmd, result.returncode, result.stdout, result.stderr)
    print(f"\n[smoke] 报告: {report_path}")

    assert result.returncode == 0, (
        f"M3FD smoke test 失败（exit={result.returncode}）\n"
        f"stderr:\n{result.stderr[-1500:]}"
    )
    assert (out_dir / "ckpt" / "latest.pt").exists(), "ckpt/latest.pt 不存在"
    _assert_no_nan_in_log(result.stdout, name)
