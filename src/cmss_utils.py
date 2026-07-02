"""
CMSS 计算与 GMM 工具模块。

对应 docs/TD.md §1.3。核心 CMSS 相似度计算移植自：
  M-SpecGene-main/pretrain/.../GMM_CMSS_SAMPLE.py 第 85–97 行（CMSS_Similarity）。
去除 MAE 预训练耦合，适配批量 [B, L, D] 输入。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from src.config import CSMAConfig

# 默认均值——训练初期 GMM 尚未拟合时使用，避免 NoneType 报错
_DEFAULT_SORTED_MEANS: tuple[float, float, float] = (0.2, 0.5, 0.8)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: CMSS 相似度计算
# ──────────────────────────────────────────────────────────────────────────────
def compute_cmss(feat_rgb: torch.Tensor, feat_ir: torch.Tensor) -> torch.Tensor:
    """
    计算每个 Patch 的 CMSS 值，直接对应 GMM_CMSS_SAMPLE.py 第 85–97 行。

    CMSS = sqrt[(cosine_sim + 1) / 2] / (var_rgb * var_ir)，全局 max 归一化。
    - 低 CMSS（→0）：高方差 + 跨模态差异大 → 目标核心区域
    - 高 CMSS（→1）：低方差 + 跨模态一致   → 平滑背景区域

    Args:
        feat_rgb: [B, L, D]  冻结 DINO 对真实 RGB 提取的 Patch 特征。
        feat_ir:  [B, L, D]  冻结 DINO 对伪 RGB 提取的 Patch 特征。

    Returns:
        cmss_map: [B, L]，值域 [0, 1]，已全局 max 归一化。
    """
    assert feat_rgb.shape == feat_ir.shape, (
        f"feat_rgb 与 feat_ir 形状不一致：{feat_rgb.shape} vs {feat_ir.shape}"
    )
    assert feat_rgb.dim() == 3, "输入须为 [B, L, D]"

    # Phase 1.1：余弦相似度，映射至 [0, 1]
    norm_rgb = feat_rgb / feat_rgb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    norm_ir = feat_ir / feat_ir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    cos_sim = (norm_rgb * norm_ir).sum(dim=-1)          # [B, L]
    r = torch.sqrt((cos_sim + 1.0) * 0.5)               # sqrt[(cos+1)/2]

    # Phase 1.2：各 Patch 的特征方差（刻画结构复杂度）
    var_rgb = feat_rgb.var(dim=-1)                       # [B, L]
    var_ir = feat_ir.var(dim=-1)                         # [B, L]

    # Phase 1.3：CMSS 并全局 max 归一化
    cmss = r / (var_rgb * var_ir + 1e-6)                 # [B, L]
    cmss = cmss / cmss.max().clamp(min=1e-6)
    return cmss


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: GMM 拟合
# ──────────────────────────────────────────────────────────────────────────────
def fit_gmm(
    cmss_values: np.ndarray,
    n_components: int = 3,
) -> tuple[np.ndarray, GaussianMixture]:
    """
    对全数据集 CMSS 值拟合 GMM，返回排序后均值与模型。

    μ₁ < μ₂ < μ₃，分别对应目标核心、边缘过渡、背景。

    Args:
        cmss_values: 形状 [N] 的 1D float32 数组，N = 样本数 × Patch 数。
        n_components: GMM 组件数，默认 3。

    Returns:
        sorted_means: [n_components] 升序排列的均值数组。
        gmm: 拟合好的 GaussianMixture 对象，供 Stage B 采样使用。
    """
    assert cmss_values.ndim == 1, "cmss_values 须为 1D 数组"
    assert len(cmss_values) >= n_components, (
        f"样本数 {len(cmss_values)} 不足以拟合 {n_components} 个组件"
    )

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42,
    )
    gmm.fit(cmss_values.reshape(-1, 1))
    sorted_means = np.sort(gmm.means_.flatten())
    return sorted_means, gmm


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: 掩码生成
# ──────────────────────────────────────────────────────────────────────────────
def build_cmss_mask(
    cmss_map: torch.Tensor,
    stage: int,
    mu1: float,
    mu2: float,
    mu3: float,  # noqa: ARG001  保留接口完整性，Stage A/C 当前未直接使用 mu3
    mask_ratio: float = 0.75,
    gmm: Optional[GaussianMixture] = None,
) -> torch.Tensor:
    """
    根据训练阶段和 GMM 均值生成 Patch 级二值掩码。

    mask=1 表示该 Patch 被掩蔽（不参与 L_align 计算）；mask=0 表示保留。

    Args:
        cmss_map:   [B, L]，compute_cmss 的输出。
        stage:      0=A（Easy），1=B（Mixed），2=C（Hard）。
        mu1:        GMM 第一均值，对应目标核心区域阈值。
        mu2:        GMM 第二均值，对应边缘过渡区域阈值。
        mu3:        GMM 第三均值，接口保留供将来扩展使用。
        mask_ratio: Stage B 的掩码比例，默认 0.75。
        gmm:        Stage B 所用 GaussianMixture 对象；Stage A/C 可为 None。

    Returns:
        mask: [B, L]，dtype=float32，值 0（保留）或 1（掩蔽）。
    """
    assert cmss_map.dim() == 2, "cmss_map 须为 [B, L]"
    B, L = cmss_map.shape
    device = cmss_map.device

    # Phase 3.1：阶段 A — 掩蔽背景（高 CMSS > μ₂），保留目标核心
    if stage == 0:
        return (cmss_map > mu2).float()

    # Phase 3.2：阶段 B — 按 GMM 概率分布随机采样掩码（TD §1.3 简化版）
    if stage == 1:
        if gmm is None:
            raise ValueError("Stage B 需要传入已拟合的 gmm 对象")
        noise_np = gmm.sample(B * L)[0].reshape(B, L).astype(np.float32)
        noise = torch.from_numpy(noise_np).to(device)
        ids = torch.argsort(noise, dim=-1)
        len_keep = int(L * (1.0 - mask_ratio))
        mask = torch.ones(B, L, device=device, dtype=torch.float32)
        mask.scatter_(1, ids[:, :len_keep], 0.0)
        return mask

    # Phase 3.3：阶段 C — 掩蔽目标核心（低 CMSS < μ₁），保留背景
    return (cmss_map < mu1).float()


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: 训练课程调度器
# ──────────────────────────────────────────────────────────────────────────────
class CMSSScheduler:
    """
    管理三阶段 GMM-CMSS 渐进课程。

    对应 M-SpecGene GMM_CMSS_SAMPLE 中 maskratio_bias / sample_range 动态调整逻辑的
    显式化版本，将连续动态调整离散化为三个阶段，降低实现复杂度（MVP 原则）。

    阶段边界（默认）：
        A（Easy）:  epoch ∈ [0,  T/3)
        B（Mixed）: epoch ∈ [T/3, 2T/3)
        C（Hard）:  epoch ∈ [2T/3, T)
    若 cfg.hard_max_epochs=N：Hard 仅占最后 N 个 epoch（缩短 Hard，保护伪 RGB）。
    """

    def __init__(self, cfg: CSMAConfig) -> None:
        self._cfg = cfg
        t = cfg.total_epochs
        if cfg.skip_hard_stage:
            # Easy + Mixed 覆盖全部 epoch，永不进入 Hard
            b0 = max(1, t // 3)
            self._stage_boundaries = [b0, t]
        elif cfg.stage_epoch_boundaries is not None:
            self._stage_boundaries = list(cfg.stage_epoch_boundaries)
        elif cfg.hard_max_epochs is not None:
            # Hard 仅最后 hard_max_epochs 轮；Easy 仍约 T/3，其余为 Mixed
            b0 = t // 3
            b1 = max(b0 + 1, t - cfg.hard_max_epochs)
            self._stage_boundaries = [b0, b1]
        else:
            self._stage_boundaries = [t // 3, t * 2 // 3]
        self._gmm_update_every: int = cfg.gmm_update_every
        self._gmm: Optional[GaussianMixture] = None
        self._sorted_means: Optional[np.ndarray] = None

    @property
    def stage_boundaries(self) -> tuple[int, int]:
        """(easy_end, mixed_end) 均为 0-based 下一阶段的起始 epoch。"""
        return self._stage_boundaries[0], self._stage_boundaries[1]

    def stage1_last_epoch(self) -> int:
        """Mixed 阶段最后一个 epoch（0-based），即进入 Hard 的前一轮。"""
        return self._stage_boundaries[1] - 1

    # ── 阶段查询 ──────────────────────────────────────────────────────────────

    def get_stage(self, epoch: int) -> int:
        """
        返回当前 epoch 所处的课程阶段编号。

        Args:
            epoch: 当前训练轮次（从 0 计数）。

        Returns:
            0（A/Easy）、1（B/Mixed）或 2（C/Hard）。
        """
        if epoch < self._stage_boundaries[0]:
            return 0
        if epoch < self._stage_boundaries[1]:
            return 1
        return 2

    # ── GMM 管理 ──────────────────────────────────────────────────────────────

    def should_update_gmm(self, epoch: int) -> bool:
        """
        判断当前 epoch 是否需要重新拟合 GMM。

        Args:
            epoch: 当前训练轮次。

        Returns:
            True 表示应在此 epoch 调用 update_gmm。
        """
        return epoch % self._gmm_update_every == 0

    def update_gmm(self, cmss_values: np.ndarray) -> None:
        """
        用本轮全量 CMSS 值重新拟合 GMM，更新内部 sorted_means。

        Args:
            cmss_values: [N] 1D float32 数组，来自遍历全量训练集的 compute_cmss 输出。
        """
        self._sorted_means, self._gmm = fit_gmm(
            cmss_values, n_components=self._cfg.gmm_n_components
        )

    @property
    def gmm(self) -> Optional[GaussianMixture]:
        """已拟合的 GaussianMixture 对象；GMM 尚未初始化时为 None。"""
        return self._gmm

    @property
    def sorted_means(self) -> tuple[float, float, float]:
        """
        返回 GMM 三个排序均值 (μ₁, μ₂, μ₃)。
        GMM 尚未拟合时返回安全默认值 (0.2, 0.5, 0.8)，避免训练初期崩溃。
        """
        if self._sorted_means is None:
            return _DEFAULT_SORTED_MEANS
        m = self._sorted_means
        return (float(m[0]), float(m[1]), float(m[2]))

    # ── 损失权重 ─────────────────────────────────────────────────────────────

    def get_loss_weights(self, epoch: int) -> tuple[float, float]:
        """
        返回当前 epoch 对应的 (lambda_align, lambda_det) 损失权重对。

        权重来自 CSMAConfig.stage_loss_weights，三阶段各对应一组：
            A: (1.0, 0.1)  特征对齐为主
            B: (0.5, 0.5)  对齐与检测并重
            C: (0.1, 1.0)  检测精度为主

        Args:
            epoch: 当前训练轮次。

        Returns:
            (lambda_align, lambda_det)。
        """
        stage = self.get_stage(epoch)
        la, ld = self._cfg.stage_loss_weights[stage]
        return float(la), float(ld)
