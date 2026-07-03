from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from src.config import CSMAConfig

_DEFAULT_SORTED_MEANS = (0.2, 0.5, 0.8)


def compute_cmss(feat_rgb: torch.Tensor, feat_ir: torch.Tensor) -> torch.Tensor:
    norm_rgb = feat_rgb / feat_rgb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    norm_ir = feat_ir / feat_ir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    cos_sim = (norm_rgb * norm_ir).sum(dim=-1)
    r = torch.sqrt((cos_sim + 1.0) * 0.5)
    var_rgb = feat_rgb.var(dim=-1)
    var_ir = feat_ir.var(dim=-1)
    cmss = r / (var_rgb * var_ir + 1e-6)
    return cmss / cmss.max().clamp(min=1e-6)


def fit_gmm(cmss_values: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, GaussianMixture]:
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42)
    gmm.fit(cmss_values.reshape(-1, 1))
    return np.sort(gmm.means_.flatten()), gmm


def build_cmss_mask(
    cmss_map: torch.Tensor, stage: int, mu1: float, mu2: float, mu3: float,
    mask_ratio: float = 0.75, gmm: Optional[GaussianMixture] = None,
) -> torch.Tensor:
    del mu3
    B, L = cmss_map.shape
    device = cmss_map.device
    if stage == 0:
        return (cmss_map > mu2).float()
    if stage == 1:
        if gmm is None:
            raise ValueError("Stage B requires gmm")
        noise = torch.from_numpy(gmm.sample(B * L)[0].reshape(B, L).astype(np.float32)).to(device)
        ids = torch.argsort(noise, dim=-1)
        len_keep = int(L * (1.0 - mask_ratio))
        mask = torch.ones(B, L, device=device)
        mask.scatter_(1, ids[:, :len_keep], 0.0)
        return mask
    return (cmss_map < mu1).float()


class CMSSScheduler:
    def __init__(self, cfg: CSMAConfig) -> None:
        self._cfg = cfg
        t = cfg.total_epochs
        if cfg.stage_epoch_boundaries is not None:
            self._stage_boundaries = list(cfg.stage_epoch_boundaries)
        elif cfg.hard_max_epochs is not None:
            b0 = t // 3
            b1 = max(b0 + 1, t - cfg.hard_max_epochs)
            self._stage_boundaries = [b0, b1]
        else:
            self._stage_boundaries = [t // 3, t * 2 // 3]
        self._gmm: Optional[GaussianMixture] = None
        self._sorted_means: Optional[np.ndarray] = None

    @property
    def stage_boundaries(self) -> tuple[int, int]:
        return self._stage_boundaries[0], self._stage_boundaries[1]

    def get_stage(self, epoch: int) -> int:
        if epoch < self._stage_boundaries[0]:
            return 0
        if epoch < self._stage_boundaries[1]:
            return 1
        return 2

    def should_update_gmm(self, epoch: int) -> bool:
        return epoch % self._cfg.gmm_update_every == 0

    def update_gmm(self, cmss_values: np.ndarray) -> None:
        self._sorted_means, self._gmm = fit_gmm(cmss_values, self._cfg.gmm_n_components)

    @property
    def gmm(self) -> Optional[GaussianMixture]:
        return self._gmm

    @property
    def sorted_means(self) -> tuple[float, float, float]:
        if self._sorted_means is None:
            return _DEFAULT_SORTED_MEANS
        m = self._sorted_means
        return float(m[0]), float(m[1]), float(m[2])

    def get_loss_weights(self, epoch: int) -> tuple[float, float]:
        return self._cfg.stage_loss_weights[self.get_stage(epoch)]
