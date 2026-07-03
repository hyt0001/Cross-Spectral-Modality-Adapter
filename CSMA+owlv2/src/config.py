from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, List, Literal, Tuple


LossMode = Literal["align_only", "det_only", "full"]


@dataclass
class CSMAConfig:
    ir_enc_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    num_rgb_prototypes: int = 512
    proto_dim: int = 256
    num_cross_attn_heads: int = 8
    use_residual: bool = True
    residual_scale: float = 0.1
    pseudo_rgb_clamp: float = 3.0
    use_group_norm: bool = True

    total_epochs: int = 40
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    loss_mode: LossMode = "full"

    mask_ratio: float = 0.75
    gmm_n_components: int = 3
    gmm_update_every: int = 1
    gmm_max_batches: int = 100

    stage_epoch_boundaries: List[int] | None = None
    hard_max_epochs: int | None = None

    stage_loss_weights: List[Tuple[float, float]] = field(
        default_factory=lambda: [(1.0, 0.1), (0.5, 0.5), (0.1, 1.0)]
    )
    det_w_l1: float = 5.0
    det_w_giou: float = 2.0
    det_focal_alpha: float = 0.25
    det_focal_gamma: float = 2.0
    # Stabilization regularizers.
    id_loss_weight: float = 0.05
    tv_loss_weight: float = 0.01
    logit_reg_weight: float = 0.01
    logit_cap: float = 6.0
    car_loss_weight: float = 1.0
    person_loss_weight: float = 1.0
    grad_skip_threshold: float = 20.0

    ir_data_root: str = "/root/autodl-tmp/train"
    text_labels: List[str] = field(default_factory=lambda: ["person", "car"])
    num_workers: int = 4
    max_steps_per_epoch: int = -1

    # fp16 can be unstable for this OWLv2+CSMA setup; keep fp32 by default.
    use_amp: bool = False
    img_size: int = 960

    model_id: str = "/root/autodl-tmp/OWLv2/owlv2-base-patch16-finetuned"
    output_dir: str = "outputs_teammate"
    vis_every: int = 10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_overrides(cls, overrides: dict[str, Any] | None = None) -> "CSMAConfig":
        cfg = cls()
        if not overrides:
            return cfg
        unknown = [k for k in overrides if not hasattr(cfg, k)]
        if unknown:
            raise ValueError(f"Unknown CSMAConfig override keys: {unknown}")
        cfg = replace(cfg, **overrides)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.total_epochs <= 0:
            raise ValueError("total_epochs must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if not (0.0 < self.residual_scale <= 1.0):
            raise ValueError("residual_scale must be in (0, 1]")
        if self.pseudo_rgb_clamp <= 0:
            raise ValueError("pseudo_rgb_clamp must be > 0")
        if self.logit_cap <= 0:
            raise ValueError("logit_cap must be > 0")
        if self.grad_skip_threshold <= 0:
            raise ValueError("grad_skip_threshold must be > 0")
        if len(self.stage_loss_weights) != 3:
            raise ValueError("stage_loss_weights must contain exactly 3 stage tuples")
        if self.stage_epoch_boundaries is not None:
            if len(self.stage_epoch_boundaries) != 2:
                raise ValueError("stage_epoch_boundaries must be [easy_end, mixed_end]")
            b0, b1 = self.stage_epoch_boundaries
            if not (0 <= b0 <= b1 <= self.total_epochs):
                raise ValueError("stage_epoch_boundaries must satisfy 0 <= b0 <= b1 <= total_epochs")
        if not self.text_labels:
            raise ValueError("text_labels must be non-empty")
