from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal


LossMode = Literal["align_only", "det_only", "full"]


@dataclass
class CSMAConfig:
    # Model structure
    ir_enc_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    num_rgb_prototypes: int = 512
    proto_dim: int = 256
    num_cross_attn_heads: int = 8
    use_residual: bool = True

    # Training pipeline
    total_epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    loss_mode: LossMode = "full"

    # CMSS / GMM
    mask_ratio: float = 0.75
    gmm_n_components: int = 3
    gmm_update_every: int = 10
    # GMM 采样上限（每次重拟合最多使用的 batch 数）；-1 表示遍历全量
    # 100 batch（≈200~800 张图）足够拟合 3 分量 GMM；smoke test 用 50
    gmm_max_batches: int = 100

    # 课程阶段边界：None 时用 [T//3, 2T//3]；若设 hard_max_epochs 则 Hard 仅占最后 N 个 epoch
    stage_epoch_boundaries: list[int] | None = None
    hard_max_epochs: int | None = None

    # Loss weights
    stage_loss_weights: list[tuple[float, float]] = field(
        default_factory=lambda: [(1.0, 0.1), (0.5, 0.5), (0.1, 1.0)]
    )
    det_w_bbox: float = 5.0
    det_w_giou: float = 2.0
    det_w_ce_enc: float = 0.1
    det_w_bbox_enc: float = 0.5
    det_w_giou_enc: float = 0.5

    # Pseudo-RGB 正则 loss 权重（Final Model 新增）
    # 默认全部关闭（0 / 1.0）以保持向后兼容；显式传参激活 Final Model 配置：
    #   起点 best: id=0.05 / tv=0.01 / logit=0.01 / clamp=3.0 / scale=0.1
    #   Final Model: id=0.005 / tv=0.05 / logit=0.02 / clamp=2.0 / scale=0.05
    id_loss_weight: float = 0.0       # L_id = MSE(pseudo_rgb, ir)；0=关闭
    tv_loss_weight: float = 0.0       # L_tv = Total Variation(pseudo_rgb)；0=关闭
    logit_reg_weight: float = 0.0     # L_logit = mean(logits²)；0=关闭
    pseudo_clamp: float = 0.0         # pseudo_rgb 像素值上限；0=不 clamp
    residual_scale: float = 1.0       # CSMA 残差缩放系数；1.0=现有行为

    # Data
    ir_data_root: str = "train"
    rgb_data_root: str = "train/rgb"
    text_prompt: str = "person. car."
    num_workers: int = 4

    # 调试 / 快速验证
    # 每 epoch 最大步数；-1=全量。smoke test 设 20 可在 2 分钟内完成 epoch
    max_steps_per_epoch: int = -1

    # 内存优化
    use_amp: bool = False             # 混合精度（fp16）训练，节省 ~50% 显存
    grad_ckpt: bool = False          # GroundingDINO 不支持 gradient_checkpointing_enable()
    # processor 图像尺寸上限（shortest_edge）；FLIR IR 640×512，不必放大到 800+
    img_size: int = 512

    # Paths / runtime outputs
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    output_dir: str = "outputs_csma"
    vis_every: int = 10

    # 消融实验控制字段（默认 "none" 不影响任何原有行为）
    # 合法值：
    #   "none"            — 主实验默认路径（三阶段 GMM-CMSS A→B→C）
    #   "random_mask"     — Ablation C-1：随机掩码 + 固定 λ=0.5/0.5
    #   "fixed_threshold" — Ablation C-2：固定 CMSS 阈值 μ₂=0.5 + 固定 λ
    #   "gmm_single_b"    — Ablation C-3：GMM 动态 + 固定 Stage B + 固定 λ
    cmss_ablation_mode: str = "none"

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict for logging or serialization."""
        return asdict(self)

    @classmethod
    def from_overrides(cls, overrides: dict[str, Any] | None = None) -> "CSMAConfig":
        """
        Build a config with optional overrides.

        Unknown keys are rejected to avoid silent misconfiguration.
        """
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
        """Validate value ranges and expected structure."""
        if len(self.ir_enc_channels) < 2 or any(c <= 0 for c in self.ir_enc_channels):
            raise ValueError("ir_enc_channels must contain at least two positive integers")

        if self.total_epochs <= 0:
            raise ValueError("total_epochs must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be > 0")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.vis_every <= 0:
            raise ValueError("vis_every must be > 0")

        if not (0.0 <= self.mask_ratio < 1.0):
            raise ValueError("mask_ratio must be in [0, 1)")
        if self.gmm_n_components <= 0:
            raise ValueError("gmm_n_components must be > 0")
        if self.gmm_update_every <= 0:
            raise ValueError("gmm_update_every must be > 0")
        if self.gmm_max_batches != -1 and self.gmm_max_batches <= 0:
            raise ValueError("gmm_max_batches must be > 0 or -1 (unlimited)")

        if self.num_rgb_prototypes <= 0:
            raise ValueError("num_rgb_prototypes must be > 0")
        if self.proto_dim <= 0:
            raise ValueError("proto_dim must be > 0")
        if self.num_cross_attn_heads <= 0:
            raise ValueError("num_cross_attn_heads must be > 0")
        if self.proto_dim % self.num_cross_attn_heads != 0:
            raise ValueError("proto_dim must be divisible by num_cross_attn_heads")

        if self.loss_mode not in ("align_only", "det_only", "full"):
            raise ValueError("loss_mode must be one of: align_only, det_only, full")

        if self.id_loss_weight < 0:
            raise ValueError("id_loss_weight must be >= 0")
        if self.tv_loss_weight < 0:
            raise ValueError("tv_loss_weight must be >= 0")
        if self.logit_reg_weight < 0:
            raise ValueError("logit_reg_weight must be >= 0")
        if self.pseudo_clamp < 0:
            raise ValueError("pseudo_clamp must be >= 0 (0 = disabled)")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be > 0")

        _valid_ablation_modes = {"none", "random_mask", "fixed_threshold", "gmm_single_b"}
        if self.cmss_ablation_mode not in _valid_ablation_modes:
            raise ValueError(
                f"cmss_ablation_mode must be one of: {_valid_ablation_modes}, "
                f"got '{self.cmss_ablation_mode}'"
            )

        if len(self.stage_loss_weights) != 3:
            raise ValueError("stage_loss_weights must contain exactly 3 stage tuples")
        for idx, pair in enumerate(self.stage_loss_weights):
            if len(pair) != 2:
                raise ValueError(f"stage_loss_weights[{idx}] must be a 2-tuple")
            la, ld = pair
            if la < 0 or ld < 0:
                raise ValueError(f"stage_loss_weights[{idx}] values must be >= 0")

        if self.stage_epoch_boundaries is not None:
            if len(self.stage_epoch_boundaries) != 2:
                raise ValueError("stage_epoch_boundaries must be [easy_end, mixed_end] (2 ints)")
            b0, b1 = self.stage_epoch_boundaries
            if not (0 < b0 < b1 <= self.total_epochs):
                raise ValueError(
                    "stage_epoch_boundaries must satisfy 0 < b0 < b1 <= total_epochs"
                )
        if self.hard_max_epochs is not None:
            if self.hard_max_epochs <= 0 or self.hard_max_epochs >= self.total_epochs:
                raise ValueError("hard_max_epochs must be in (0, total_epochs)")
