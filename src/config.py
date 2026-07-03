from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal


LossMode = Literal["align_only", "det_only", "full"]
AdapterMode = Literal["pixel", "feature"]
ValMetric = Literal["map_50", "person_car_mean"]


@dataclass
class CSMAConfig:
    # 适配器模式：pixel=CSMA 像素翻译；feature=FeatureAdapter 特征级适配
    adapter_mode: AdapterMode = "pixel"

    # Model structure (pixel / CSMA)
    ir_enc_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    num_rgb_prototypes: int = 512
    proto_dim: int = 256
    num_cross_attn_heads: int = 8
    use_residual: bool = True

    # FeatureAdapter 结构（feature 模式）
    fa_hidden_dim: int = 512
    fa_num_layers: int = 3
    fa_use_residual: bool = True
    # 最后一层 Linear 零初始化，使训练初期 out≈x（恒等映射）
    fa_zero_init: bool = True

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
    # True 时 Mixed 延续至训练结束，永不进入 Hard（Stage C）
    skip_hard_stage: bool = False

    # Val 早停指标：map_50=六类/全类 mAP；person_car_mean=仅 person+car AP 均值（M3FD 推荐）
    val_metric: ValMetric = "map_50"

    # Loss weights
    stage_loss_weights: list[tuple[float, float]] = field(
        default_factory=lambda: [(1.0, 0.1), (0.5, 0.5), (0.1, 1.0)]
    )
    det_w_bbox: float = 5.0
    det_w_giou: float = 2.0
    det_w_ce_enc: float = 0.1
    det_w_bbox_enc: float = 0.5
    det_w_giou_enc: float = 0.5
    # 像素重建损失权重：MSE(pseudo_rgb, rgb_real)，量级≈0.1-1.0，弥补 L_align 极小的问题
    # 设为 0.0 可完全禁用（向后兼容）
    lambda_recon: float = 1.0
    # pseudo 正则（OWL Final Model 对齐项）
    lambda_id: float = 0.0          # MSE(pseudo, IR)；M3FD 降低可减 car 假框
    lambda_tv: float = 0.0          # Total Variation 平滑
    lambda_logit_reg: float = 0.0   # 检测 logit 均值正则
    pseudo_clamp: float = 0.0       # pseudo 像素 clamp 上限；0=不 clamp
    residual_scale: float = 1.0     # pseudo = IR + scale * delta
    ema_decay: float = 0.0          # >0 时启用 EMA；0=关闭

    # 多层 L_align：指定要对齐的 DINO encoder 层索引（0-based，最多 6 层）。
    # 空列表 [] 表示只对齐 encoder 入口特征（原始行为）。
    # 建议 [1, 3, 5]（浅/中/深层），每层独立计算余弦对齐损失后取平均。
    align_layer_indices: list[int] = field(default_factory=lambda: [1, 3, 5])

    # bbox 加权 L_align：GT 框区域内的 patch 权重倍率（>1 加强目标区域对齐）。
    # 设为 1.0 退化为均匀权重（原始行为）。
    bbox_align_weight: float = 3.0

    # Pseudo-RGB 正则（参照 Final Model 配置；默认值向后兼容已有 checkpoint）
    # id_loss：限制 pseudo 与输入 IR 的 L1 距离，防止 car 亮块原样透传导致假框激增
    id_loss_weight: float = 0.005
    # tv_loss：total variation，鼓励 pseudo 图像平滑、减少噪点框
    tv_loss_weight: float = 0.05
    # logit_reg：对 GDINO 解码器 logits 的 L1 正则，抑制整体置信度过高
    logit_reg_weight: float = 0.02
    # pseudo_clamp：截断 pseudo 像素极端值（GDINO 归一化空间）；0 = 不截断（兼容旧权重）
    pseudo_clamp: float = 2.0
    # residual_scale：残差加法系数；1.0 = 当前行为（兼容旧权重），Final Model 推荐 0.05
    residual_scale: float = 1.0

    # Data
    ir_data_root: str = "train"
    rgb_data_root: str = "train/rgb"
    text_prompt: str = "person. car."
    num_workers: int = 4

    # 调试 / 快速验证
    # 每 epoch 最大步数；-1=全量。smoke test 设 20 可在 2 分钟内完成 epoch
    max_steps_per_epoch: int = -1

    # 内存优化
    use_amp: bool = True             # 混合精度（fp16）训练，节省 ~50% 显存
    grad_ckpt: bool = False          # GroundingDINO 不支持 gradient_checkpointing_enable()
    # processor 图像尺寸上限（shortest_edge）；FLIR IR 640×512，不必放大到 800+
    img_size: int = 512

    # Paths / runtime outputs
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    output_dir: str = "outputs_csma"
    vis_every: int = 10

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
        if self.lambda_recon < 0:
            raise ValueError("lambda_recon must be >= 0")
        if self.lambda_id < 0:
            raise ValueError("lambda_id must be >= 0")
        if self.lambda_tv < 0:
            raise ValueError("lambda_tv must be >= 0")
        if self.lambda_logit_reg < 0:
            raise ValueError("lambda_logit_reg must be >= 0")
        if self.pseudo_clamp < 0:
            raise ValueError("pseudo_clamp must be >= 0")
        if self.residual_scale < 0:
            raise ValueError("residual_scale must be >= 0")
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError("ema_decay must be in [0, 1)")

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
        if self.val_metric not in ("map_50", "person_car_mean"):
            raise ValueError("val_metric must be 'map_50' or 'person_car_mean'")

        if self.adapter_mode not in ("pixel", "feature"):
            raise ValueError("adapter_mode must be 'pixel' or 'feature'")
        if self.fa_hidden_dim <= 0:
            raise ValueError("fa_hidden_dim must be > 0")
        if self.fa_num_layers < 2:
            raise ValueError("fa_num_layers must be >= 2")

        if self.adapter_mode == "feature":
            if self.lambda_recon != 0.0:
                raise ValueError("feature 模式要求 lambda_recon=0.0")
            if len(self.align_layer_indices) != 0:
                raise ValueError("feature 模式要求 align_layer_indices=[]（单层 token 对齐）")
