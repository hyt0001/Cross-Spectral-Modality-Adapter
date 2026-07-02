"""
DINO 视觉 token 提取与 FeatureAdapter 注入桥接。

在 encoder 入口（`vision_features`）通过 forward_pre_hook 注入适配结果，
避免重复实现 decoder / loss 头逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import GroundingDinoForObjectDetection

from src.feature_adapter import FeatureAdapter


@dataclass
class VisionBundle:
    """
    DINO encoder 入口所需的视觉侧张量集合。

    Attributes:
        vision_features:           [B, L, D] input_proj 后 flatten 拼接的 token。
        vision_attention_mask:     [B, L] True=padding（与 encoder 接口一致取反后传入）。
        vision_position_embedding: [B, L, D] 含 level_embed 的位置编码。
        spatial_shapes:            [num_levels, 2]。
        spatial_shapes_list:       各 level (H, W) 列表。
        level_start_index:         [num_levels]。
        valid_ratios:              [B, num_levels, 2]。
        mask_flatten:              [B, L] padding mask（未取反，供 two_stage 使用）。
    """

    vision_features: torch.Tensor
    vision_attention_mask: torch.Tensor
    vision_position_embedding: torch.Tensor
    spatial_shapes: torch.Tensor
    spatial_shapes_list: list[tuple[int, int]]
    level_start_index: torch.Tensor
    valid_ratios: torch.Tensor
    mask_flatten: torch.Tensor


def extract_vision_tokens(
    dino_model: GroundingDinoForObjectDetection,
    pixel_values: torch.Tensor,
    pixel_mask: torch.Tensor,
) -> VisionBundle:
    """
    复现 GroundingDinoModel.forward 中 backbone→input_proj→flatten 逻辑，
    返回 encoder 入口处的 `vision_features` 及辅助张量。

    Args:
        dino_model:    GroundingDinoForObjectDetection（仅用 .model 子模块）。
        pixel_values:  [B, 3, H, W]。
        pixel_mask:    [B, H, W]。

    Returns:
        VisionBundle。
    """
    model = dino_model.model
    batch_size, _, height, width = pixel_values.shape
    device = pixel_values.device

    if pixel_mask is None:
        pixel_mask = torch.ones(batch_size, height, width, dtype=torch.long, device=device)

    vision_features, position_embeddings_list = model.backbone(pixel_values, pixel_mask)

    feature_maps: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for level, (source, mask) in enumerate(vision_features):
        feature_maps.append(model.input_proj_vision[level](source))
        masks.append(mask)

    if model.config.num_feature_levels > len(feature_maps):
        _len_sources = len(feature_maps)
        for level in range(_len_sources, model.config.num_feature_levels):
            if level == _len_sources:
                source = model.input_proj_vision[level](vision_features[-1][0])
            else:
                source = model.input_proj_vision[level](feature_maps[-1])
            mask = nn.functional.interpolate(
                pixel_mask[None].float(), size=source.shape[-2:]
            ).to(torch.bool)[0]
            pos_l = model.backbone.position_embedding(source, mask).to(source.dtype)
            feature_maps.append(source)
            masks.append(mask)
            position_embeddings_list.append(pos_l)

    source_flatten: list[torch.Tensor] = []
    mask_flatten_list: list[torch.Tensor] = []
    lvl_pos_embed_flatten: list[torch.Tensor] = []
    spatial_shapes_list: list[tuple[int, int]] = []

    for level, (source, mask, pos_embed) in enumerate(
        zip(feature_maps, masks, position_embeddings_list)
    ):
        _, _, h_lvl, w_lvl = source.shape
        spatial_shapes_list.append((h_lvl, w_lvl))
        src = source.flatten(2).transpose(1, 2)
        msk = mask.flatten(1)
        pos = pos_embed.flatten(2).transpose(1, 2)
        lvl_pos = pos + model.level_embed[level].view(1, 1, -1)
        source_flatten.append(src)
        mask_flatten_list.append(msk)
        lvl_pos_embed_flatten.append(lvl_pos)

    vision_feat = torch.cat(source_flatten, dim=1)
    mask_flatten = torch.cat(mask_flatten_list, dim=1)
    lvl_pos_embed = torch.cat(lvl_pos_embed_flatten, dim=1)
    spatial_shapes = torch.as_tensor(
        spatial_shapes_list, dtype=torch.long, device=device
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    valid_ratios = torch.stack([model.get_valid_ratio(m) for m in masks], dim=1).float()

    return VisionBundle(
        vision_features=vision_feat,
        vision_attention_mask=~mask_flatten,
        vision_position_embedding=lvl_pos_embed,
        spatial_shapes=spatial_shapes,
        spatial_shapes_list=spatial_shapes_list,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios,
        mask_flatten=mask_flatten,
    )


def forward_dino_with_feature_adapter(
    dino: GroundingDinoForObjectDetection,
    feature_adapter: FeatureAdapter,
    pixel_values: torch.Tensor,
    pixel_mask: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: Optional[list[dict[str, Any]]] = None,
    adapted_cache: Optional[dict[str, torch.Tensor]] = None,
) -> Any:
    """
    单次 DINO 前向，在 encoder 入口用 FeatureAdapter 替换 vision_features。

    Args:
        dino:            冻结的 GroundingDinoForObjectDetection。
        feature_adapter: 可训练 FeatureAdapter。
        pixel_values:    IR 图像 [B,3,H,W]。
        pixel_mask:      [B,H,W]。
        input_ids:       [B,T]。
        attention_mask:  [B,T]。
        labels:          检测损失标签；推理时可 None。
        adapted_cache:   若提供，写入 adapter 输出 token 供 L_align 复用。

    Returns:
        GroundingDinoObjectDetectionOutput（含 loss / loss_dict）。
    """

    def _pre_hook(module: nn.Module, args: tuple, kwargs: dict) -> tuple:
        vf = kwargs.get("vision_features")
        if vf is None and args:
            vf = args[0]
        if vf is None:
            return args, kwargs
        adapted = feature_adapter(vf)
        if adapted_cache is not None:
            adapted_cache["feat"] = adapted
        kwargs["vision_features"] = adapted
        return args, kwargs

    handle = dino.model.encoder.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    try:
        return dino(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    finally:
        handle.remove()


def forward_dino_eval_with_feature_adapter(
    dino: GroundingDinoForObjectDetection,
    feature_adapter: FeatureAdapter,
    pixel_values: torch.Tensor,
    pixel_mask: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Any:
    """
    推理专用封装（无 labels），内部调用 forward_dino_with_feature_adapter。
    """
    feature_adapter.eval()
    with torch.no_grad():
        return forward_dino_with_feature_adapter(
            dino,
            feature_adapter,
            pixel_values,
            pixel_mask,
            input_ids,
            attention_mask,
            labels=None,
        )
