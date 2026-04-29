"""IR translator MVP：冻结 Grounding DINO + 可训练残差翻译网络 / CSMA 适配器。"""

from src.csma import CSMA, IREncoder, PixelDecoder, RGBPrototypeCrossAttention

__all__ = [
    "CSMA",
    "IREncoder",
    "PixelDecoder",
    "RGBPrototypeCrossAttention",
]
