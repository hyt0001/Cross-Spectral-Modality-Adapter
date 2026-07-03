"""
IR 域随机化增强（IRAugment）。

目的：训练 YOLO-CSMA 时，随机扰动 IR 图像的亮度/对比度/噪声/直方图分布，
      迫使 CSMA 学习传感器无关的 IR→pseudo-RGB 映射，改善跨数据集泛化。

只作用于 IR 图像（PIL），标注坐标不受影响。
增强强度通过 prob / 各参数控制，训练时 p=0.5~0.8 即可；eval 时不使用。
"""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class IRAugment:
    """
    IR 图像域随机化增强流水线。

    每次调用随机选择以下扰动的子集独立施加：
      1. 全局亮度偏移（模拟不同传感器基准温度）
      2. 对比度拉伸 / 压缩（模拟不同动态范围）
      3. Gamma 校正（非线性响应差异）
      4. 高斯噪声（传感器噪声差异）
      5. 随机水平翻转（LLVIP 有大量双向行人）
      6. 直方图均衡化（50% 概率，强制拉平分布）

    使用方式：
        aug = IRAugment(prob=0.8)
        pil_aug = aug(pil_ir)   # 训练时
    """

    def __init__(
        self,
        prob: float = 0.8,
        brightness_range: Tuple[float, float] = (0.6, 1.4),
        contrast_range: Tuple[float, float] = (0.6, 1.6),
        gamma_range: Tuple[float, float] = (0.5, 2.0),
        noise_std_max: float = 15.0,
        hflip_prob: float = 0.5,
        equalize_prob: float = 0.3,
    ) -> None:
        self.prob = prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.noise_std_max = noise_std_max
        self.hflip_prob = hflip_prob
        self.equalize_prob = equalize_prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img

        img = img.copy()

        # 1. 亮度
        if random.random() < 0.7:
            factor = random.uniform(*self.brightness_range)
            img = ImageEnhance.Brightness(img).enhance(factor)

        # 2. 对比度
        if random.random() < 0.7:
            factor = random.uniform(*self.contrast_range)
            img = ImageEnhance.Contrast(img).enhance(factor)

        # 3. Gamma（uint8 LUT）
        if random.random() < 0.5:
            gamma = random.uniform(*self.gamma_range)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.power(arr, gamma)
            img = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))

        # 4. 高斯噪声
        if random.random() < 0.5:
            std = random.uniform(0.0, self.noise_std_max)
            arr = np.array(img, dtype=np.float32)
            noise = np.random.randn(*arr.shape).astype(np.float32) * std
            img = Image.fromarray((arr + noise).clip(0, 255).astype(np.uint8))

        # 5. 直方图均衡化（模拟 LLVIP 高对比度行人特征）
        if random.random() < self.equalize_prob:
            from PIL import ImageOps
            img = ImageOps.equalize(img)

        return img
