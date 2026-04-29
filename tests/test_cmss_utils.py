"""
cmss_utils 单元 smoke 测试。

验证 compute_cmss / fit_gmm / build_cmss_mask / CMSSScheduler 的形状、值域与逻辑正确性。
运行命令：
    CUDA_VISIBLE_DEVICES=0 conda run -n RGBtest python -m pytest tests/test_cmss_utils.py -v
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.cmss_utils import (
    CMSSScheduler,
    build_cmss_mask,
    compute_cmss,
    fit_gmm,
)
from src.config import CSMAConfig


class TestComputeCmss(unittest.TestCase):
    """compute_cmss 形状与数值正确性。"""

    def test_output_shape(self) -> None:
        """输出形状应为 [B, L]。"""
        B, L, D = 2, 16, 256
        rgb = torch.randn(B, L, D)
        ir = torch.randn(B, L, D)
        out = compute_cmss(rgb, ir)
        self.assertEqual(tuple(out.shape), (B, L))

    def test_value_range(self) -> None:
        """输出值域应在 [0, 1]，且无 NaN / Inf。"""
        B, L, D = 2, 16, 256
        out = compute_cmss(torch.randn(B, L, D), torch.randn(B, L, D))
        self.assertFalse(torch.isnan(out).any(), "存在 NaN")
        self.assertFalse(torch.isinf(out).any(), "存在 Inf")
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_max_normalized_to_one(self) -> None:
        """全局 max 归一化后最大值应为 1。"""
        out = compute_cmss(torch.randn(3, 32, 128), torch.randn(3, 32, 128))
        self.assertAlmostEqual(float(out.max()), 1.0, places=4)

    def test_identical_low_variance_inputs(self) -> None:
        """相同低方差特征时，高余弦相似度 + 低方差 → 大 CMSS（高于随机基线）。"""
        feat = torch.zeros(1, 4, 64)
        feat[0, 0, 0] = 1.0
        feat[0, 1, 1] = 1.0
        feat[0, 2, 2] = 1.0
        feat[0, 3, 3] = 1.0
        out = compute_cmss(feat, feat)
        self.assertGreater(float(out.mean()), 0.0)


class TestFitGmm(unittest.TestCase):
    """fit_gmm 接口正确性。"""

    def test_sorted_means_ascending(self) -> None:
        """返回均值应严格升序。"""
        vals = np.concatenate([
            np.random.normal(0.2, 0.05, 300),
            np.random.normal(0.5, 0.05, 400),
            np.random.normal(0.8, 0.05, 300),
        ]).astype(np.float32)
        means, _ = fit_gmm(vals, n_components=3)
        self.assertEqual(len(means), 3)
        self.assertTrue(means[0] < means[1] < means[2])

    def test_gmm_object_returned(self) -> None:
        """应返回已拟合的 GaussianMixture 对象。"""
        from sklearn.mixture import GaussianMixture
        vals = np.random.rand(200).astype(np.float32)
        _, gmm = fit_gmm(vals)
        self.assertIsInstance(gmm, GaussianMixture)


class TestBuildCmssMask(unittest.TestCase):
    """build_cmss_mask 三阶段逻辑。"""

    def _make_map(self) -> torch.Tensor:
        """构造确定性的 [1, 10] CMSS map，值从 0.0 到 0.9。"""
        return torch.linspace(0.0, 0.9, 10).unsqueeze(0)

    def test_stage_a_masks_background(self) -> None:
        """Stage 0：CMSS > mu2 的位置 mask==1（背景被掩）。"""
        cmss = self._make_map()
        mu1, mu2, mu3 = 0.2, 0.5, 0.8
        mask = build_cmss_mask(cmss, stage=0, mu1=mu1, mu2=mu2, mu3=mu3)
        self.assertEqual(tuple(mask.shape), (1, 10))
        expected = (cmss > mu2).float()
        self.assertTrue(torch.allclose(mask, expected))

    def test_stage_c_masks_targets(self) -> None:
        """Stage 2：CMSS < mu1 的位置 mask==1（目标核心被掩）。"""
        cmss = self._make_map()
        mu1, mu2, mu3 = 0.2, 0.5, 0.8
        mask = build_cmss_mask(cmss, stage=2, mu1=mu1, mu2=mu2, mu3=mu3)
        expected = (cmss < mu1).float()
        self.assertTrue(torch.allclose(mask, expected))

    def test_stage_b_mask_ratio(self) -> None:
        """Stage 1：mask==1 的比例应约等于 mask_ratio（±5%）。"""
        from sklearn.mixture import GaussianMixture
        vals = np.random.rand(500).astype(np.float32)
        _, gmm = fit_gmm(vals)
        B, L = 4, 100
        cmss = torch.rand(B, L)
        mask_ratio = 0.75
        mu1, mu2, mu3 = 0.2, 0.5, 0.8
        mask = build_cmss_mask(
            cmss, stage=1, mu1=mu1, mu2=mu2, mu3=mu3,
            mask_ratio=mask_ratio, gmm=gmm,
        )
        actual_ratio = float(mask.mean())
        self.assertAlmostEqual(actual_ratio, mask_ratio, delta=0.1)

    def test_stage_b_requires_gmm(self) -> None:
        """Stage B 缺少 gmm 时应抛出 ValueError。"""
        cmss = torch.rand(2, 10)
        with self.assertRaises(ValueError):
            build_cmss_mask(cmss, stage=1, mu1=0.2, mu2=0.5, mu3=0.8)


class TestCMSSScheduler(unittest.TestCase):
    """CMSSScheduler 阶段、GMM 更新与损失权重。"""

    def _make_scheduler(self, total_epochs: int = 90) -> CMSSScheduler:
        cfg = CSMAConfig.from_overrides({"total_epochs": total_epochs})
        return CMSSScheduler(cfg)

    def test_stage_boundaries(self) -> None:
        """epoch 落在三个阶段的正确区间。"""
        sched = self._make_scheduler(total_epochs=90)
        self.assertEqual(sched.get_stage(0), 0)
        self.assertEqual(sched.get_stage(29), 0)
        self.assertEqual(sched.get_stage(30), 1)
        self.assertEqual(sched.get_stage(59), 1)
        self.assertEqual(sched.get_stage(60), 2)
        self.assertEqual(sched.get_stage(89), 2)

    def test_default_sorted_means(self) -> None:
        """GMM 未更新时 sorted_means 返回默认值不报错。"""
        sched = self._make_scheduler()
        m = sched.sorted_means
        self.assertEqual(len(m), 3)
        self.assertAlmostEqual(m[0], 0.2)
        self.assertAlmostEqual(m[1], 0.5)
        self.assertAlmostEqual(m[2], 0.8)

    def test_update_gmm_replaces_defaults(self) -> None:
        """update_gmm 后 sorted_means 应被真实值替换。"""
        sched = self._make_scheduler()
        vals = np.concatenate([
            np.random.normal(0.1, 0.02, 200),
            np.random.normal(0.4, 0.02, 300),
            np.random.normal(0.7, 0.02, 200),
        ]).astype(np.float32)
        sched.update_gmm(vals)
        m = sched.sorted_means
        self.assertLess(m[0], m[1])
        self.assertLess(m[1], m[2])
        self.assertNotAlmostEqual(m[0], 0.2, places=2)

    def test_should_update_gmm(self) -> None:
        """should_update_gmm 应在 gmm_update_every 的倍数 epoch 返回 True。"""
        cfg = CSMAConfig.from_overrides({"gmm_update_every": 10})
        sched = CMSSScheduler(cfg)
        self.assertTrue(sched.should_update_gmm(0))
        self.assertTrue(sched.should_update_gmm(10))
        self.assertFalse(sched.should_update_gmm(7))

    def test_get_loss_weights_per_stage(self) -> None:
        """各阶段权重应与 CSMAConfig.stage_loss_weights 一致。"""
        sched = self._make_scheduler(total_epochs=90)
        self.assertAlmostEqual(sched.get_loss_weights(0)[0], 1.0)   # Stage A
        self.assertAlmostEqual(sched.get_loss_weights(30)[0], 0.5)  # Stage B
        self.assertAlmostEqual(sched.get_loss_weights(60)[1], 1.0)  # Stage C


if __name__ == "__main__":
    unittest.main()
