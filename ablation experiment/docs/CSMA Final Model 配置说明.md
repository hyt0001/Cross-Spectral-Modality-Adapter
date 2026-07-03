
  
## 1. 训练是怎么一步步走过来的



可以想成 **三轮**，每一轮的权重都是下一轮的起点：
  

```

第一轮：从头训练 CSMA

→ 得到 checkpoint A（我们叫 gmm1_warm best）

→ AP50 ≈ 0.47

  

第二轮：在 A 上用小学习率继续微调（lr = 1e-5，其他超参不变）

→ 得到 checkpoint B（起点 best / finetune best）★

→ AP50 ≈ 0.48，person 明显变好，car 还略低

  

第三轮：在 B 上改 pseudo 相关正则，再训 2 个 epoch

→ 得到 Final Model ★★

→ AP50 ≈ 0.49，person 和 car 都更好

```

  
  
  

---

  

## 3. 从起点 best 到 Final Model：到底改了什么

  

### 3.1 没改什么

  

- **检测 loss、对齐 loss 的主干**没动（person/car 谁更重要也没改）

- **三阶段 λ（align / det 的比例 schedule）**没专门锁

- **学习率**仍是 `1e-5`

- **不是从头训练**，是直接加载起点 best 的权重继续训

  

### 3.2 改了什么（重点）

  

训练的总 loss 大致是：

  

```

总 loss = 对齐项 + 检测项

+ id_loss_weight × 「pseudo 图像要像原始 IR」

+ tv_loss_weight × 「pseudo 图像要平滑」

+ logit_reg_weight × 「检测分数别整体太高」

```

  

**相对起点 best，我们只动了后面三项的系数（以及 pseudo 生成的两个上限）：**

  

| 参数 | 起点 best 用的 | Final Model 用的 | 变大还是变小 | 通俗理解 |

|------|---------------|------------------|-------------|----------|

| **id_loss_weight** | 0.05 | **0.005** | **变小（÷10）** | 不再强迫 pseudo 图像紧贴 IR；IR 里 car 是大亮块，贴太紧容易出一堆假框 |

| **tv_loss_weight** | 0.01 | **0.05** | **变大（×5）** | 更要求 pseudo 平滑、少噪点，减少乱框 |

| **logit_reg_weight** | 0.01 | **0.02** | 变大（×2） | 略微压制「满屏检测框」，辅助项，不是主因 |

| **pseudo_clamp** | 3.0 | **2.0** | 变小 | pseudo 像素值不允许太极端，亮块不会那么炸 |

| **residual_scale** | 0.1 | **0.05** | 变小 | CSMA 对输入的改动幅度更小、更保守 |

  

**记忆口诀：**

  

- **id 放松**（最重要）→ car 假框少

- **tv 加强** → pseudo 更平滑

- **clamp / residual 收紧** → 输出更稳

  

---

  

## 4. 为什么要这样改（直觉版）

  

CSMA 的作用：把 **红外 IR** 转成 **伪 RGB（pseudo RGB）**，再送给冻结的检测器。

  

问题在于：

  

1. **person** 在 IR 里是小热斑，检测器本来弱，需要保留一点 IR 结构才能检出来

2. **car** 在 IR 里是大亮块；如果 loss 里一直说「pseudo 必须像 IR」（id_loss 权重大），大亮块原样进检测器 → **假框变多** → car AP 被拉低

  

之前训到起点 best 时，person 已经提上来了，但 car 仍略低于 baseline。

Final Model 的做法是：**在已有权重上，放松「像 IR」这条约束，加强「要平滑」这条约束**，让 pseudo 更像「正常的 RGB」，而不是 IR 的复印件。

  

只训 **2 个 epoch** 就停——训久了 n_preds（预测框数量）又会涨，car 会再掉。

  

---

  

## 5. 给队友迁移到其他模型时的建议

  

1. **先有一个自己的「起点 best」**

等价于我们的 checkpoint B：在你的模型上 CSMA 已经训到一版 AP 不错的权重。

  

2. **只改 pseudo 相关正则，短训 2 epoch**

重点：`id_loss_weight` 降到原来的 **1/10**，`tv_loss_weight` 提到原来的 **5 倍**；`pseudo_clamp`、`residual_scale` 酌情收紧。

  

3. **别训太久**

每个 epoch 看验证集；我们 ep2 最好，继续训 person 涨、car 跌。

  

4. **保存每一轮的 EMA 权重**

不要只留一个「AP 最高」的 best.pt，可能选到 car 更差的 epoch。我们最终用的是 **第 2 epoch 的 EMA 权重**。

  

5. **评估 threshold 固定**

同一数据集上 CSMA 和 baseline 用同一个 T（我们正式结果是 T=0.2）。

  

---

  
  
  

## 7. 一句话总结

  

> **Final Model = 在上一版 best 权重上，把「pseudo 要像 IR」的约束放松 10 倍、把「pseudo 要平滑」加强 5 倍，再训 2 个 epoch；检测器不动，只动 CSMA。**

  

如有问题可以直接对照 `项目记录.md` 第九节，或找我们对一下你模型上的 loss 项是否一一对应。