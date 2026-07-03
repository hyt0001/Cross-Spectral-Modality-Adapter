## **Cross-Spectral Modality Adapter**
定位：一个即插即用的轻量级红外模态适配器
使冻结参数的RGB视觉模型具备红外目标检测能力

## 技术架构
═════════════════════════════════════════════════════════════
 训练阶段（Training）— 需要 RGB-IR 配对数据
═════════════════════════════════════════════════════════════

     I_rgb ─────────────────────────────────┐
     [B,3,H,W]                              │
                                     冻结DINO Backbone
                                     (input_proj后)
                                            │ F_rgb [B,L,256]
                                            │ stop_gradient
     I_ir ──────► ┌──────────────────┐      │
     [B,3,H,W]   │   CSMA（可训练）  │      ▼
                  │  ┌────┐ ┌──────┐ │  ┌──────────────┐
                  │  │ IRE│►│ RPCA │ │  │  CMSS计算    │
                  │  └────┘ └──────┘ │  │  + GMM掩码   │  ──► M_cmss [B,L]
                  │       ┌──┘       │  └──────────────┘
                  │  ┌────┘          │        ▲
                  │  │  PD  │        │   F_ir [B,L,256]
                  └──┴──────┴────────┘        │
                         │ Î_rgb              │
                         │ [B,3,H,W]          │
                         │                    │
                         ▼                    │
                  冻结DINO Backbone ───────────┘
                         │
                         ▼
                  冻结DINO Transformer
                         │
                         ▼
                  检测输出 ──────────────────► L_det
                  (boxes, logits)
                                              L_align = MSE(F_ir, F_rgb)[M_cmss=0]
                                                   │
                                    L_total = λ₁·L_align + λ₂·L_det
                                                   │
                                              ▼ backward ▼
                                         仅更新 CSMA 权重

═════════════════════════════════════════════════════════════
 推理阶段（Inference）— 仅需红外图像
═════════════════════════════════════════════════════════════

     I_ir ──► CSMA ──► Î_rgb ──► 冻结Grounding DINO ──► 检测框 + 文本标签
              2M参数         text_prompt（任意开放词汇）


## Process
1.输入rgb图像，经过GD的Backbone提取特征，得到F_rgb。

2.输入ir图像，经过CSMA得到**伪RGB图**，即Î_rgb。它再经过GD的Backbone提取得到，F_ir

3.CSMA内部：IRE是红外特征提取器，从红外图像提取特征。选用了CNN金字塔结构
   RPCA是**把红外图像特征 和 可学习的 RGB prototype token 做cross attention**。一个“让红外特征向 RGB 特征空间靠近”的模块。不是直接让红外图变成肉眼看起来很真实的 RGB，而是希望它变成 **Grounding DINO 这种 RGB 预训练模型更容易理解的输入**
   PD 是 Pixel Decoder，它把中间特征重新变回三通道图像，也就是伪 RGB 图像
   
4.把伪RGB图输入GD，得到输出。输出和真实标签对比可以得到**检测损失L_det**

5.F_rgb和F_ir进行CMSS计算，去判断某一部分是否需要**关注**

6.只计算需要关注部分的MSE误差（认为其他部分的误差没必要计算），即**对齐损失L_align**

7.总损失函数  L_total = λ₁·L_align + λ₂·L_det

8.根据总损失反向传播，**仅更新 CSMA 的权重**

#### RPCA
`rgb_prototypes` 会逐渐学到：
```
Grounding DINO 喜欢什么样的 RGB 特征？红外特征应该往哪些 RGB 风格方向调整？哪些视觉模式对 person/car 检测有帮助？
```
训练结束后，这些知识就存在 prototype token 里。
推理时虽然没有真实 RGB 图像，但模型可以让红外特征去“查询”这些 prototype token。

红外特征 x_ir 作为 Query
RGB prototype token 作为 Key 和 Value

