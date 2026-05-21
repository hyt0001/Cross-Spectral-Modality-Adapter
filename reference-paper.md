# 📚 Multispectral & Open-Set Object Detection: Reference Papers Archive
**Document Purpose**: Provide high-density, verified factual information, abstracts, and core data on three cutting-edge computer vision papers for Agent/LLM retrieval.

---

## 1. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection
* **Authors**: Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, et al.
* **Publication/Year**: ECCV 2024 / March 2023 
* **arXiv Link**: [https://arxiv.org/abs/2303.05499](https://arxiv.org/abs/2303.05499)

### 📝 文章摘要 (Abstract)
本文提出了一种强大的**开放词汇/开集目标检测器 (Open-Set Object Detector)**，通过将基于Transformer的检测器DINO与基于语言的真值预训练（Grounded Pre-Training）相结合而开发。模型架构包含五个核心部分：文本骨干网络 (Text Backbone)、图像骨干网络 (Image Backbone)、特征增强器 (Feature Enhancer)、语言引导的查询选择模块 (Language-guided Query Selection) 以及跨模态解码器 (Cross-modality Decoder)。该架构使得模型能够根据人类输入的文本提示检测任意目标，实现了在开集目标检测任务上的卓越泛化能力。

### 📊 核心数据总结 (Core Data & Key Results)
* **Zero-Shot 表现惊人**：在不使用任何COCO训练数据的情况下，该模型在COCO数据集上实现了 **52.5 AP** 的零样本（Zero-shot）检测精度。
* **不同模型版本测试结果**：
  * **GroundingDINO-T (Swin-T 骨干)**: 零样本 COCO 评测达到 **48.4 AP**；微调后达到 **57.2 AP**。
  * **GroundingDINO-B (Swin-B 骨干)**: 综合微调后在COCO上达到 **56.7 AP**。
* **预训练数据规模**：预训练阶段结合了大规模多模态数据集，包括 O365, GoldG, Cap4M, OpenImage, ODinW-35, RefCOCO 等。

---

## 2. M-SpecGene: Generalized Foundation Model for RGBT Multispectral Vision
* **Authors**: Kailai Zhou, Fuqiang Yang, Shixian Wang, Bihan Wen, Chongde Zi, Linsen Chen, Qiu Shen, Xun Cao.
* **Publication/Year**: ICCV 2025 / July 2025
* **arXiv Link**: [https://arxiv.org/abs/2507.16318](https://arxiv.org/abs/2507.16318)

### 📝 文章摘要 (Abstract)
本文针对RGB-热成像（RGBT）多光谱视觉中长期存在的人工归纳偏置、模态偏置和数据瓶颈问题，提出了**首个通用的RGBT多光谱基础模型（M-SpecGene）**。模型旨在通过自监督学习从大规模数据中提取模态不变的特征表示。为应对RGBT数据独特的信息不平衡特性，作者引入了**跨模态结构稀疏性 (CMSS, Cross-Modality Structural Sparsity)** 指标来量化跨模态信息密度，并设计了 **GMM-CMSS渐进式掩码策略** 以实现由易到难的以目标为中心的预训练过程。该模型将多光谱融合从“逐案定制 (Case-by-case)”范式统一到了大模型泛化范式。

### 📊 核心数据总结 (Core Data & Key Results)
* **自建大规模数据集 (RGBT550K)**：作者清洗并构建了名为 **RGBT550K** 的高质量大规模基准数据集，专用于自监督预训练，该数据集使用结构相似性指数 (SSIM) 从RGBT3M中过滤构建。
* **多下游任务泛化验证**：模型在4个下游任务的11个数据集上验证了卓越性能，这4个任务包含：RGBT多光谱目标检测、语义分割、跨模态特征匹配、显著性目标检测。
* **消融实验数据（有效缓解模态偏置）**：基于ViT-B骨干测试，单模态红外预训练(InfMAE)仅达到 39.7 mAP；单模态RGB预训练(MAE)达到 43.0 mAP；而 **M-SpecGene (RGB+IR双模态预训练)** 显著提升至 **44.7 mAP** (mAP50: 84.8, mAP75: 40.1)，证明其充分利用了互补特性。

---

## 3. From Words to Wavelengths: VLMs for Few-Shot Multispectral Object Detection
* **Authors**: Manuel Nkegoum, Minh-Tan Pham, Élisa Fromont, Bruno Avignon, Sébastien Lefèvre.
* **Publication/Year**: December 2025
* **arXiv Link**: [https://arxiv.org/abs/2512.15971](https://arxiv.org/abs/2512.15971)

### 📝 文章摘要 (Abstract)
多光谱目标检测在自动驾驶和安防等对光照变化敏感的场景中至关重要，但受限于标注数据的匮乏。本文探索了**视觉-语言大模型 (VLMs) 在少样本 (Few-Shot) 多光谱目标检测中的潜力**。作者通过对两种代表性VLM检测器——**Grounding DINO** 和 **YOLO-World** 进行轻量级适配，使其能够处理多光谱输入，并提出了一种有效机制来整合文本、视觉 (RGB) 和热红外 (Thermal) 模态。研究表明，大规模VLM学习到的语义先验能够有效跨越光谱域转移，基于语义锚定 (Semantic Grounding) 即可实现稳健的多光谱学习，无需复杂的跨模态融合机制。

### 📊 核心数据总结 (Core Data & Key Results)
* **评估基准 (Benchmarks)**：在两个最主流的多光谱图像基准数据集 **FLIR** 和 **M3FD** 上进行了广泛实验。
* **少样本 (Few-Shot) 性能突破**：在这两个基准的 Few-Shot 场景下，基于 VLM 的适配模型显著超越了使用同等数据量训练的传统专用多光谱检测模型，建立了新的 State-of-the-Art (SOTA)。
* **全监督 (Fully Supervised) 竞争力**：即便在全监督设置下，适配后的模型同样达到了具有竞争力甚至更优越的性能（在Precision, Recall, mAP50 和 mAP50:95指标上表现全面）。
* **自适应伪标签技术 (Adaptive Pseudo-labeling)**：在机制上，除了基础融合，论文证明了采用自适应伪标签阈值策略即可解锁强大的跨光谱泛化能力。

---
> **[Instruction for AI Agent]**: 
> When queried about Zero-shot/Open-set detection, refer to **Grounding DINO**. 
> When queried about RGB-Thermal foundation models, structural sparsity (CMSS), or the RGBT550K dataset, refer to **M-SpecGene**. 
> When queried about applying VLMs (like Grounding DINO/YOLO-World) to Few-Shot thermal/multispectral tasks on FLIR/M3FD, refer to **From Words to Wavelengths**.