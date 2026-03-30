# 毕业设计相关实验记录：HD-Net复现与轻量化改进

本仓库主要记录本人本科毕业设计阶段已经完成的实验工作。

目前实际完成的内容，主要集中在 **HD-Net 模型复现、机制分析与轻量化改进** 上；
前期虽然调研过扩散模型在分割任务中的应用，但初步尝试进展缓慢，考虑到又面临复试准备，于是选择转为围绕 HD-Net 的轻量化改进开展工作。

---

## 1. 工作内容概述

当前已完成的工作主要包括：

- HD-Net 模型复现
- 训练、验证、推理流程搭建
- 中间结果导出与机制分析
- 推理阶段消融实验
- Stage3 参数共享轻量化改进
- 实验结果整理与论文初稿撰写

这些工作对应的是一个较完整的实验过程，而不只是单纯跑通模型。

---

## 2. 任务与数据集

任务为**高分辨率遥感影像建筑物分割**。

目前主要使用的数据集为：

- Inria Aerial Image Labeling Dataset

此外，前期也对 U-Net、DeepLabV3+ 等基线模型做过了解和实验尝试，但当前仓库的核心实验结果主要仍围绕 Inria 和 HD-Net 展开。

---

## 3. 基线模型

本仓库复现的基线模型为：

**HD-Net: High-Resolution Decoupled Network for Building Footprint Extraction via Deeply Supervised Body and Boundary Decomposition**

> 论文作者: Yuxuan Li, Danfeng Hong, Chenyu Li, Jing Yao, Jocelyn Chanussot
> 发表于: ISPRS Journal of Photogrammetry and Remote Sensing, 2024
> 原文链接: https://authors.elsevier.com/a/1iYW63I9x1qnCx

该模型的核心思路，是将建筑物分割中的区域信息与边界信息进行解耦建模，并通过多阶段 refinement 与动态融合提升预测效果。

在复现过程中，我重点关注了以下几个方面：

- 主体-边界解耦机制
- flow 对齐模块的作用
- Stage3 的重复细化结构
- 动态融合在推理阶段的表现

---

## 4. 我的主要改动

在对 baseline 结构进行分析后，我将改动重点放在 Stage3。

原始 Stage3 由多组结构相近的子模块串行组成，每一层使用独立参数。
在当前版本中，我尝试将其中的重复结构改为**参数共享**形式，即使用同一组参数进行多步迭代调用，以减少模型参数量和计算复杂度。

这一改动的出发点，不是重新设计一个全新网络，而是在尽量保持原有整体流程不变的前提下，对其重复结构进行压缩。

---

## 5. 当前结果

目前已完成的主要结果如下：

- 参数量：**13.89M -> 4.21M**
- FLOPs：**201.37G -> 152.45G**
- 官方 benchmark Overall IoU：**71.20% -> 71.10%**

从当前结果来看，参数共享版本在显著减少参数量的情况下，整体性能基本保持稳定。

需要说明的是，目前部分实验之间的训练轮次尚未完全对齐，因此现阶段的结果主要用于说明该改动具备可行性，后续仍可继续补充更严格的对比实验。

---

## 6. 已完成的分析工作

除最终结果外，当前还完成了以下几类分析：

- 中间特征导出与可视化
- edge map / seg map 等结果观察
- baseline 与 shared 的对比
- 去除 flow、去除解耦、去除动态权重等推理阶段消融
- 训练过程与实验记录整理

这些内容主要用于帮助理解模型机制，以及支持后续论文写作。

---

## 7. 仓库内容说明

### 7.1 核心代码

| 文件 | 说明 |
|------|------|
| model/HDNet.py | Baseline网络实现 |
| model/HDNet_shared.py | 参数共享轻量化版本 |
| utils/dataset.py | 数据加载与预处理 |
| utils/sync_batchnorm/ | 分布式训练同步BatchNorm |
| eval/eval_HDNet.py | IoU、Precision、Recall等评估指标计算 |
| Train_HDNet.py | 模型训练脚本 |
| Eval_HDNet.py | 模型验证脚本 |
| split_inria.py | Inria数据集划分脚本 |

### 7.2 辅助脚本

| 文件 | 说明 |
|------|------|
| analyze_params.py | 统计模型参数量 |
| cal_FLOPs.py | 统计模型FLOPs计算量 |
| boundary_dataset_generate.py | 根据距离图生成边界标签 |
| visualize.py / Eval_HDNet_visualize.py | 分割结果可视化 |
| plot_training_curve.py | 绘制训练loss曲线 |
| compress.py | 模型权重压缩脚本 |
| render_vis_from_pt.py | 从权重文件导出中间特征图 |
| infer_inria_test.py | Inria测试集推理脚本 |
| make_panel.py | 多图拼接成panel |

### 7.3 配置文件

| 文件 | 说明 |
|------|------|
| requirements.txt | Python依赖包 |
| mmseg_environment.yml | Conda环境配置 |
| dataset/final_train.txt | 本地训练集划分 |
| dataset/val.txt | 本地验证集划分 |
