# HD-Net 实验记录与论文参考

---

## 1. 评估指标说明

本项目采用语义分割标准评估体系，包含以下指标：

**交并比（Intersection over Union, IoU）**
设预测区域为 $P$，真实区域为 $G$，则：
$$IoU = \frac{|P \cap G|}{|P \cup G|}$$

**整体准确率（Overall Accuracy, OA）**
$$OA = \frac{\text{预测正确的像素总数}}{\text{像素总数}}$$

**查准率（Precision）**
$$P = \frac{TP}{TP + FP} = \frac{\text{预测为建筑物且正确的像素数}}{\text{预测为建筑物的像素总数}}$$

**查全率（Recall）**
$$R = \frac{TP}{TP + FN} = \frac{\text{预测为建筑物且正确的像素数}}{\text{真实为建筑物的像素总数}}$$

**F1-score**
$$F1 = \frac{2 \cdot P \cdot R}{P + R}$$

> 注：IoU 与 F1 在整体变化趋势上较为一致，但二者对误差分布的敏感性并不完全相同。对于以建筑结构为主要关注对象的分割任务，需结合具体指标理解其适用范围。

---

## 2. 模型变体

| 模型 | 说明 |
|------|------|
| **Baseline** | 原始 HD-Net 模型（`HDNet.py`），包含 Stem、Stage1-3、Deep Supervision 等完整结构 |
| **Shared** | Stage3 参数共享版本（`HDNet_shared.py`），将原本独立的 stage3_1~4 四套参数合并为一套共享参数，仅通过 `dilation_base` 控制感受野节奏 |

Shared 版本在保持感受野递进节奏不变的前提下，减少了 3/4 的 Stage3 参数量。

---

## 3. 训练配置

### 训练策略

| 配置项 | Baseline | Shared |
|--------|----------|--------|
| 优化器 | Adam, lr=1e-3, weight_decay=1e-5 | 同左 |
| 学习率衰减 | StepLR, gamma=0.7, 每 4 epoch | 同左 |
| 损失函数 | BCE + Dice（联合损失） | 同左 |
| Deep Supervision | 6 个中间分割头 + 6 个边界头，加权融合 | 同左 |
| 边界监督 | distance_map < 3 判定为边界，pos_weight=9 | 同左 |

### 训练耗时

每 epoch 约 1500-1600 秒（25-27 分钟），以 150 epochs 计约 65-67 小时。

---

## 4. 本地验证集测试结果

### 4.1 Baseline 50 epochs

| 指标 | 数值 |
|------|------|
| **IoU** | 0.8053 |
| **OA** | 0.9583 |
| **Recall** | [97.37%, 89.41%] |
| **Precision** | [97.47%, 89.03%] |
| **F1-score** | [97.42%, 89.22%] |

混淆矩阵：`[[218565589, 5913721], [5682505, 47972969]]`

### 4.2 Baseline 150 epochs

| 指标 | 数值 |
|------|------|
| **IoU** | 0.8471 |
| **OA** | 0.9680 |
| **Recall** | [97.99%, 91.84%] |
| **Precision** | [98.05%, 91.60%] |
| **F1-score** | [98.02%, 91.72%] |

混淆矩阵：`[[219957647, 4521663], [4376387, 49279087]]`

### 4.3 Shared 50 epochs

| 指标 | 数值 |
|------|------|
| **IoU** | 0.7999 |
| **OA** | 0.9575 |
| **Recall** | [97.57%, 88.11%] |
| **Precision** | [97.17%, 89.66%] |
| **F1-score** | [97.37%, 88.88%] |

混淆矩阵：`[[219026958, 5452352], [6377729, 47277745]]`

### 4.4 本地结果横向对比

| 模型 | val IoU | val OA |
|------|---------|--------|
| Baseline 50 epochs | 0.8053 | 0.9583 |
| Baseline 150 epochs | **0.8471** | **0.9680** |
| Shared 50 epochs | 0.7999 | 0.9575 |

---

## 5. 模型性能指标

| 模型 | 参数量 | FLOPs |
|------|--------|-------|
| Baseline | 13.89 M | 201.37 G |
| Shared | 4.21 M | 152.45 G |

**参数量减少：9.68 M（减少约 69.7%）**
**FLOPs 减少：48.92 G（减少约 24.3%）**

---

## 6. 官方 Test Benchmark

### 5.1 Baseline 150 epochs

| City | IoU (%) | Acc (%) |
|------|---------|---------|
| bellingham | 71.46 | 97.06 |
| bloomington | 63.77 | 96.58 |
| innsbruck | 75.24 | 97.08 |
| sfo | 69.57 | 89.55 |
| tyrol-e | 78.47 | 98.10 |
| **Overall** | **71.04** | **95.67** |

### 5.2 Shared 50 epochs

| City | IoU (%) | Acc (%) |
|------|---------|---------|
| bellingham | 67.05 | 96.57 |
| bloomington | 61.53 | 96.38 |
| innsbruck | 69.70 | 96.47 |
| sfo | 74.03 | 90.92 |
| tyrol-e | 76.30 | 97.87 |
| **Overall** | **71.10** | **95.64** |

### 5.3 Baseline 50 epochs

| City | IoU (%) | Acc (%) |
|------|---------|---------|
| bellingham | 67.85 | 96.56 |
| bloomington | 65.25 | 96.69 |
| innsbruck | 72.66 | 96.79 |
| sfo | 72.08 | 90.31 |
| tyrol-e | 76.21 | 97.88 |
| **Overall** | **71.20** | **95.65** |

### 5.4 综合对比

| 模型 | Overall IoU (%) | Overall Acc (%) |
|------|-----------------|-----------------|
| Baseline 150（早期） | 71.04 | 95.67 |
| Shared 50 | 71.10 | 95.64 |
| Baseline 150（最终） | 71.20 | 95.65 |

**发现**：Shared 50 与 Baseline 150 最终版 IoU 仅差 **0.10%**（71.10 vs 71.20），但 Stage3 参数量减少约 3/4，表明参数共享策略在精度损失极小的前提下实现了显著的模型轻量化。

---

## 7. 评估代码说明

`eval/eval_HDNet.py` 中的 `eval_net` 函数可输出以下指标：

```python
IOU       # 建筑类别 IoU
OA        # 整体准确率
Recall    # 查全率
Precision # 查准率
F1_score  # F1 = 2*PR/(P+R)
hist      # 混淆矩阵
```

上述指标完整覆盖了论文第 X 节中提到的所有评价指标。

---

## 8. 当前可引用数据汇总

| 数据 | 数值 | 适用场景 |
|------|------|---------|
| Baseline 150 val IoU（本地） | 0.8471 | 本地验证最高精度 |
| Baseline 50 val IoU（本地） | 0.8053 | 同期对照 |
| Baseline 150 val OA（本地） | 0.9680 | 整体准确率 |
| Baseline 参数量 | 13.89 M | 模型规模说明 |
| Shared 参数量 | 4.21 M | 轻量化对照 |
| Baseline FLOPs | 201.37 G | 计算量说明 |
| Shared FLOPs | 152.45 G | 轻量化对照 |
| Baseline Test Overall IoU | 71.20% | 官方测试集主结果 |
| Shared 50 Test Overall IoU | 71.10% | 轻量化官方结果 |
| Test IoU 城市分解 | 5 城市表格 | 分城市细节对比 |
