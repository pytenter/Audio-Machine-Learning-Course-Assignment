# ESC-50 四分类音频分类实验报告

## 1. 实验任务

从 ESC-50 中选择 `dog, rain, crying_baby, clock_tick` 四类环境声音，每类 40 条，共 160 条。数据使用固定随机种子 42 进行分层划分：70% 训练集（112 条）和 30% 测试集（48 条），因此每类训练 28 条、测试 12 条。

![选取类别与数据划分](./assignment/results/data_split.png)

## 2. 特征提取

音频统一重采样到 22050 Hz。处理步骤为：预加重、25 ms 分帧、10 ms 帧移、Hamming 窗、功率谱、64 个 Mel 滤波器、取对数、DCT。每帧保留 40 个 MFCC；再对时间维分别计算均值和标准差，因此每段音频得到 **80 维**特征。

下图分别展示四个类别中一段代表音频的 MFCC。横轴是时间，纵轴是 MFCC 系数编号，颜色表示系数值。

![四类音频的 MFCC 特征](./assignment/results/mfcc_features.png)

## 3. PCA 分析

PCA 只在标准化后的训练数据上拟合，再用于训练集和测试集，避免数据泄漏。累计解释方差达到至少 80% 所需的主成分数为 **14**，实际累计解释方差为 **81.06%**。三维特征空间见 `pca_3d.png`。

![PCA 三维特征空间](./assignment/results/pca_3d.png)

![PCA 累计解释方差](./assignment/results/pca_cumulative_variance.png)

## 4. kNN 实验

在训练集上使用 5 折分层交叉验证比较 k=1 至 20，最优值为 **k=4**，平均交叉验证准确率为 **77.67%**。最后仅在选定 k 后评估测试集，测试准确率为 **72.92%**。调参曲线见 `knn_k_performance.png`。

![不同 k 值的 kNN 交叉验证准确率](./assignment/results/knn_k_performance.png)

## 5. 第二个分类器：SVM

使用 RBF 核 SVM（标准化、C=10、gamma=`scale`）在同一训练集上训练，在同一测试集上的准确率为 **87.50%**。

## 6. 结果比较与结论

| 模型 | 测试准确率 |
|---|---:|
| kNN (k=4) | 72.92% |
| RBF-SVM | 87.50% |

![kNN 与 SVM 测试准确率对比](./assignment/results/model_comparison.png)

本实验中测试集表现更好的模型是 **RBF-SVM**。混淆矩阵见 `confusion_matrices.png`，逐类别 precision、recall 和 F1-score 见 `classification_report.csv`。由于总样本量只有 160，单次 70/30 划分的结果存在随机波动；训练阶段使用交叉验证选择 k，测试集只用于最终比较。

![kNN 与 SVM 混淆矩阵](./assignment/results/confusion_matrices.png)

## 7. 复现实验

```bash
python assignment/esc50_audio_classification.py
```

