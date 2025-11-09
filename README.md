# MovieLens 20M 推荐系统与聚类分析

一个基于 MovieLens 20M 数据集的综合机器学习项目，实现了推荐系统、聚类算法和降维分析。

## 项目概述

本项目应用了多种机器学习算法来分析电影数据，包括：

1. **推荐系统**：基于 SVD (奇异值分解) 的协同过滤
2. **聚类算法**：K-means 和层次聚类
3. **降维算法**：PCA (主成分分析)

## 主要功能

### 1. 推荐系统
- 实现基于矩阵分解 (SVD) 的协同过滤算法
- 为用户生成个性化电影推荐
- 查找相似电影
- 评估推荐质量（MAE, RMSE）

### 2. 聚类分析
- 使用 K-means 和层次聚类对电影进行分组
- 比较不同聚类方法的性能
- 使用多种指标评估聚类质量：
  - 轮廓系数 (Silhouette Score)
  - Davies-Bouldin 指数
  - Calinski-Harabasz 指数
- 自动寻找最优聚类数 k

### 3. 降维分析
- 使用 PCA 对高维特征进行降维
- 分析降维对聚类效果的影响
- 可视化方差解释

## 项目结构

```
DSA5101/
├── main.py                  # 主程序入口
├── requirements.txt         # Python 依赖包
├── download_data.sh         # 数据下载脚本
├── README.md               # 项目说明文档
├── src/
│   ├── data_loader.py      # 数据加载和预处理
│   ├── recommender.py      # 推荐系统实现
│   ├── clustering.py       # 聚类算法实现
│   └── visualization.py    # 可视化工具
├── data/                   # 数据目录
│   └── ml-20m/            # MovieLens 20M 数据集
├── figures/                # 生成的图表
└── results/                # 分析结果
```

## 数据集

本项目使用 [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)，包含：
- 20,000,263 条评分数据
- 27,278 部电影
- 138,493 个用户
- 465,564 个标签
- 电影基因组数据

## 安装和使用

### 1. 环境配置

确保已安装 Python 3.7+，然后安装依赖包：

```bash
pip install -r requirements.txt
```

### 2. 下载数据

有两种方式获取数据：

**方式一：使用提供的脚本**
```bash
chmod +x download_data.sh
./download_data.sh
```

**方式二：手动下载**
1. 访问 [Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset/data)
2. 下载 `ml-20m.zip`
3. 解压到 `data/ml-20m/` 目录

确保存在以下文件：
- `data/ml-20m/ratings.csv`
- `data/ml-20m/movies.csv`
- `data/ml-20m/genome-scores.csv` (可选)

### 3. 运行分析

```bash
python main.py
```

程序将依次执行：
1. 数据加载和预处理
2. 推荐系统训练和评估
3. 聚类分析（K-means vs 层次聚类）
4. 降维分析（PCA 对聚类的影响）

运行完成后，所有可视化结果将保存到 `figures/` 目录。

## 算法详解

### 1. SVD 协同过滤

使用奇异值分解 (SVD) 对用户-电影评分矩阵进行降维：

```
R ≈ U × Σ × V^T
```

其中：
- R: 用户-电影评分矩阵
- U: 用户特征矩阵
- Σ: 奇异值矩阵
- V^T: 电影特征矩阵

**优点：**
- 降低维度，减少噪声
- 捕捉潜在特征
- 计算效率高

### 2. K-means 聚类

通过迭代优化将电影分配到 k 个聚类中：

```python
# 伪代码
1. 随机初始化 k 个聚类中心
2. 重复直到收敛：
   a. 将每个样本分配到最近的聚类中心
   b. 更新聚类中心为该聚类所有样本的均值
```

**特点：**
- 快速、可扩展
- 需要预先指定聚类数 k
- 对初始值敏感

### 3. 层次聚类

自底向上构建聚类层次结构：

```python
# 伪代码
1. 每个样本作为一个聚类
2. 重复直到只剩一个聚类：
   a. 找到距离最近的两个聚类
   b. 合并这两个聚类
```

**特点：**
- 不需要预先指定聚类数
- 可以生成树状图
- 计算复杂度较高

### 4. PCA 降维

找到数据方差最大的主成分方向：

```
X_reduced = X × W
```

其中 W 是前 k 个主成分构成的矩阵。

**优点：**
- 去除冗余特征
- 降低噪声
- 提高计算效率

## 评估指标

### 推荐系统指标

- **MAE (Mean Absolute Error)**：平均绝对误差，越小越好
- **RMSE (Root Mean Square Error)**：均方根误差，越小越好

### 聚类质量指标

- **轮廓系数 (Silhouette Score)**：范围 [-1, 1]，越大越好
  - 衡量样本与其所在聚类的相似度和与其他聚类的差异度

- **Davies-Bouldin 指数**：越小越好
  - 衡量聚类间的分离度和聚类内的紧密度

- **Calinski-Harabasz 指数**：越大越好
  - 衡量聚类间方差和聚类内方差的比率

## 实验结果

运行 `main.py` 后，会在 `figures/` 目录生成以下可视化结果：

1. `recommendation_performance.png` - 推荐系统性能
2. `clustering_comparison.png` - K-means vs 层次聚类比较
3. `kmeans_distribution.png` - 聚类分布
4. `kmeans_2d_visualization.png` - 2D 可视化聚类结果
5. `silhouette_elbow.png` - 轮廓系数肘部曲线
6. `davies_bouldin_elbow.png` - DB 指数肘部曲线
7. `pca_variance.png` - PCA 方差解释
8. `pca_comparison.png` - 降维前后聚类效果对比

## 主要发现

### 1. 推荐系统
- SVD 协同过滤能够有效预测用户评分
- 低维度（50 维）即可捕捉主要特征
- 冷启动问题需要额外处理

### 2. 聚类分析
- K-means 速度快，适合大规模数据
- 层次聚类能够揭示层次结构
- 电影类型对聚类结果有显著影响

### 3. 降维效果
- PCA 能够显著减少特征维度
- 降维后聚类速度提升
- 适当的降维可能改善聚类质量（去除噪声）
- 过度降维会损失重要信息

## 扩展方向

1. **推荐系统**：
   - 实现深度学习推荐模型（如 NCF, AutoEncoder）
   - 加入时间因素（时序推荐）
   - 混合推荐（内容+协同过滤）

2. **聚类分析**：
   - DBSCAN 等基于密度的聚类
   - 谱聚类
   - 图聚类算法

3. **特征工程**：
   - 使用 NLP 处理标签和电影描述
   - 提取更多电影元数据特征
   - 用户行为特征工程

## 依赖库

- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy >= 1.7.0

## 参考资料

1. [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
2. [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
3. [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
4. [PCA: Principal Component Analysis](https://scikit-learn.org/stable/modules/decomposition.html#pca)

## 许可证

本项目仅用于学习和研究目的。MovieLens 数据集使用需遵循 GroupLens 的许可条款。

## 作者

DSA5101 课程项目

## 致谢

感谢 GroupLens Research 提供的 MovieLens 数据集。
