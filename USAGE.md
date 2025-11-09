# 使用指南

## 快速开始

### 1. 准备数据

**选项 A: 自动下载（推荐）**
```bash
./download_data.sh
```

**选项 B: 手动下载**
1. 访问 https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset
2. 下载并解压到 `data/ml-20m/`

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行完整分析

```bash
python main.py
```

这将执行完整的分析流程，包括：
- 推荐系统训练和评估
- 聚类分析（K-means 和层次聚类）
- 降维分析（PCA）
- 生成所有可视化结果

运行时间：约 10-30 分钟（取决于数据量和硬件）

## 交互式分析

如果你想进行交互式探索，可以使用 Jupyter notebook：

```bash
jupyter notebook notebooks/quick_demo.ipynb
```

## 自定义分析

### 修改参数

你可以在 `main.py` 中修改以下参数：

**数据采样**
```python
# 在 main.py 中修改
loader.load_data(sample_size=2000000)  # 使用 200 万条数据
# 或
loader.load_data()  # 使用全部数据（2000 万条）
```

**推荐系统参数**
```python
recommender = SVDRecommender(n_components=50)  # SVD 维度
```

**聚类参数**
```python
labels = clusterer.kmeans_clustering(features, n_clusters=10)  # 聚类数
```

**降维参数**
```python
reducer = DimensionalityReducer(n_components=20)  # PCA 维度
```

### 使用单个模块

**只运行推荐系统**
```python
from src.data_loader import MovieLensLoader
from src.recommender import SVDRecommender

loader = MovieLensLoader()
loader.load_data(sample_size=1000000)

user_movie_matrix, _ = loader.preprocess_for_recommendation()
recommender = SVDRecommender(n_components=50)
recommender.fit(user_movie_matrix)

# 获取推荐
recommendations = recommender.recommend_for_user(user_id=1, top_n=10)
```

**只运行聚类**
```python
from src.data_loader import MovieLensLoader
from src.clustering import MovieClusterer

loader = MovieLensLoader()
loader.load_data()

movie_features = loader.create_movie_features(use_genome=False)
clusterer = MovieClusterer()
labels = clusterer.kmeans_clustering(movie_features, n_clusters=10)
```

## 输出文件

运行后会生成以下文件：

### 可视化结果 (`figures/` 目录)
- `recommendation_performance.png` - 推荐系统性能
- `clustering_comparison.png` - 聚类方法比较
- `kmeans_distribution.png` - K-means 聚类分布
- `kmeans_2d_visualization.png` - 2D 聚类可视化
- `silhouette_elbow.png` - 轮廓系数肘部曲线
- `davies_bouldin_elbow.png` - DB 指数肘部曲线
- `pca_variance.png` - PCA 方差解释
- `pca_comparison.png` - 降维前后比较
- `pca_reduced_clusters_2d.png` - 降维后聚类可视化

## 常见问题

### Q: 内存不足怎么办？

A: 减少数据采样大小：
```python
loader.load_data(sample_size=500000)  # 使用 50 万条数据
```

### Q: 运行时间太长？

A: 有几个优化选项：
1. 使用更小的数据样本
2. 减少 SVD 维度
3. 减少聚类数 k
4. 不使用 genome scores（设置 `use_genome=False`）

### Q: 如何解释聚类结果？

A: 查看每个聚类的：
1. 主要电影类型（top_genres）
2. 聚类大小
3. 代表性电影示例

### Q: 如何评估推荐质量？

A: 主要看两个指标：
- **MAE**：平均绝对误差，越小越好（典型值：0.6-0.9）
- **RMSE**：均方根误差，越小越好（典型值：0.8-1.2）

### Q: 什么是好的聚类？

A: 看三个指标：
- **轮廓系数**：> 0.5 表示聚类清晰
- **Davies-Bouldin**：< 1.0 表示聚类紧密
- **Calinski-Harabasz**：越大越好（相对值）

## 进阶使用

### 保存模型

```python
import pickle

# 保存推荐模型
with open('results/recommender.pkl', 'wb') as f:
    pickle.dump(recommender, f)

# 加载模型
with open('results/recommender.pkl', 'rb') as f:
    recommender = pickle.load(f)
```

### 导出聚类结果

```python
import pandas as pd

# 创建电影-聚类映射
movie_clusters = pd.DataFrame({
    'movieId': clusterer.movie_ids,
    'cluster': clusterer.labels
})

# 保存到 CSV
movie_clusters.to_csv('results/movie_clusters.csv', index=False)
```

### 批量推荐

```python
# 为多个用户生成推荐
user_recommendations = {}
for user_id in user_list:
    recommendations = recommender.recommend_for_user(user_id, top_n=20)
    user_recommendations[user_id] = recommendations

# 保存推荐结果
import json
with open('results/recommendations.json', 'w') as f:
    json.dump(user_recommendations, f)
```

## 性能优化建议

1. **内存优化**：
   - 使用数据采样
   - 增加过滤阈值（min_user_ratings, min_movie_ratings）

2. **速度优化**：
   - 使用较小的 SVD 维度
   - K-means 比层次聚类快
   - 降低聚类数 k

3. **质量优化**：
   - 使用完整数据集
   - 使用 genome scores 作为特征
   - 调整超参数（交叉验证）

## 技术支持

如有问题，请查看：
1. README.md - 项目概述
2. 源代码注释 - 详细实现说明
3. Jupyter notebook - 交互式示例
