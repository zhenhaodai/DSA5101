# 基于图的推荐系统

本项目实现了基于 PageRank 算法的图推荐系统，并与传统的 SVD 协同过滤方法进行了对比和融合。

## 新增功能

### 1. PageRank 推荐系统 (`PageRankRecommender`)

基于图结构的推荐算法，结合了 PageRank 和协同过滤：

- **二分图构建**：创建用户-电影二分图，边权重为评分
- **PageRank 算法**：计算电影在图中的重要性分数
- **协同过滤**：使用余弦相似度计算电影之间的相似性
- **混合推荐**：结合 PageRank 分数和协同过滤分数生成推荐

**参数：**
- `alpha`: PageRank 阻尼系数 (默认 0.85)
- `cf_weight`: 协同过滤权重 (默认 0.5，范围 0-1)

### 2. 混合推荐系统 (`HybridRecommender`)

结合 SVD 和 PageRank 的优势：

- **SVD 推荐**：基于矩阵分解的协同过滤
- **PageRank 推荐**：基于图结构的推荐
- **加权融合**：可调节的权重组合两种方法的推荐结果

**参数：**
- `svd_weight`: SVD 权重 (默认 0.5，范围 0-1)

### 3. 消融实验脚本 (`ablation_study.py`)

系统性地比较不同推荐算法的性能：

- **基线模型**：SVD 协同过滤
- **PageRank 模型**：测试不同 CF 权重 (0.3, 0.5, 0.7)
- **混合模型**：测试不同 SVD 权重 (0.3, 0.5, 0.7)
- **性能指标**：MAE、RMSE
- **可视化**：生成性能对比图表

## 使用方法

### 运行主程序（包含所有推荐系统）

```bash
python main.py
```

主程序将：
1. 训练 SVD 基线模型
2. 训练 PageRank + 协同过滤模型
3. 训练混合推荐模型
4. 对比三种方法的性能
5. 展示推荐示例

### 运行消融实验

```bash
python ablation_study.py
```

消融实验将：
1. 系统性地测试不同参数配置
2. 生成详细的性能对比报告
3. 保存可视化结果到 `./figures/` 目录

## 推荐系统对比

### SVD 基线
- **优点**：训练速度快，预测准确
- **缺点**：忽略了用户-电影网络的拓扑结构

### PageRank + 协同过滤
- **优点**：利用图结构捕获电影的全局重要性
- **缺点**：计算开销较大（需要构建图）

### 混合推荐
- **优点**：结合两种方法的优势，性能更稳定
- **缺点**：需要训练两个模型

## 代码结构

```
DSA5101/
├── src/
│   └── recommender.py          # 推荐系统实现
│       ├── SVDRecommender      # SVD 基线
│       ├── PageRankRecommender # PageRank + CF
│       └── HybridRecommender   # 混合推荐
├── main.py                     # 主程序
├── ablation_study.py          # 消融实验脚本
└── figures/                   # 可视化结果输出目录
```

## 算法原理

### PageRank 在推荐系统中的应用

1. **构建用户-电影二分图**
   - 用户和电影作为图的节点
   - 评分作为边的权重

2. **应用 PageRank 算法**
   - 计算每个电影节点的重要性分数
   - 分数反映电影在网络中的"受欢迎程度"

3. **结合协同过滤**
   - 使用余弦相似度计算电影之间的相似性
   - 基于用户历史评分推荐相似电影

4. **混合评分**
   ```
   score = cf_weight * cf_score + (1 - cf_weight) * pagerank_score
   ```

### 混合推荐策略

```
final_score = svd_weight * svd_score + (1 - svd_weight) * pagerank_score
```

通过调节权重参数，可以在精确性和多样性之间取得平衡。

## 实验结果

运行 `ablation_study.py` 后，可在 `./figures/` 目录查看：

- `ablation_mae_comparison.png` - MAE 对比柱状图
- `ablation_rmse_comparison.png` - RMSE 对比柱状图
- `pagerank_cf_weight_analysis.png` - CF 权重影响分析
- `hybrid_weight_analysis.png` - 混合权重影响分析
- `ablation_heatmap.png` - 性能热力图
- `ablation_results.csv` - 详细结果数据

## 参考文献

1. **PageRank 算法**：Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank citation ranking: Bringing order to the web.

2. **协同过滤**：Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems.

3. **图推荐系统**：Gori, M., & Pucci, A. (2007). ItemRank: A random-walk based scoring algorithm for recommender systems.

## 注意事项

- PageRank 推荐系统需要构建完整的用户-电影图，计算开销较大
- 建议在训练前对数据进行适当采样（如使用 200 万条评分）
- 混合推荐系统的性能取决于权重参数的选择，建议通过消融实验确定最佳参数
