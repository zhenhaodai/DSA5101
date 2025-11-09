"""
MovieLens 推荐系统和聚类分析主程序

本项目实现了以下算法：
1. 推荐系统：基于 SVD 的协同过滤
2. 聚类算法：K-means 和层次聚类
3. 降维算法：PCA，并分析降维对聚类效果的影响
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.data_loader import MovieLensLoader
from src.recommender import (SVDRecommender, PageRankRecommender, HybridRecommender,
                             ALSRecommender, ItemKNNRecommender)
from src.clustering import MovieClusterer, DimensionalityReducer
from src.visualization import Visualizer
from src.evaluation import RecommenderEvaluator, create_relevance_set
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def evaluate_ranking_quality(recommender, test_ratings, k=10, sample_users=500):
    """
    评估推荐排序质量

    Args:
        recommender: 推荐器对象
        test_ratings: 测试集
        k: Top-K
        sample_users: 采样用户数量

    Returns:
        排序质量指标字典
    """
    # 创建相关项目集合（评分>=4.0视为相关）
    user_relevant = create_relevance_set(test_ratings, threshold=4.0)

    # 采样用户
    sampled_users = list(user_relevant.keys())[:sample_users]

    # 生成推荐
    user_recommendations = {}
    for user_id in sampled_users:
        try:
            recs = recommender.recommend_for_user(user_id, top_n=k, exclude_rated=True)
            user_recommendations[user_id] = [movie_id for movie_id, _ in recs]
        except:
            continue

    # 计算指标
    evaluator = RecommenderEvaluator(k=k)
    metrics = evaluator.evaluate_recommendations(
        user_recommendations,
        {uid: user_relevant[uid] for uid in user_recommendations if uid in user_relevant},
        k=k
    )

    print(f"  Precision@{k}: {metrics[f'Precision@{k}']:.4f}")
    print(f"  Recall@{k}: {metrics[f'Recall@{k}']:.4f}")
    print(f"  NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}")
    print(f"  HitRate@{k}: {metrics[f'HitRate@{k}']:.4f}")

    return metrics


def main():
    """主函数"""
    print("=" * 80)
    print("MovieLens 20M 推荐系统和聚类分析项目")
    print("=" * 80)
    print("\n本项目将依次执行以下分析：")
    print("1. 数据加载和预处理")
    print("2. 推荐系统（SVD 基线 + PageRank + 混合推荐）")
    print("3. 电影聚类分析（K-means vs 层次聚类）")
    print("4. 降维分析（PCA 对聚类效果的影响）")
    print("\n" + "=" * 80)

    # ========== 1. 数据加载 ==========
    print("\n\n[步骤 1/4] 数据加载和预处理")
    print("-" * 80)

    loader = MovieLensLoader(data_dir='./data')

    # 加载数据（使用采样以加快速度，实际使用时可以去掉 sample_size 参数）
    # 注意：如果是第一次运行，请确保已下载 MovieLens 20M 数据集到 data/ml-20m/ 目录
    try:
        # 使用 200 万条评分数据进行演示（完整数据集有 2000 万条）
        loader.load_data(sample_size=2000000)
    except FileNotFoundError:
        print("\n错误：找不到数据文件！")
        print("请按照以下步骤操作：")
        print("1. 从 Kaggle 下载 MovieLens 20M 数据集")
        print("2. 解压到 data/ml-20m/ 目录")
        print("3. 确保存在以下文件：")
        print("   - data/ml-20m/ratings.csv")
        print("   - data/ml-20m/movies.csv")
        print("   - data/ml-20m/genome-scores.csv (可选)")
        return

    # ========== 2. 推荐系统 ==========
    print("\n\n[步骤 2/4] 推荐系统分析")
    print("-" * 80)

    # 预处理数据
    user_movie_matrix, filtered_ratings = loader.preprocess_for_recommendation(
        min_user_ratings=50,
        min_movie_ratings=50
    )

    # 评估推荐系统 - 使用改进的 80/20 分割策略
    # 确保测试集中的用户和电影都在训练集中出现过
    print("\n正在划分训练集和测试集...")

    # 对每个用户，随机选择 20% 的评分作为测试集
    train_list = []
    test_list = []

    for user_id in filtered_ratings['userId'].unique():
        user_ratings = filtered_ratings[filtered_ratings['userId'] == user_id]
        n_test = max(1, int(len(user_ratings) * 0.2))  # 至少1个测试样本

        # 随机打乱
        user_ratings = user_ratings.sample(frac=1, random_state=42)

        test_list.append(user_ratings.iloc[:n_test])
        train_list.append(user_ratings.iloc[n_test:])

    train_ratings = pd.concat(train_list, ignore_index=True)
    test_ratings = pd.concat(test_list, ignore_index=True)

    print(f"训练集: {len(train_ratings)} 条评分")
    print(f"测试集: {len(test_ratings)} 条评分")

    # 创建训练矩阵
    train_matrix = train_ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        fill_value=0
    )

    # 2.1 训练 SVD 基线模型（传统矩阵分解）
    print("\n训练 SVD 基线模型（传统 MF）...")
    svd_recommender = SVDRecommender(n_components=200)
    svd_recommender.fit(train_matrix)
    svd_metrics = svd_recommender.evaluate(test_ratings)

    # 评估排序质量
    print("\n评估 SVD 排序质量 (Top-10):")
    svd_ranking = evaluate_ranking_quality(svd_recommender, test_ratings, k=10, sample_users=500)

    # 2.2 训练 PageRank 推荐模型
    print("\n训练 PageRank + 协同过滤模型...")
    pagerank_recommender = PageRankRecommender(alpha=0.85, cf_weight=0.5)
    pagerank_recommender.fit(train_matrix)
    pr_metrics = pagerank_recommender.evaluate(test_ratings)

    # 评估排序质量
    print("\n评估 PageRank 排序质量 (Top-10):")
    pr_ranking = evaluate_ranking_quality(pagerank_recommender, test_ratings, k=10, sample_users=500)

    # 2.3 训练混合推荐模型
    print("\n训练混合推荐模型 (SVD + PageRank)...")
    hybrid_recommender = HybridRecommender(
        svd_recommender=svd_recommender,
        pagerank_recommender=pagerank_recommender,
        svd_weight=0.5
    )
    hybrid_metrics = hybrid_recommender.evaluate(test_ratings)

    # 评估排序质量
    print("\n评估混合推荐排序质量 (Top-10):")
    hybrid_ranking = evaluate_ranking_quality(hybrid_recommender, test_ratings, k=10, sample_users=500)

    # 2.4 训练 ALS 推荐模型（现代矩阵分解）⭐ NEW
    print("\n训练 ALS 推荐模型（现代 MF）...")
    als_recommender = ALSRecommender(n_factors=100, n_iterations=10, regularization=0.01)
    als_recommender.fit(train_matrix)
    als_metrics = als_recommender.evaluate(test_ratings)

    # 评估排序质量
    print("\n评估 ALS 排序质量 (Top-10):")
    als_ranking = evaluate_ranking_quality(als_recommender, test_ratings, k=10, sample_users=500)

    # 2.5 训练 ItemKNN 推荐模型（传统协同过滤）⭐ NEW
    print("\n训练 ItemKNN 推荐模型（传统 CF - 基于物品）...")
    itemknn_recommender = ItemKNNRecommender(k=30, similarity_metric='cosine')
    itemknn_recommender.fit(train_matrix)
    itemknn_metrics = itemknn_recommender.evaluate(test_ratings)

    # 评估排序质量
    print("\n评估 ItemKNN 排序质量 (Top-10):")
    itemknn_ranking = evaluate_ranking_quality(itemknn_recommender, test_ratings, k=10, sample_users=500)

    # 对比不同推荐系统的性能
    print("\n\n" + "=" * 100)
    print("推荐系统完整评估对比")
    print("=" * 100)

    print("\n【评分预测指标】（越低越好）")
    print("-" * 100)
    rating_comparison = pd.DataFrame({
        '模型': ['SVD (传统MF)', 'ALS (现代MF)', 'ItemKNN (传统CF)', 'PageRank', '混合推荐'],
        'MAE': [svd_metrics['MAE'], als_metrics['MAE'], itemknn_metrics['MAE'],
                pr_metrics['MAE'], hybrid_metrics['MAE']],
        'RMSE': [svd_metrics['RMSE'], als_metrics['RMSE'], itemknn_metrics['RMSE'],
                 pr_metrics['RMSE'], hybrid_metrics['RMSE']]
    })
    print(rating_comparison.to_string(index=False, float_format='%.4f'))

    print("\n\n【排序质量指标】（越高越好）")
    print("-" * 100)
    ranking_comparison = pd.DataFrame({
        '模型': ['SVD (传统MF)', 'ALS (现代MF)', 'ItemKNN (传统CF)', 'PageRank', '混合推荐'],
        'Precision@10': [svd_ranking['Precision@10'], als_ranking['Precision@10'],
                        itemknn_ranking['Precision@10'], pr_ranking['Precision@10'],
                        hybrid_ranking['Precision@10']],
        'Recall@10': [svd_ranking['Recall@10'], als_ranking['Recall@10'],
                     itemknn_ranking['Recall@10'], pr_ranking['Recall@10'],
                     hybrid_ranking['Recall@10']],
        'NDCG@10': [svd_ranking['NDCG@10'], als_ranking['NDCG@10'],
                   itemknn_ranking['NDCG@10'], pr_ranking['NDCG@10'],
                   hybrid_ranking['NDCG@10']],
        'HitRate@10': [svd_ranking['HitRate@10'], als_ranking['HitRate@10'],
                      itemknn_ranking['HitRate@10'], pr_ranking['HitRate@10'],
                      hybrid_ranking['HitRate@10']]
    })
    print(ranking_comparison.to_string(index=False, float_format='%.4f'))
    print("=" * 100)

    # 展示推荐示例对比
    print("\n\n推荐示例对比:")
    print("-" * 80)
    sample_user = train_matrix.index[10]
    print(f"\n为用户 {sample_user} 生成推荐:\n")

    # SVD 推荐
    svd_recs = svd_recommender.recommend_for_user(sample_user, top_n=5)
    print("SVD 基线推荐:")
    for i, (movie_id, score) in enumerate(svd_recs, 1):
        movie_info = loader.get_movie_info([movie_id])
        if len(movie_info) > 0:
            title = movie_info.iloc[0]['title']
            print(f"  {i}. {title} (分数: {score:.4f})")

    # PageRank 推荐
    pr_recs = pagerank_recommender.recommend_for_user(sample_user, top_n=5)
    print("\nPageRank + CF 推荐:")
    for i, (movie_id, score) in enumerate(pr_recs, 1):
        movie_info = loader.get_movie_info([movie_id])
        if len(movie_info) > 0:
            title = movie_info.iloc[0]['title']
            print(f"  {i}. {title} (分数: {score:.4f})")

    # 混合推荐
    hybrid_recs = hybrid_recommender.recommend_for_user(sample_user, top_n=5)
    print("\n混合推荐 (SVD + PageRank):")
    for i, (movie_id, score) in enumerate(hybrid_recs, 1):
        movie_info = loader.get_movie_info([movie_id])
        if len(movie_info) > 0:
            title = movie_info.iloc[0]['title']
            print(f"  {i}. {title} (分数: {score:.4f})")

    # ALS 推荐 ⭐ NEW
    als_recs = als_recommender.recommend_for_user(sample_user, top_n=5)
    print("\nALS 推荐 ⭐:")
    for i, (movie_id, score) in enumerate(als_recs, 1):
        movie_info = loader.get_movie_info([movie_id])
        if len(movie_info) > 0:
            title = movie_info.iloc[0]['title']
            print(f"  {i}. {title} (分数: {score:.4f})")

    # ItemKNN 推荐 ⭐ NEW
    itemknn_recs = itemknn_recommender.recommend_for_user(sample_user, top_n=5)
    print("\nItemKNN 推荐 ⭐:")
    for i, (movie_id, score) in enumerate(itemknn_recs, 1):
        movie_info = loader.get_movie_info([movie_id])
        if len(movie_info) > 0:
            title = movie_info.iloc[0]['title']
            print(f"  {i}. {title} (分数: {score:.4f})")

    # 可视化
    viz = Visualizer(save_dir='./figures')

    # 可视化推荐系统对比
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))
    models = ['SVD\n(基线)', 'ALS\n⭐', 'ItemKNN\n⭐', 'PageRank\n+ CF', '混合推荐']
    mae_values = [svd_metrics['MAE'], als_metrics['MAE'], itemknn_metrics['MAE'],
                  pr_metrics['MAE'], hybrid_metrics['MAE']]
    rmse_values = [svd_metrics['RMSE'], als_metrics['RMSE'], itemknn_metrics['RMSE'],
                   pr_metrics['RMSE'], hybrid_metrics['RMSE']]

    x = np.arange(len(models))
    width = 0.35

    plt.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8, color='#2E86AB')
    plt.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.8, color='#A23B72')

    plt.xlabel('推荐系统', fontsize=12)
    plt.ylabel('误差', fontsize=12)
    plt.title('推荐系统性能对比', fontsize=14, fontweight='bold')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figures/recommender_comparison.png', dpi=300, bbox_inches='tight')
    print("\n\n推荐系统对比图已保存: ./figures/recommender_comparison.png")

    # 使用最佳模型的指标
    metrics = hybrid_metrics if hybrid_metrics['RMSE'] < svd_metrics['RMSE'] else svd_metrics

    # ========== 3. 聚类分析 ==========
    print("\n\n[步骤 3/4] 聚类分析")
    print("-" * 80)

    # 创建电影特征
    movie_features = loader.create_movie_features(use_genome=False)
    print(f"\n使用 {movie_features.shape[0]} 部电影进行聚类")

    # K-means 聚类
    clusterer_kmeans = MovieClusterer()
    labels_kmeans = clusterer_kmeans.kmeans_clustering(movie_features, n_clusters=10)

    # 层次聚类
    clusterer_hierarchical = MovieClusterer()
    labels_hierarchical = clusterer_hierarchical.hierarchical_clustering(
        movie_features, n_clusters=10
    )

    # 比较两种方法
    print("\n聚类方法比较：")
    kmeans_metrics = MovieClusterer.evaluate_clustering(movie_features, labels_kmeans)
    hierarchical_metrics = MovieClusterer.evaluate_clustering(movie_features, labels_hierarchical)

    viz.plot_clustering_comparison({
        'K-means': kmeans_metrics,
        'Hierarchical': hierarchical_metrics
    })

    # 展示聚类统计
    print("\nK-means 聚类统计：")
    cluster_stats = clusterer_kmeans.get_cluster_statistics(loader.movies)
    print(cluster_stats.to_string())

    # 可视化聚类分布
    viz.plot_cluster_distribution(labels_kmeans, 'kmeans_distribution.png')
    viz.plot_pca_2d_clusters(movie_features, labels_kmeans, 'kmeans_2d_visualization.png')

    # 寻找最优 k 值
    print("\n寻找最优 k 值：")
    optimal_k_results = clusterer_kmeans.find_optimal_k(
        movie_features, k_range=range(5, 21)
    )

    # 绘制肘部曲线
    viz.plot_elbow_curve(
        list(optimal_k_results.keys()),
        optimal_k_results,
        metric_name='silhouette',
        save_name='silhouette_elbow.png'
    )

    viz.plot_elbow_curve(
        list(optimal_k_results.keys()),
        optimal_k_results,
        metric_name='davies_bouldin',
        save_name='davies_bouldin_elbow.png'
    )

    # ========== 4. 降维分析 ==========
    print("\n\n[步骤 4/4] 降维分析 (PCA 对聚类效果的影响)")
    print("-" * 80)

    # 使用 PCA 降维
    reducer = DimensionalityReducer(n_components=20)

    # 比较降维前后的聚类效果
    comparison_results = reducer.compare_clustering_with_without_pca(
        movie_features,
        n_clusters=10,
        method='kmeans'
    )

    # 可视化 PCA 方差解释
    viz.plot_pca_variance(comparison_results['pca'])

    # 可视化降维前后的比较
    viz.plot_multi_metrics_comparison(comparison_results)

    # 可视化降维后的聚类
    viz.plot_pca_2d_clusters(
        comparison_results['reduced']['features'],
        comparison_results['reduced']['labels'],
        'pca_reduced_clusters_2d.png'
    )

    # ========== 总结 ==========
    print("\n\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print("\n主要发现：")
    print("\n1. 推荐系统性能：")
    print(f"   - MAE: {metrics['MAE']:.4f}")
    print(f"   - RMSE: {metrics['RMSE']:.4f}")

    print("\n2. 聚类算法比较：")
    print(f"   - K-means 轮廓系数: {kmeans_metrics['silhouette']:.4f}")
    print(f"   - 层次聚类轮廓系数: {hierarchical_metrics['silhouette']:.4f}")

    print("\n3. 降维效果分析：")
    original_silhouette = comparison_results['original']['metrics']['silhouette']
    reduced_silhouette = comparison_results['reduced']['metrics']['silhouette']
    print(f"   - 原始特征轮廓系数: {original_silhouette:.4f}")
    print(f"   - PCA 降维后轮廓系数: {reduced_silhouette:.4f}")

    if reduced_silhouette > original_silhouette:
        print("   - 结论：PCA 降维提升了聚类效果！")
    else:
        print("   - 结论：原始特征聚类效果更好。")

    print("\n所有可视化结果已保存到 ./figures/ 目录")
    print("=" * 80)


if __name__ == '__main__':
    main()
