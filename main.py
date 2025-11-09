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
from src.recommender import SVDRecommender
from src.clustering import MovieClusterer, DimensionalityReducer
from src.visualization import Visualizer
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def main():
    """主函数"""
    print("=" * 80)
    print("MovieLens 20M 推荐系统和聚类分析项目")
    print("=" * 80)
    print("\n本项目将依次执行以下分析：")
    print("1. 数据加载和预处理")
    print("2. 推荐系统（SVD 协同过滤）")
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

    # 训练推荐模型
    recommender = SVDRecommender(n_components=50)
    recommender.fit(user_movie_matrix)

    # 评估推荐系统
    # 使用 80/20 分割
    train_size = int(len(filtered_ratings) * 0.8)
    train_ratings = filtered_ratings.iloc[:train_size]
    test_ratings = filtered_ratings.iloc[train_size:]

    # 重新训练（使用训练集）
    train_matrix = train_ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        fill_value=0
    )
    recommender.fit(train_matrix)

    # 评估
    metrics = recommender.evaluate(test_ratings)

    # 展示推荐示例
    print("\n推荐示例：")
    sample_users = user_movie_matrix.index[:5]
    for user_id in sample_users:
        recommendations = recommender.recommend_for_user(user_id, top_n=5)
        print(f"\n用户 {user_id} 的推荐电影:")
        for movie_id, score in recommendations[:3]:
            movie_info = loader.get_movie_info([movie_id])
            if len(movie_info) > 0:
                title = movie_info.iloc[0]['title']
                print(f"  - {title} (预测评分: {score:.2f})")

    # 可视化
    viz = Visualizer(save_dir='./figures')
    viz.plot_recommendation_performance({
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE']
    })

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
