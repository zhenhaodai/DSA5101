"""
推荐系统消融实验
对比不同推荐算法的性能，包括：
1. SVD 基线 (Baseline)
2. PageRank + 协同过滤
3. 混合推荐系统 (不同权重组合)
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.data_loader import MovieLensLoader
from src.recommender import (SVDRecommender, PageRankRecommender, HybridRecommender,
                             ALSRecommender, ItemKNNRecommender)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def run_ablation_study():
    """执行消融实验"""
    print("=" * 80)
    print("推荐系统消融实验")
    print("=" * 80)

    # ========== 1. 数据加载 ==========
    print("\n[步骤 1/4] 加载数据...")
    print("-" * 80)

    loader = MovieLensLoader(data_dir='./data')

    try:
        # 使用 200 万条评分数据
        loader.load_data(sample_size=2000000)
    except FileNotFoundError:
        print("\n错误：找不到数据文件！请确保数据已下载到 data/ml-20m/ 目录")
        return

    # 预处理数据
    user_movie_matrix, filtered_ratings = loader.preprocess_for_recommendation(
        min_user_ratings=50,
        min_movie_ratings=50
    )

    # 划分训练集和测试集 (80/20)
    train_size = int(len(filtered_ratings) * 0.8)
    train_ratings = filtered_ratings.iloc[:train_size]
    test_ratings = filtered_ratings.iloc[train_size:]

    # 创建训练矩阵
    train_matrix = train_ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        fill_value=0
    )

    print(f"训练集大小: {len(train_ratings)} 条评分")
    print(f"测试集大小: {len(test_ratings)} 条评分")
    print(f"用户-电影矩阵: {train_matrix.shape}")

    # ========== 2. 训练基线模型 (SVD) ==========
    print("\n[步骤 2/6] 训练基线模型 - SVD")
    print("-" * 80)

    svd_recommender = SVDRecommender(n_components=200)  # 提升维度
    svd_recommender.fit(train_matrix)

    # 评估 SVD
    print("\n评估 SVD 基线:")
    svd_metrics = svd_recommender.evaluate(test_ratings)

    # ========== 3. 训练 ALS 模型 ⭐ NEW ==========
    print("\n[步骤 3/6] 训练 ALS 模型 ⭐")
    print("-" * 80)

    als_recommender = ALSRecommender(n_factors=100, n_iterations=10, regularization=0.01)
    als_recommender.fit(train_matrix)

    print("\n评估 ALS:")
    als_metrics = als_recommender.evaluate(test_ratings)

    # ========== 4. 训练 ItemKNN 模型 ⭐ NEW ==========
    print("\n[步骤 4/6] 训练 ItemKNN 模型 ⭐")
    print("-" * 80)

    itemknn_recommender = ItemKNNRecommender(k=30, similarity_metric='cosine')
    itemknn_recommender.fit(train_matrix)

    print("\n评估 ItemKNN:")
    itemknn_metrics = itemknn_recommender.evaluate(test_ratings)

    # ========== 5. 训练 PageRank 模型 ==========
    print("\n[步骤 5/6] 训练 PageRank + 协同过滤模型")
    print("-" * 80)

    # 测试不同的 cf_weight
    pagerank_results = []
    cf_weights = [0.3, 0.5, 0.7]

    for cf_weight in cf_weights:
        print(f"\n测试 PageRank (CF权重={cf_weight}):")
        pr_recommender = PageRankRecommender(alpha=0.85, cf_weight=cf_weight)
        pr_recommender.fit(train_matrix)

        pr_metrics = pr_recommender.evaluate(test_ratings)
        pagerank_results.append({
            'model': f'PageRank (CF={cf_weight})',
            'cf_weight': cf_weight,
            'MAE': pr_metrics['MAE'],
            'RMSE': pr_metrics['RMSE'],
            'n_predictions': pr_metrics['n_predictions']
        })

    # ========== 6. 训练混合模型 ==========
    print("\n[步骤 6/6] 训练混合推荐系统 (SVD + PageRank)")
    print("-" * 80)

    # 使用最佳的 PageRank 配置
    best_pr = PageRankRecommender(alpha=0.85, cf_weight=0.5)
    best_pr.fit(train_matrix)

    # 测试不同的混合权重
    hybrid_results = []
    svd_weights = [0.3, 0.5, 0.7]

    for svd_weight in svd_weights:
        print(f"\n测试混合模型 (SVD权重={svd_weight}):")
        hybrid_recommender = HybridRecommender(
            svd_recommender=svd_recommender,
            pagerank_recommender=best_pr,
            svd_weight=svd_weight
        )

        hybrid_metrics = hybrid_recommender.evaluate(test_ratings)
        hybrid_results.append({
            'model': f'Hybrid (SVD={svd_weight})',
            'svd_weight': svd_weight,
            'MAE': hybrid_metrics['MAE'],
            'RMSE': hybrid_metrics['RMSE'],
            'n_predictions': hybrid_metrics['n_predictions']
        })

    # ========== 5. 汇总结果 ==========
    print("\n\n" + "=" * 80)
    print("消融实验结果")
    print("=" * 80)

    # 创建结果表格
    results = []

    # 添加 SVD 基线
    results.append({
        'model': 'SVD (Baseline)',
        'parameter': '-',
        'MAE': svd_metrics['MAE'],
        'RMSE': svd_metrics['RMSE'],
        'n_predictions': svd_metrics['n_predictions']
    })

    # 添加 ALS 结果 ⭐
    results.append({
        'model': 'ALS ⭐',
        'parameter': 'f=100,i=10',
        'MAE': als_metrics['MAE'],
        'RMSE': als_metrics['RMSE'],
        'n_predictions': als_metrics['n_predictions']
    })

    # 添加 ItemKNN 结果 ⭐
    results.append({
        'model': 'ItemKNN ⭐',
        'parameter': 'k=30',
        'MAE': itemknn_metrics['MAE'],
        'RMSE': itemknn_metrics['RMSE'],
        'n_predictions': itemknn_metrics['n_predictions']
    })

    # 添加 PageRank 结果
    for pr_result in pagerank_results:
        results.append({
            'model': pr_result['model'],
            'parameter': f"CF={pr_result['cf_weight']}",
            'MAE': pr_result['MAE'],
            'RMSE': pr_result['RMSE'],
            'n_predictions': pr_result['n_predictions']
        })

    # 添加混合模型结果
    for hybrid_result in hybrid_results:
        results.append({
            'model': hybrid_result['model'],
            'parameter': f"SVD={hybrid_result['svd_weight']}",
            'MAE': hybrid_result['MAE'],
            'RMSE': hybrid_result['RMSE'],
            'n_predictions': hybrid_result['n_predictions']
        })

    results_df = pd.DataFrame(results)
    print("\n性能对比:")
    print(results_df.to_string(index=False))

    # 找出最佳模型
    best_model = results_df.loc[results_df['RMSE'].idxmin()]
    print("\n" + "-" * 80)
    print(f"最佳模型: {best_model['model']}")
    print(f"参数: {best_model['parameter']}")
    print(f"MAE: {best_model['MAE']:.4f}")
    print(f"RMSE: {best_model['RMSE']:.4f}")

    # ========== 6. 可视化结果 ==========
    print("\n正在生成可视化...")

    os.makedirs('./figures', exist_ok=True)

    # 6.1 MAE 对比柱状图
    plt.figure(figsize=(12, 6))

    colors = ['#2E86AB' if 'Baseline' in m else '#A23B72' if 'PageRank' in m else '#F18F01'
              for m in results_df['model']]

    plt.bar(range(len(results_df)), results_df['MAE'], color=colors, alpha=0.7)
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('MAE (越低越好)', fontsize=12)
    plt.title('消融实验: MAE 对比', fontsize=14, fontweight='bold')
    plt.xticks(range(len(results_df)), results_df['model'], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figures/ablation_mae_comparison.png', dpi=300, bbox_inches='tight')
    print("  - 保存 MAE 对比图: ./figures/ablation_mae_comparison.png")

    # 6.2 RMSE 对比柱状图
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(results_df)), results_df['RMSE'], color=colors, alpha=0.7)
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('RMSE (越低越好)', fontsize=12)
    plt.title('消融实验: RMSE 对比', fontsize=14, fontweight='bold')
    plt.xticks(range(len(results_df)), results_df['model'], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figures/ablation_rmse_comparison.png', dpi=300, bbox_inches='tight')
    print("  - 保存 RMSE 对比图: ./figures/ablation_rmse_comparison.png")

    # 6.3 PageRank CF 权重影响
    plt.figure(figsize=(10, 6))
    pr_df = pd.DataFrame(pagerank_results)
    plt.plot(pr_df['cf_weight'], pr_df['MAE'], marker='o', linewidth=2,
             markersize=8, label='MAE', color='#A23B72')
    plt.plot(pr_df['cf_weight'], pr_df['RMSE'], marker='s', linewidth=2,
             markersize=8, label='RMSE', color='#2E86AB')
    plt.xlabel('CF 权重', fontsize=12)
    plt.ylabel('误差', fontsize=12)
    plt.title('PageRank: CF 权重对性能的影响', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figures/pagerank_cf_weight_analysis.png', dpi=300, bbox_inches='tight')
    print("  - 保存 CF 权重分析图: ./figures/pagerank_cf_weight_analysis.png")

    # 6.4 混合模型权重影响
    plt.figure(figsize=(10, 6))
    hybrid_df = pd.DataFrame(hybrid_results)
    plt.plot(hybrid_df['svd_weight'], hybrid_df['MAE'], marker='o', linewidth=2,
             markersize=8, label='MAE', color='#F18F01')
    plt.plot(hybrid_df['svd_weight'], hybrid_df['RMSE'], marker='s', linewidth=2,
             markersize=8, label='RMSE', color='#2E86AB')
    plt.xlabel('SVD 权重', fontsize=12)
    plt.ylabel('误差', fontsize=12)
    plt.title('混合模型: SVD 权重对性能的影响', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figures/hybrid_weight_analysis.png', dpi=300, bbox_inches='tight')
    print("  - 保存混合权重分析图: ./figures/hybrid_weight_analysis.png")

    # 6.5 综合对比热力图
    plt.figure(figsize=(10, 6))
    comparison_data = results_df[['model', 'MAE', 'RMSE']].copy()
    comparison_data = comparison_data.set_index('model')

    # 归一化数据以便比较
    comparison_normalized = (comparison_data - comparison_data.min()) / (comparison_data.max() - comparison_data.min())

    sns.heatmap(comparison_normalized.T, annot=comparison_data.T.values,
                fmt='.4f', cmap='RdYlGn_r', cbar_kws={'label': '归一化误差'},
                linewidths=0.5)
    plt.title('消融实验: 模型性能热力图', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('指标', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('./figures/ablation_heatmap.png', dpi=300, bbox_inches='tight')
    print("  - 保存性能热力图: ./figures/ablation_heatmap.png")

    # ========== 7. 推荐示例对比 ==========
    print("\n\n" + "=" * 80)
    print("推荐示例对比")
    print("=" * 80)

    # 选择一个样本用户
    sample_user = train_matrix.index[10]
    print(f"\n为用户 {sample_user} 生成推荐:")

    # SVD 推荐
    svd_recs = svd_recommender.recommend_for_user(sample_user, top_n=5)
    print(f"\nSVD 基线推荐:")
    for i, (movie_id, score) in enumerate(svd_recs, 1):
        movie_info = loader.get_movie_info([movie_id])
        if len(movie_info) > 0:
            title = movie_info.iloc[0]['title']
            print(f"  {i}. {title} (分数: {score:.4f})")

    # PageRank 推荐
    pr_recs = best_pr.recommend_for_user(sample_user, top_n=5)
    print(f"\nPageRank + CF 推荐:")
    for i, (movie_id, score) in enumerate(pr_recs, 1):
        movie_info = loader.get_movie_info([movie_id])
        if len(movie_info) > 0:
            title = movie_info.iloc[0]['title']
            print(f"  {i}. {title} (分数: {score:.4f})")

    # 混合推荐
    best_hybrid = HybridRecommender(svd_recommender, best_pr, svd_weight=0.5)
    hybrid_recs = best_hybrid.recommend_for_user(sample_user, top_n=5)
    print(f"\n混合推荐 (SVD=0.5):")
    for i, (movie_id, score) in enumerate(hybrid_recs, 1):
        movie_info = loader.get_movie_info([movie_id])
        if len(movie_info) > 0:
            title = movie_info.iloc[0]['title']
            print(f"  {i}. {title} (分数: {score:.4f})")

    # ========== 8. 保存结果 ==========
    results_df.to_csv('./figures/ablation_results.csv', index=False)
    print("\n\n结果已保存到 ./figures/ablation_results.csv")

    print("\n" + "=" * 80)
    print("消融实验完成！")
    print("=" * 80)

    return results_df


if __name__ == '__main__':
    results = run_ablation_study()
