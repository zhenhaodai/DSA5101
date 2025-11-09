"""
推荐系统排序质量评估演示

展示 Precision@K, Recall@K, NDCG@K 等指标
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.data_loader import MovieLensLoader
from src.recommender import SVDRecommender, ALSRecommender, ItemKNNRecommender
from src.evaluation import RecommenderEvaluator, create_relevance_set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 80)
    print("推荐系统排序质量评估")
    print("=" * 80)

    # 1. 加载数据
    print("\n[步骤 1/3] 加载数据...")
    loader = MovieLensLoader(data_dir='./data')

    try:
        loader.load_data(sample_size=2000000)
    except FileNotFoundError:
        print("错误：找不到数据文件！")
        return

    # 预处理
    user_movie_matrix, filtered_ratings = loader.preprocess_for_recommendation(
        min_user_ratings=50,
        min_movie_ratings=50
    )

    # 划分训练测试集
    print("\n正在划分训练集和测试集...")
    train_list = []
    test_list = []

    for user_id in filtered_ratings['userId'].unique():
        user_ratings = filtered_ratings[filtered_ratings['userId'] == user_id]
        n_test = max(1, int(len(user_ratings) * 0.2))
        user_ratings = user_ratings.sample(frac=1, random_state=42)
        test_list.append(user_ratings.iloc[:n_test])
        train_list.append(user_ratings.iloc[n_test:])

    train_ratings = pd.concat(train_list, ignore_index=True)
    test_ratings = pd.concat(test_list, ignore_index=True)

    train_matrix = train_ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        fill_value=0
    )

    print(f"训练集: {len(train_ratings)} 条")
    print(f"测试集: {len(test_ratings)} 条")

    # 2. 训练模型并评估
    print("\n[步骤 2/3] 训练模型并评估排序质量")
    print("-" * 80)

    results = []

    # 2.1 SVD
    print("\n--- SVD 推荐系统 ---")
    svd = SVDRecommender(n_components=200)
    svd.fit(train_matrix)

    # 评分预测指标
    svd_rating_metrics = svd.evaluate(test_ratings)

    # 排序质量指标
    svd_ranking_metrics = evaluate_ranking_quality(
        svd, test_ratings, k=10, sample_users=500
    )

    results.append({
        'Model': 'SVD',
        **svd_rating_metrics,
        **svd_ranking_metrics
    })

    # 2.2 ALS
    print("\n--- ALS 推荐系统 ---")
    als = ALSRecommender(n_factors=100, n_iterations=10)
    als.fit(train_matrix)

    als_rating_metrics = als.evaluate(test_ratings)
    als_ranking_metrics = evaluate_ranking_quality(
        als, test_ratings, k=10, sample_users=500
    )

    results.append({
        'Model': 'ALS',
        **als_rating_metrics,
        **als_ranking_metrics
    })

    # 2.3 ItemKNN
    print("\n--- ItemKNN 推荐系统 ---")
    itemknn = ItemKNNRecommender(k=30)
    itemknn.fit(train_matrix)

    itemknn_rating_metrics = itemknn.evaluate(test_ratings)
    itemknn_ranking_metrics = evaluate_ranking_quality(
        itemknn, test_ratings, k=10, sample_users=500
    )

    results.append({
        'Model': 'ItemKNN',
        **itemknn_rating_metrics,
        **itemknn_ranking_metrics
    })

    # 3. 结果对比
    print("\n[步骤 3/3] 结果对比")
    print("=" * 80)

    results_df = pd.DataFrame(results)

    # 选择关键指标显示
    display_cols = ['Model', 'MAE', 'RMSE', 'Precision@10', 'Recall@10',
                   'NDCG@10', 'HitRate@10', 'MRR']

    print("\n完整评估结果:")
    print(results_df[display_cols].to_string(index=False, float_format='%.4f'))

    # 可视化
    visualize_results(results_df)

    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)

    return results_df


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
    from src.evaluation import RecommenderEvaluator, create_relevance_set

    print(f"  评估 Top-{k} 排序质量 (采样 {sample_users} 用户)...")

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

    print(f"    Precision@{k}: {metrics[f'Precision@{k}']:.4f}")
    print(f"    Recall@{k}: {metrics[f'Recall@{k}']:.4f}")
    print(f"    NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}")

    return metrics


def visualize_results(results_df):
    """可视化评估结果"""
    print("\n生成可视化图表...")

    os.makedirs('./figures', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = results_df['Model']
    x = np.arange(len(models))
    width = 0.25

    # 1. 评分预测指标
    ax = axes[0, 0]
    ax.bar(x - width, results_df['MAE'], width, label='MAE', alpha=0.8)
    ax.bar(x, results_df['RMSE'], width, label='RMSE', alpha=0.8)
    ax.set_xlabel('模型')
    ax.set_ylabel('误差')
    ax.set_title('评分预测指标 (越低越好)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # 2. 排序质量指标
    ax = axes[0, 1]
    ax.bar(x - width, results_df['Precision@10'], width, label='Precision@10', alpha=0.8)
    ax.bar(x, results_df['Recall@10'], width, label='Recall@10', alpha=0.8)
    ax.bar(x + width, results_df['F1@10'], width, label='F1@10', alpha=0.8)
    ax.set_xlabel('模型')
    ax.set_ylabel('分数')
    ax.set_title('Precision & Recall (越高越好)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # 3. NDCG & Hit Rate
    ax = axes[1, 0]
    ax.bar(x - width/2, results_df['NDCG@10'], width, label='NDCG@10', alpha=0.8, color='#2E86AB')
    ax.bar(x + width/2, results_df['HitRate@10'], width, label='HitRate@10', alpha=0.8, color='#A23B72')
    ax.set_xlabel('模型')
    ax.set_ylabel('分数')
    ax.set_title('NDCG & Hit Rate (越高越好)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # 4. MRR & MAP
    ax = axes[1, 1]
    ax.bar(x - width/2, results_df['MRR'], width, label='MRR', alpha=0.8, color='#F18F01')
    ax.bar(x + width/2, results_df['MAP'], width, label='MAP', alpha=0.8, color='#2A9D8F')
    ax.set_xlabel('模型')
    ax.set_ylabel('分数')
    ax.set_title('MRR & MAP (越高越好)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('./figures/ranking_evaluation.png', dpi=300, bbox_inches='tight')
    print("  图表已保存: ./figures/ranking_evaluation.png")


if __name__ == '__main__':
    results = main()
