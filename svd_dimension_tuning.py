"""
SVD 维度调优实验
测试不同 n_components 对推荐效果的影响
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.data_loader import MovieLensLoader
from src.recommender import SVDRecommender
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def svd_dimension_tuning():
    """测试不同 SVD 维度的性能"""
    print("=" * 80)
    print("SVD 维度调优实验")
    print("=" * 80)

    # 1. 加载数据
    print("\n[步骤 1/3] 加载数据...")
    print("-" * 80)

    loader = MovieLensLoader(data_dir='./data')

    try:
        loader.load_data(sample_size=2000000)
    except FileNotFoundError:
        print("\n错误：找不到数据文件！")
        return

    # 预处理
    user_movie_matrix, filtered_ratings = loader.preprocess_for_recommendation(
        min_user_ratings=50,
        min_movie_ratings=50
    )

    # 改进的数据分割策略（按用户分割）
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

    print(f"训练集: {len(train_ratings)} 条评分")
    print(f"测试集: {len(test_ratings)} 条评分")

    # 2. 测试不同维度
    print("\n[步骤 2/3] 测试不同 SVD 维度")
    print("-" * 80)

    # 测试的维度范围
    dimensions = [20, 50, 100, 150, 200, 250, 300, 400, 500]
    results = []

    for n_comp in dimensions:
        print(f"\n测试 n_components={n_comp}:")

        try:
            # 训练 SVD
            svd = SVDRecommender(n_components=n_comp)
            svd.fit(train_matrix)

            # 评估
            metrics = svd.evaluate(test_ratings)

            results.append({
                'n_components': n_comp,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'n_predictions': metrics['n_predictions'],
                'explained_variance': np.sum(svd.svd.explained_variance_ratio_)
            })

        except Exception as e:
            print(f"  错误: {e}")
            continue

    # 3. 结果分析
    print("\n\n" + "=" * 80)
    print("实验结果")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    print("\n性能对比:")
    print(results_df.to_string(index=False, float_format='%.4f'))

    # 找到最佳维度
    best_mae = results_df.loc[results_df['MAE'].idxmin()]
    best_rmse = results_df.loc[results_df['RMSE'].idxmin()]

    print("\n" + "-" * 80)
    print(f"最佳 MAE: n_components={int(best_mae['n_components'])}, MAE={best_mae['MAE']:.4f}")
    print(f"最佳 RMSE: n_components={int(best_rmse['n_components'])}, RMSE={best_rmse['RMSE']:.4f}")

    # 4. 可视化
    print("\n[步骤 3/3] 生成可视化图表")
    print("-" * 80)

    os.makedirs('./figures', exist_ok=True)

    # 4.1 MAE vs 维度
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MAE 曲线
    axes[0, 0].plot(results_df['n_components'], results_df['MAE'],
                    marker='o', linewidth=2, markersize=8, color='#E63946')
    axes[0, 0].axvline(x=best_mae['n_components'], color='red',
                       linestyle='--', alpha=0.5, label=f'最佳={int(best_mae["n_components"])}')
    axes[0, 0].set_xlabel('n_components', fontsize=12)
    axes[0, 0].set_ylabel('MAE (越低越好)', fontsize=12)
    axes[0, 0].set_title('SVD 维度 vs MAE', fontsize=14, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    # RMSE 曲线
    axes[0, 1].plot(results_df['n_components'], results_df['RMSE'],
                    marker='s', linewidth=2, markersize=8, color='#457B9D')
    axes[0, 1].axvline(x=best_rmse['n_components'], color='blue',
                       linestyle='--', alpha=0.5, label=f'最佳={int(best_rmse["n_components"])}')
    axes[0, 1].set_xlabel('n_components', fontsize=12)
    axes[0, 1].set_ylabel('RMSE (越低越好)', fontsize=12)
    axes[0, 1].set_title('SVD 维度 vs RMSE', fontsize=14, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    # 解释方差曲线
    axes[1, 0].plot(results_df['n_components'], results_df['explained_variance'] * 100,
                    marker='^', linewidth=2, markersize=8, color='#2A9D8F')
    axes[1, 0].set_xlabel('n_components', fontsize=12)
    axes[1, 0].set_ylabel('解释方差比例 (%)', fontsize=12)
    axes[1, 0].set_title('SVD 维度 vs 解释方差', fontsize=14, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # MAE 改进率
    baseline_mae = results_df[results_df['n_components'] == 50]['MAE'].values[0]
    improvement = ((baseline_mae - results_df['MAE']) / baseline_mae * 100)
    axes[1, 1].bar(results_df['n_components'], improvement, color='#F4A261', alpha=0.7)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].set_xlabel('n_components', fontsize=12)
    axes[1, 1].set_ylabel('相对于 n=50 的改进率 (%)', fontsize=12)
    axes[1, 1].set_title(f'MAE 改进率 (基线: n=50, MAE={baseline_mae:.4f})',
                         fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('./figures/svd_dimension_tuning.png', dpi=300, bbox_inches='tight')
    print("  - 保存图表: ./figures/svd_dimension_tuning.png")

    # 5. 保存结果
    results_df.to_csv('./figures/svd_dimension_results.csv', index=False)
    print("  - 保存数据: ./figures/svd_dimension_results.csv")

    # 6. 推荐建议
    print("\n\n" + "=" * 80)
    print("建议")
    print("=" * 80)

    # 计算性价比最高的配置
    # 定义"性价比" = MAE改进 / (维度增加带来的复杂度)
    baseline_idx = results_df[results_df['n_components'] == 50].index[0]

    for idx, row in results_df.iterrows():
        if idx == baseline_idx:
            continue

        mae_improvement = (baseline_mae - row['MAE']) / baseline_mae * 100
        complexity_increase = (row['n_components'] - 50) / 50 * 100

        if mae_improvement > 5:  # 至少改进 5%
            print(f"\n✅ 推荐使用 n_components={int(row['n_components'])}")
            print(f"   MAE: {row['MAE']:.4f} (改进 {mae_improvement:.1f}%)")
            print(f"   RMSE: {row['RMSE']:.4f}")
            print(f"   解释方差: {row['explained_variance']*100:.1f}%")
            break
    else:
        print(f"\n⚠️ 当前配置 n_components=50 已经不错")
        print(f"   继续增加维度收益不大")

    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)

    return results_df


if __name__ == '__main__':
    results = svd_dimension_tuning()
