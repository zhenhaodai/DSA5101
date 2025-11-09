"""
可视化模块
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import os


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")


class Visualizer:
    """可视化工具类"""

    def __init__(self, save_dir: str = './figures'):
        """
        初始化可视化器

        Args:
            save_dir: 图片保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_elbow_curve(self, k_values: List[int], metrics: Dict[int, Dict],
                        metric_name: str = 'inertia', save_name: str = 'elbow_curve.png'):
        """
        绘制肘部曲线

        Args:
            k_values: k 值列表
            metrics: 每个 k 的指标字典
            metric_name: 要绘制的指标名称
            save_name: 保存文件名
        """
        plt.figure(figsize=(10, 6))

        values = [metrics[k][metric_name] for k in k_values if metrics[k][metric_name] is not None]
        k_valid = [k for k in k_values if metrics[k][metric_name] is not None]

        plt.plot(k_valid, values, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        plt.title(f'{metric_name.replace("_", " ").title()} vs Number of Clusters', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
        plt.close()

    def plot_clustering_comparison(self, metrics_dict: Dict[str, Dict],
                                   save_name: str = 'clustering_comparison.png'):
        """
        比较不同聚类方法的性能

        Args:
            metrics_dict: 不同方法的指标字典
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        methods = list(metrics_dict.keys())
        metrics_names = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
        titles = ['Silhouette Score\n(Higher is Better)',
                 'Davies-Bouldin Index\n(Lower is Better)',
                 'Calinski-Harabasz Index\n(Higher is Better)']

        for idx, (metric, title) in enumerate(zip(metrics_names, titles)):
            values = [metrics_dict[method][metric] for method in methods]

            axes[idx].bar(methods, values, alpha=0.7)
            axes[idx].set_title(title, fontsize=12)
            axes[idx].set_ylabel('Score', fontsize=10)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')

            # 添加数值标签
            for i, v in enumerate(values):
                axes[idx].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
        plt.close()

    def plot_pca_variance(self, pca, save_name: str = 'pca_variance.png'):
        """
        绘制 PCA 方差解释图

        Args:
            pca: 拟合好的 PCA 对象
            save_name: 保存文件名
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 方差解释比例
        ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1),
               pca.explained_variance_ratio_, alpha=0.7)
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax1.set_title('Variance Explained by Each Principal Component', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')

        # 累积方差解释比例
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        ax2.plot(range(1, len(cumsum) + 1), cumsum, 'ro-', linewidth=2, markersize=6)
        ax2.axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
        ax2.axhline(y=0.90, color='b', linestyle='--', label='90% Variance')
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
        ax2.set_title('Cumulative Variance Explained', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
        plt.close()

    def plot_cluster_distribution(self, labels: np.ndarray,
                                  save_name: str = 'cluster_distribution.png'):
        """
        绘制聚类分布

        Args:
            labels: 聚类标签
            save_name: 保存文件名
        """
        plt.figure(figsize=(10, 6))

        unique, counts = np.unique(labels, return_counts=True)

        plt.bar(unique, counts, alpha=0.7)
        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Number of Items', fontsize=12)
        plt.title('Distribution of Items Across Clusters', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, (cluster_id, count) in enumerate(zip(unique, counts)):
            plt.text(cluster_id, count, str(count), ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
        plt.close()

    def plot_pca_2d_clusters(self, features: pd.DataFrame, labels: np.ndarray,
                            save_name: str = 'pca_2d_clusters.png'):
        """
        在 2D PCA 空间中可视化聚类

        Args:
            features: 特征矩阵
            labels: 聚类标签
            save_name: 保存文件名
        """
        from sklearn.decomposition import PCA

        # PCA 降维到 2D
        pca_2d = PCA(n_components=2, random_state=42)
        features_2d = pca_2d.fit_transform(features)

        plt.figure(figsize=(10, 8))

        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                            c=labels, cmap='tab10', alpha=0.6, s=30)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        plt.title('Clusters Visualized in 2D PCA Space', fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
        plt.close()

    def plot_recommendation_performance(self, metrics: Dict[str, float],
                                       save_name: str = 'recommendation_performance.png'):
        """
        绘制推荐系统性能

        Args:
            metrics: 评估指标字典
            save_name: 保存文件名
        """
        plt.figure(figsize=(8, 6))

        metric_names = list(metrics.keys())
        values = list(metrics.values())

        colors = ['#ff9999', '#66b3ff']
        bars = plt.bar(metric_names, values, alpha=0.7, color=colors)
        plt.ylabel('Error', fontsize=12)
        plt.title('Recommendation System Performance', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
        plt.close()

    def plot_multi_metrics_comparison(self, results: Dict[str, Dict],
                                     save_name: str = 'pca_comparison.png'):
        """
        比较原始特征和 PCA 降维后的聚类效果

        Args:
            results: 包含原始和降维后结果的字典
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        methods = ['Original Features', 'PCA Reduced']
        metrics_names = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
        titles = ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index']

        original_metrics = results['original']['metrics']
        reduced_metrics = results['reduced']['metrics']

        for idx, (metric, title) in enumerate(zip(metrics_names, titles)):
            values = [original_metrics[metric], reduced_metrics[metric]]

            bars = axes[idx].bar(methods, values, alpha=0.7, color=['#ff9999', '#66b3ff'])
            axes[idx].set_title(title, fontsize=12)
            axes[idx].set_ylabel('Score', fontsize=10)
            axes[idx].tick_params(axis='x', rotation=15)
            axes[idx].grid(True, alpha=0.3, axis='y')

            # 添加数值标签
            for bar, v in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
        plt.close()
