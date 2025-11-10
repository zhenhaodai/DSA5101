"""
聚类算法模块：K-means 和层次聚类
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Dict, List, Tuple, Optional
import time
from tqdm import tqdm


class MovieClusterer:
    """电影聚类分析器"""

    def __init__(self):
        self.features = None
        self.movie_ids = None
        self.labels = None
        self.method = None
        self.n_clusters = None

    def kmeans_clustering(self, features: pd.DataFrame, n_clusters: int = 10,
                         random_state: int = 42) -> np.ndarray:
        """
        K-means 聚类

        Args:
            features: 特征矩阵
            n_clusters: 聚类数量
            random_state: 随机种子

        Returns:
            聚类标签
        """
        print(f"\n正在执行 K-means 聚类 (k={n_clusters})...")
        start_time = time.time()

        self.features = features
        self.movie_ids = features.index
        self.method = 'kmeans'
        self.n_clusters = n_clusters

        # K-means 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state,
                       n_init=10, max_iter=300)
        self.labels = kmeans.fit_predict(features)

        # 评估聚类质量
        metrics = self.evaluate_clustering(features, self.labels)

        print(f"K-means 聚类完成！用时 {time.time() - start_time:.2f} 秒")
        print(f"轮廓系数: {metrics['silhouette']:.4f}")
        print(f"Davies-Bouldin 指数: {metrics['davies_bouldin']:.4f}")
        print(f"Calinski-Harabasz 指数: {metrics['calinski_harabasz']:.2f}")

        return self.labels

    def hierarchical_clustering(self, features: pd.DataFrame, n_clusters: int = 10,
                               linkage_method: str = 'ward') -> np.ndarray:
        """
        层次聚类

        Args:
            features: 特征矩阵
            n_clusters: 聚类数量
            linkage_method: 连接方法 (ward, complete, average, single)

        Returns:
            聚类标签
        """
        print(f"\n正在执行层次聚类 (n_clusters={n_clusters}, linkage={linkage_method})...")
        start_time = time.time()

        self.features = features
        self.movie_ids = features.index
        self.method = 'hierarchical'
        self.n_clusters = n_clusters

        # 层次聚类
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )
        self.labels = hierarchical.fit_predict(features)

        # 评估聚类质量
        metrics = self.evaluate_clustering(features, self.labels)

        print(f"层次聚类完成！用时 {time.time() - start_time:.2f} 秒")
        print(f"轮廓系数: {metrics['silhouette']:.4f}")
        print(f"Davies-Bouldin 指数: {metrics['davies_bouldin']:.4f}")
        print(f"Calinski-Harabasz 指数: {metrics['calinski_harabasz']:.2f}")

        return self.labels

    @staticmethod
    def evaluate_clustering(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        评估聚类质量

        Args:
            features: 特征矩阵
            labels: 聚类标签

        Returns:
            评估指标字典
        """
        # 轮廓系数（Silhouette Score）：[-1, 1]，越大越好
        silhouette = silhouette_score(features, labels)

        # Davies-Bouldin 指数：越小越好
        davies_bouldin = davies_bouldin_score(features, labels)

        # Calinski-Harabasz 指数：越大越好
        calinski_harabasz = calinski_harabasz_score(features, labels)

        return {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski_harabasz
        }

    def get_cluster_statistics(self, movie_info: pd.DataFrame) -> pd.DataFrame:
        """
        获取聚类统计信息

        Args:
            movie_info: 电影信息

        Returns:
            聚类统计 DataFrame
        """
        if self.labels is None:
            raise ValueError("请先执行聚类")

        cluster_stats = []

        for cluster_id in range(self.n_clusters):
            cluster_mask = self.labels == cluster_id
            cluster_movie_ids = self.movie_ids[cluster_mask]

            # 获取该聚类中的电影信息
            cluster_movies = movie_info[movie_info['movieId'].isin(cluster_movie_ids)]

            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_movie_ids),
                'percentage': len(cluster_movie_ids) / len(self.labels) * 100,
            }

            # 统计类型分布
            if 'genres' in cluster_movies.columns:
                all_genres = []
                for genres in cluster_movies['genres']:
                    if genres != '(no genres listed)':
                        all_genres.extend(genres.split('|'))

                if all_genres:
                    from collections import Counter
                    genre_counts = Counter(all_genres)
                    top_genres = genre_counts.most_common(3)
                    stats['top_genres'] = ', '.join([f"{g}({c})" for g, c in top_genres])
                else:
                    stats['top_genres'] = 'N/A'

            cluster_stats.append(stats)

        return pd.DataFrame(cluster_stats)

    def find_optimal_k(self, features: pd.DataFrame, k_range: range = range(2, 21),
                      method: str = 'kmeans') -> Dict[int, Dict[str, float]]:
        """
        使用肘部法则和轮廓系数找到最优 k 值

        Args:
            features: 特征矩阵
            k_range: k 值范围
            method: 聚类方法

        Returns:
            每个 k 值的评估指标
        """
        print(f"\n正在寻找最优 k 值 (范围: {k_range.start}-{k_range.stop-1})...")

        results = {}

        for k in tqdm(k_range, desc="  测试不同 k 值"):
            if method == 'kmeans':
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                inertia = kmeans.inertia_
            else:
                hierarchical = AgglomerativeClustering(n_clusters=k)
                labels = hierarchical.fit_predict(features)
                inertia = None

            metrics = self.evaluate_clustering(features, labels)
            metrics['inertia'] = inertia

            results[k] = metrics

        return results


class DimensionalityReducer:
    """降维分析器"""

    def __init__(self, n_components: int = 50):
        """
        初始化降维器

        Args:
            n_components: 降维后的维度
        """
        self.n_components = n_components
        self.pca = None
        self.reduced_features = None
        self.original_features = None

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        使用 PCA 降维

        Args:
            features: 原始特征矩阵

        Returns:
            降维后的特征矩阵
        """
        # 自动调整 n_components 不超过特征维度
        n_features = features.shape[1]
        actual_n_components = min(self.n_components, n_features)

        if actual_n_components < self.n_components:
            print(f"\n警告: 特征维度 ({n_features}) 小于指定的 n_components ({self.n_components})")
            print(f"自动调整为 n_components={actual_n_components}")

        print(f"\n正在使用 PCA 降维 (n_components={actual_n_components})...")
        start_time = time.time()

        self.original_features = features
        self.pca = PCA(n_components=actual_n_components, random_state=42)
        reduced_data = self.pca.fit_transform(features)

        self.reduced_features = pd.DataFrame(
            reduced_data,
            index=features.index,
            columns=[f'PC{i+1}' for i in range(actual_n_components)]
        )

        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA 降维完成！用时 {time.time() - start_time:.2f} 秒")
        print(f"原始维度: {features.shape[1]} -> 降维后: {actual_n_components}")
        print(f"解释方差比例: {explained_variance:.4f}")

        return self.reduced_features

    def compare_clustering_with_without_pca(self, original_features: pd.DataFrame,
                                           n_clusters: int = 10,
                                           method: str = 'kmeans') -> Dict[str, Dict]:
        """
        比较降维前后的聚类效果

        Args:
            original_features: 原始特征
            n_clusters: 聚类数量
            method: 聚类方法

        Returns:
            比较结果字典
        """
        print(f"\n正在比较降维前后的聚类效果...")

        # 降维
        reduced_features = self.fit_transform(original_features)

        # 在原始特征上聚类
        clusterer_original = MovieClusterer()
        if method == 'kmeans':
            labels_original = clusterer_original.kmeans_clustering(original_features, n_clusters)
        else:
            labels_original = clusterer_original.hierarchical_clustering(original_features, n_clusters)

        metrics_original = MovieClusterer.evaluate_clustering(original_features, labels_original)

        # 在降维特征上聚类
        clusterer_reduced = MovieClusterer()
        if method == 'kmeans':
            labels_reduced = clusterer_reduced.kmeans_clustering(reduced_features, n_clusters)
        else:
            labels_reduced = clusterer_reduced.hierarchical_clustering(reduced_features, n_clusters)

        metrics_reduced = MovieClusterer.evaluate_clustering(reduced_features, labels_reduced)

        print("\n比较结果:")
        print("=" * 60)
        print(f"{'指标':<25} {'原始特征':<15} {'PCA降维后':<15}")
        print("=" * 60)
        print(f"{'特征维度':<25} {original_features.shape[1]:<15} {reduced_features.shape[1]:<15}")
        print(f"{'轮廓系数 (越大越好)':<25} {metrics_original['silhouette']:<15.4f} {metrics_reduced['silhouette']:<15.4f}")
        print(f"{'Davies-Bouldin (越小越好)':<25} {metrics_original['davies_bouldin']:<15.4f} {metrics_reduced['davies_bouldin']:<15.4f}")
        print(f"{'Calinski-Harabasz (越大越好)':<25} {metrics_original['calinski_harabasz']:<15.2f} {metrics_reduced['calinski_harabasz']:<15.2f}")
        print("=" * 60)

        return {
            'original': {
                'metrics': metrics_original,
                'labels': labels_original,
                'features': original_features
            },
            'reduced': {
                'metrics': metrics_reduced,
                'labels': labels_reduced,
                'features': reduced_features
            },
            'pca': self.pca
        }
