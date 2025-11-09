"""
基于矩阵分解的协同过滤推荐系统
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import time


class SVDRecommender:
    """基于 SVD 的协同过滤推荐系统"""

    def __init__(self, n_components: int = 50):
        """
        初始化推荐系统

        Args:
            n_components: SVD 分解的维度
        """
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = None
        self.movie_factors = None
        self.user_movie_matrix = None
        self.user_ids = None
        self.movie_ids = None

    def fit(self, user_movie_matrix: pd.DataFrame) -> None:
        """
        训练推荐模型

        Args:
            user_movie_matrix: 用户-电影评分矩阵
        """
        print(f"\n正在训练 SVD 推荐模型 (n_components={self.n_components})...")
        start_time = time.time()

        self.user_movie_matrix = user_movie_matrix
        self.user_ids = user_movie_matrix.index
        self.movie_ids = user_movie_matrix.columns

        # 对用户-电影矩阵进行 SVD 分解
        # U * Sigma * V^T ≈ R
        self.user_factors = self.svd.fit_transform(user_movie_matrix)

        # 获取电影因子矩阵
        self.movie_factors = self.svd.components_.T

        explained_variance = np.sum(self.svd.explained_variance_ratio_)
        print(f"训练完成！用时 {time.time() - start_time:.2f} 秒")
        print(f"解释方差比例: {explained_variance:.4f}")

    def predict_rating(self, user_idx: int, movie_idx: int) -> float:
        """
        预测用户对电影的评分

        Args:
            user_idx: 用户在矩阵中的索引
            movie_idx: 电影在矩阵中的索引

        Returns:
            预测评分
        """
        prediction = np.dot(self.user_factors[user_idx], self.movie_factors[movie_idx])
        # 限制评分范围在 0.5-5.0
        return np.clip(prediction, 0.5, 5.0)

    def recommend_for_user(self, user_id: int, top_n: int = 10,
                          exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        为指定用户推荐电影

        Args:
            user_id: 用户 ID
            top_n: 推荐电影数量
            exclude_rated: 是否排除用户已评分的电影

        Returns:
            推荐电影列表 [(movie_id, predicted_rating), ...]
        """
        if user_id not in self.user_ids:
            raise ValueError(f"用户 {user_id} 不在训练数据中")

        user_idx = self.user_ids.get_loc(user_id)

        # 计算该用户对所有电影的预测评分
        predictions = np.dot(self.user_factors[user_idx], self.movie_factors.T)
        predictions = np.clip(predictions, 0.5, 5.0)

        # 如果需要排除已评分电影
        if exclude_rated:
            rated_mask = self.user_movie_matrix.iloc[user_idx] > 0
            predictions[rated_mask] = -1

        # 获取 top-N 推荐
        top_indices = np.argsort(predictions)[::-1][:top_n]
        recommendations = [
            (self.movie_ids[idx], predictions[idx])
            for idx in top_indices
            if predictions[idx] > 0
        ]

        return recommendations

    def find_similar_movies(self, movie_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        查找相似电影

        Args:
            movie_id: 电影 ID
            top_n: 返回的相似电影数量

        Returns:
            相似电影列表 [(movie_id, similarity), ...]
        """
        if movie_id not in self.movie_ids:
            raise ValueError(f"电影 {movie_id} 不在训练数据中")

        movie_idx = self.movie_ids.get_loc(movie_id)

        # 计算余弦相似度
        movie_vector = self.movie_factors[movie_idx].reshape(1, -1)
        similarities = cosine_similarity(movie_vector, self.movie_factors)[0]

        # 获取最相似的电影（排除自己）
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
        similar_movies = [
            (self.movie_ids[idx], similarities[idx])
            for idx in similar_indices
        ]

        return similar_movies

    def evaluate(self, test_ratings: pd.DataFrame) -> Dict[str, float]:
        """
        评估推荐系统

        Args:
            test_ratings: 测试集评分数据

        Returns:
            评估指标字典
        """
        print("\n正在评估推荐系统...")

        predictions = []
        actuals = []

        for _, row in test_ratings.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']

            try:
                user_idx = self.user_ids.get_loc(user_id)
                movie_idx = self.movie_ids.get_loc(movie_id)
                pred_rating = self.predict_rating(user_idx, movie_idx)

                predictions.append(pred_rating)
                actuals.append(actual_rating)
            except KeyError:
                # 用户或电影不在训练集中
                continue

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # 计算评估指标
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'n_predictions': len(predictions)
        }

        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"评估样本数: {len(predictions)}")

        return metrics


class CollaborativeFiltering:
    """基于相似度的协同过滤（作为对比）"""

    def __init__(self, similarity_type: str = 'cosine'):
        """
        初始化协同过滤

        Args:
            similarity_type: 相似度类型（cosine, pearson）
        """
        self.similarity_type = similarity_type
        self.user_similarity = None
        self.movie_similarity = None
        self.user_movie_matrix = None

    def fit(self, user_movie_matrix: pd.DataFrame) -> None:
        """训练模型"""
        print(f"\n正在计算{self.similarity_type}相似度矩阵...")
        start_time = time.time()

        self.user_movie_matrix = user_movie_matrix

        # 计算用户相似度
        self.user_similarity = cosine_similarity(user_movie_matrix)

        # 计算电影相似度
        self.movie_similarity = cosine_similarity(user_movie_matrix.T)

        print(f"相似度计算完成！用时 {time.time() - start_time:.2f} 秒")

    def recommend_user_based(self, user_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """基于用户的协同过滤推荐"""
        if user_id not in self.user_movie_matrix.index:
            raise ValueError(f"用户 {user_id} 不在训练数据中")

        user_idx = self.user_movie_matrix.index.get_loc(user_id)

        # 获取相似用户
        similar_users = self.user_similarity[user_idx]

        # 计算预测评分（加权平均）
        predictions = np.dot(similar_users, self.user_movie_matrix.values)
        predictions = predictions / (np.sum(np.abs(similar_users)) + 1e-8)

        # 排除已评分电影
        rated_mask = self.user_movie_matrix.iloc[user_idx] > 0
        predictions[rated_mask] = -1

        # 获取 top-N
        top_indices = np.argsort(predictions)[::-1][:top_n]
        recommendations = [
            (self.user_movie_matrix.columns[idx], predictions[idx])
            for idx in top_indices
            if predictions[idx] > 0
        ]

        return recommendations
