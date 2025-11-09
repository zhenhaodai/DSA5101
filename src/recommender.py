"""
基于矩阵分解的协同过滤推荐系统
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import List, Tuple, Dict
import time
import networkx as nx


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


class PageRankRecommender:
    """基于 PageRank 和协同过滤的图推荐系统"""

    def __init__(self, alpha: float = 0.85, cf_weight: float = 0.5):
        """
        初始化 PageRank 推荐系统

        Args:
            alpha: PageRank 算法的阻尼系数 (通常为 0.85)
            cf_weight: 协同过滤权重 (0-1之间，1-cf_weight 为 PageRank 权重)
        """
        self.alpha = alpha
        self.cf_weight = cf_weight
        self.user_movie_matrix = None
        self.graph = None
        self.pagerank_scores = None
        self.cosine_sim_matrix = None
        self.user_ids = None
        self.movie_ids = None

    def fit(self, user_movie_matrix: pd.DataFrame) -> None:
        """
        训练 PageRank 推荐模型

        Args:
            user_movie_matrix: 用户-电影评分矩阵
        """
        print(f"\n正在训练 PageRank 推荐模型 (alpha={self.alpha}, cf_weight={self.cf_weight})...")
        start_time = time.time()

        self.user_movie_matrix = user_movie_matrix
        self.user_ids = user_movie_matrix.index
        self.movie_ids = user_movie_matrix.columns

        # 1. 计算余弦相似度矩阵（用于协同过滤）
        print("  - 计算余弦相似度矩阵...")
        normalized_data = normalize(user_movie_matrix, axis=0)
        self.cosine_sim_matrix = cosine_similarity(normalized_data.T)
        self.cosine_sim_df = pd.DataFrame(
            self.cosine_sim_matrix,
            index=self.movie_ids,
            columns=self.movie_ids
        )

        # 2. 创建二分图（用户-电影图）
        print("  - 创建用户-电影二分图...")
        self.graph = nx.Graph()

        # 添加用户节点和电影节点
        self.graph.add_nodes_from(self.user_ids, bipartite=0)
        self.graph.add_nodes_from(self.movie_ids, bipartite=1)

        # 添加边（基于评分）
        edge_count = 0
        for user in self.user_ids:
            for movie in self.movie_ids:
                rating = user_movie_matrix.loc[user, movie]
                if rating > 0:  # 只添加有评分的边
                    self.graph.add_edge(user, movie, weight=rating)
                    edge_count += 1

        print(f"    图中包含 {len(self.graph.nodes)} 个节点, {edge_count} 条边")

        # 3. 应用 PageRank 算法
        print("  - 应用 PageRank 算法...")
        pagerank_all = nx.pagerank(self.graph, alpha=self.alpha)

        # 提取电影的 PageRank 分数
        self.pagerank_scores = {
            movie: score
            for movie, score in pagerank_all.items()
            if movie in self.movie_ids
        }

        print(f"训练完成！用时 {time.time() - start_time:.2f} 秒")

    def recommend_for_user(self, user_id: int, top_n: int = 10,
                          exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        为指定用户推荐电影（混合 PageRank 和协同过滤）

        Args:
            user_id: 用户 ID
            top_n: 推荐电影数量
            exclude_rated: 是否排除用户已评分的电影

        Returns:
            推荐电影列表 [(movie_id, combined_score), ...]
        """
        if user_id not in self.user_ids:
            raise ValueError(f"用户 {user_id} 不在训练数据中")

        user_idx = self.user_ids.get_loc(user_id)
        user_ratings = self.user_movie_matrix.loc[user_id]

        # 获取用户已评分的电影
        rated_movies = user_ratings[user_ratings > 0].index.tolist()

        # 1. 协同过滤分数：基于用户评分和电影相似度
        cf_scores = {}
        for movie in self.movie_ids:
            if exclude_rated and movie in rated_movies:
                continue

            # 计算基于已评分电影的相似度加权分数
            similarity_scores = []
            for rated_movie in rated_movies:
                sim = self.cosine_sim_df.loc[movie, rated_movie]
                rating = user_ratings[rated_movie]
                similarity_scores.append(sim * rating)

            if similarity_scores:
                cf_scores[movie] = np.mean(similarity_scores)
            else:
                cf_scores[movie] = 0

        # 2. 归一化协同过滤分数
        if cf_scores:
            max_cf = max(cf_scores.values()) if cf_scores.values() else 1
            if max_cf > 0:
                cf_scores = {k: v / max_cf for k, v in cf_scores.items()}

        # 3. 归一化 PageRank 分数
        max_pr = max(self.pagerank_scores.values()) if self.pagerank_scores else 1
        normalized_pr_scores = {
            k: v / max_pr for k, v in self.pagerank_scores.items()
        }

        # 4. 混合分数：结合协同过滤和 PageRank
        combined_scores = {}
        for movie in self.movie_ids:
            if exclude_rated and movie in rated_movies:
                continue

            cf_score = cf_scores.get(movie, 0)
            pr_score = normalized_pr_scores.get(movie, 0)

            # 加权组合
            combined_score = (
                self.cf_weight * cf_score +
                (1 - self.cf_weight) * pr_score
            )
            combined_scores[movie] = combined_score

        # 5. 排序并返回 top-N
        top_movies = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        return top_movies

    def get_movie_pagerank(self, movie_id: int) -> float:
        """获取指定电影的 PageRank 分数"""
        if movie_id not in self.movie_ids:
            raise ValueError(f"电影 {movie_id} 不在训练数据中")
        return self.pagerank_scores.get(movie_id, 0)

    def evaluate(self, test_ratings: pd.DataFrame) -> Dict[str, float]:
        """
        评估 PageRank 推荐系统

        Args:
            test_ratings: 测试集评分数据

        Returns:
            评估指标字典
        """
        print("\n正在评估 PageRank 推荐系统...")

        predictions = []
        actuals = []

        for _, row in test_ratings.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']

            try:
                # 使用混合分数预测评分（需要转换到评分范围）
                user_idx = self.user_ids.get_loc(user_id)
                user_ratings = self.user_movie_matrix.loc[user_id]
                rated_movies = user_ratings[user_ratings > 0].index.tolist()

                # 计算协同过滤分数
                similarity_scores = []
                for rated_movie in rated_movies:
                    if rated_movie != movie_id and movie_id in self.movie_ids:
                        sim = self.cosine_sim_df.loc[movie_id, rated_movie]
                        rating = user_ratings[rated_movie]
                        similarity_scores.append(sim * rating)

                if similarity_scores:
                    pred_rating = np.mean(similarity_scores)
                    pred_rating = np.clip(pred_rating, 0.5, 5.0)

                    predictions.append(pred_rating)
                    actuals.append(actual_rating)

            except (KeyError, ValueError):
                continue

        if len(predictions) > 0:
            predictions = np.array(predictions)
            actuals = np.array(actuals)

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
        else:
            print("警告：没有足够的数据进行评估")
            return {'MAE': float('inf'), 'RMSE': float('inf'), 'n_predictions': 0}


class HybridRecommender:
    """混合推荐系统：结合 SVD 和 PageRank"""

    def __init__(self, svd_recommender: SVDRecommender,
                 pagerank_recommender: PageRankRecommender,
                 svd_weight: float = 0.5):
        """
        初始化混合推荐系统

        Args:
            svd_recommender: SVD 推荐器
            pagerank_recommender: PageRank 推荐器
            svd_weight: SVD 权重 (0-1之间，1-svd_weight 为 PageRank 权重)
        """
        self.svd_recommender = svd_recommender
        self.pagerank_recommender = pagerank_recommender
        self.svd_weight = svd_weight

    def recommend_for_user(self, user_id: int, top_n: int = 10,
                          exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        为用户生成混合推荐

        Args:
            user_id: 用户 ID
            top_n: 推荐电影数量
            exclude_rated: 是否排除已评分电影

        Returns:
            推荐电影列表 [(movie_id, combined_score), ...]
        """
        # 获取 SVD 推荐
        svd_recs = self.svd_recommender.recommend_for_user(
            user_id, top_n=top_n*2, exclude_rated=exclude_rated
        )
        svd_scores = dict(svd_recs)

        # 获取 PageRank 推荐
        pr_recs = self.pagerank_recommender.recommend_for_user(
            user_id, top_n=top_n*2, exclude_rated=exclude_rated
        )
        pr_scores = dict(pr_recs)

        # 合并所有电影
        all_movies = set(svd_scores.keys()) | set(pr_scores.keys())

        # 归一化分数
        max_svd = max(svd_scores.values()) if svd_scores else 1
        max_pr = max(pr_scores.values()) if pr_scores else 1

        # 计算混合分数
        combined_scores = {}
        for movie in all_movies:
            svd_score = svd_scores.get(movie, 0) / max_svd
            pr_score = pr_scores.get(movie, 0) / max_pr

            combined_score = (
                self.svd_weight * svd_score +
                (1 - self.svd_weight) * pr_score
            )
            combined_scores[movie] = combined_score

        # 排序并返回 top-N
        top_movies = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        return top_movies

    def evaluate(self, test_ratings: pd.DataFrame) -> Dict[str, float]:
        """
        评估混合推荐系统

        Args:
            test_ratings: 测试集评分数据

        Returns:
            评估指标字典
        """
        print(f"\n正在评估混合推荐系统 (SVD权重={self.svd_weight})...")

        predictions = []
        actuals = []

        for _, row in test_ratings.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']

            try:
                # 获取 SVD 预测
                user_idx = self.svd_recommender.user_ids.get_loc(user_id)
                movie_idx = self.svd_recommender.movie_ids.get_loc(movie_id)
                svd_pred = self.svd_recommender.predict_rating(user_idx, movie_idx)

                # 获取 PageRank + CF 预测
                pr_user_idx = self.pagerank_recommender.user_ids.get_loc(user_id)
                user_ratings = self.pagerank_recommender.user_movie_matrix.loc[user_id]
                rated_movies = user_ratings[user_ratings > 0].index.tolist()

                similarity_scores = []
                for rated_movie in rated_movies:
                    if rated_movie != movie_id:
                        sim = self.pagerank_recommender.cosine_sim_df.loc[movie_id, rated_movie]
                        rating = user_ratings[rated_movie]
                        similarity_scores.append(sim * rating)

                if similarity_scores:
                    pr_pred = np.mean(similarity_scores)
                    pr_pred = np.clip(pr_pred, 0.5, 5.0)
                else:
                    pr_pred = 3.0  # 默认评分

                # 混合预测
                hybrid_pred = self.svd_weight * svd_pred + (1 - self.svd_weight) * pr_pred
                hybrid_pred = np.clip(hybrid_pred, 0.5, 5.0)

                predictions.append(hybrid_pred)
                actuals.append(actual_rating)

            except (KeyError, ValueError):
                continue

        if len(predictions) > 0:
            predictions = np.array(predictions)
            actuals = np.array(actuals)

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
        else:
            print("警告：没有足够的数据进行评估")
            return {'MAE': float('inf'), 'RMSE': float('inf'), 'n_predictions': 0}
