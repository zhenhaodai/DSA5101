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
from tqdm import tqdm


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

        for _, row in tqdm(test_ratings.iterrows(), total=len(test_ratings), desc="  评估进度"):
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
        for user in tqdm(self.user_ids, desc="  - 添加图边"):
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
        优化版本：使用向量化计算提升性能

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
        rated_movies = user_ratings[user_ratings > 0]

        # 1. 协同过滤分数：使用矩阵运算优化
        # 获取已评分电影的相似度矩阵切片和评分
        rated_movie_ids = rated_movies.index
        ratings_array = rated_movies.values

        # 获取所有电影与已评分电影的相似度矩阵
        # similarity_matrix: (所有电影 × 已评分电影)
        similarity_matrix = self.cosine_sim_df.loc[:, rated_movie_ids].values

        # 向量化计算CF分数: (相似度矩阵 × 评分向量) / 已评分电影数
        cf_scores_array = np.dot(similarity_matrix, ratings_array) / len(rated_movies)

        # 2. 归一化协同过滤分数
        max_cf = np.max(cf_scores_array) if cf_scores_array.max() > 0 else 1
        cf_scores_normalized = cf_scores_array / max_cf if max_cf > 0 else cf_scores_array

        # 3. 归一化 PageRank 分数 (转为数组)
        max_pr = max(self.pagerank_scores.values()) if self.pagerank_scores else 1
        pr_scores_array = np.array([self.pagerank_scores.get(mid, 0) for mid in self.movie_ids])
        pr_scores_normalized = pr_scores_array / max_pr if max_pr > 0 else pr_scores_array

        # 4. 混合分数：向量化计算
        combined_scores_array = (
            self.cf_weight * cf_scores_normalized +
            (1 - self.cf_weight) * pr_scores_normalized
        )

        # 5. 排除已评分电影
        if exclude_rated:
            rated_movie_indices = [self.movie_ids.get_loc(mid) for mid in rated_movie_ids]
            combined_scores_array[rated_movie_indices] = -1

        # 6. 获取 top-N
        top_indices = np.argsort(combined_scores_array)[::-1][:top_n]
        top_movies = [
            (self.movie_ids[idx], combined_scores_array[idx])
            for idx in top_indices
            if combined_scores_array[idx] > 0
        ]

        return top_movies

    def get_movie_pagerank(self, movie_id: int) -> float:
        """获取指定电影的 PageRank 分数"""
        if movie_id not in self.movie_ids:
            raise ValueError(f"电影 {movie_id} 不在训练数据中")
        return self.pagerank_scores.get(movie_id, 0)

    def evaluate(self, test_ratings: pd.DataFrame, sample_size: int = 10000) -> Dict[str, float]:
        """
        评估 PageRank 推荐系统（使用采样加速）

        Args:
            test_ratings: 测试集评分数据
            sample_size: 采样大小（默认10000，设为None使用全部数据）

        Returns:
            评估指标字典
        """
        print("\n正在评估 PageRank 推荐系统...")

        # 如果测试集太大，随机采样以加速评估
        if sample_size and len(test_ratings) > sample_size:
            test_sample = test_ratings.sample(n=sample_size, random_state=42)
            print(f"  (使用 {sample_size}/{len(test_ratings)} 条样本进行评估)")
        else:
            test_sample = test_ratings

        predictions = []
        actuals = []

        # 批量处理以提升效率
        print(f"  处理中... ", end='', flush=True)
        processed = 0

        for _, row in test_sample.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']

            try:
                # 快速检查
                if user_id not in self.user_ids or movie_id not in self.movie_ids:
                    continue

                # 获取用户评分（矢量化）
                user_ratings = self.user_movie_matrix.loc[user_id]
                rated_mask = user_ratings > 0
                rated_movies = user_ratings[rated_mask]

                if len(rated_movies) == 0:
                    continue

                # 矢量化计算相似度（关键优化）
                if movie_id in self.cosine_sim_df.index:
                    similarities = self.cosine_sim_df.loc[movie_id, rated_movies.index]
                    # 去除目标电影自身
                    if movie_id in similarities.index:
                        similarities = similarities.drop(movie_id)

                    if len(similarities) > 0:
                        # 加权平均
                        weighted_sum = (similarities * rated_movies[similarities.index]).sum()
                        similarity_sum = similarities.sum()

                        if similarity_sum > 0:
                            pred_rating = weighted_sum / similarity_sum
                        else:
                            pred_rating = rated_movies.mean()

                        pred_rating = np.clip(pred_rating, 0.5, 5.0)

                        predictions.append(pred_rating)
                        actuals.append(actual_rating)

            except (KeyError, ValueError):
                continue

            # 进度提示
            processed += 1
            if processed % 2000 == 0:
                print(f"{processed}...", end='', flush=True)

        print("完成!")

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

class ALSRecommender:
    """基于 ALS (Alternating Least Squares) 的推荐系统

    ALS 是工业界标准的矩阵分解算法，被 Spotify、Netflix 等公司使用
    优势：比 SVD 更快，效果更好，支持隐式反馈
    """

    def __init__(self, n_factors: int = 50, n_iterations: int = 15, regularization: float = 0.01):
        """
        初始化 ALS 推荐系统

        Args:
            n_factors: 潜在因子数量
            n_iterations: 迭代次数
            regularization: 正则化参数
        """
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.user_factors = None
        self.item_factors = None
        self.user_ids = None
        self.movie_ids = None
        self.user_movie_matrix = None

    def fit(self, user_movie_matrix: pd.DataFrame) -> None:
        """
        训练 ALS 模型

        Args:
            user_movie_matrix: 用户-电影评分矩阵
        """
        print(f"\n正在训练 ALS 推荐模型 (factors={self.n_factors}, iterations={self.n_iterations})...")
        start_time = time.time()

        self.user_movie_matrix = user_movie_matrix
        self.user_ids = user_movie_matrix.index
        self.movie_ids = user_movie_matrix.columns

        # 转换为 numpy 数组
        R = user_movie_matrix.values
        n_users, n_items = R.shape

        # 初始化用户和物品因子矩阵
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # ALS 交替优化
        for iteration in tqdm(range(self.n_iterations), desc="  ALS 迭代"):
            # 固定物品因子，更新用户因子
            for u in range(n_users):
                # 找到用户 u 评分过的物品
                rated_items = R[u, :] > 0
                if not np.any(rated_items):
                    continue

                # 提取相关的物品因子和评分
                item_factors_u = self.item_factors[rated_items, :]
                ratings_u = R[u, rated_items]

                # 求解最小二乘问题: min ||ratings_u - user_factor · item_factors_u^T||^2 + reg||user_factor||^2
                A = item_factors_u.T @ item_factors_u + self.regularization * np.eye(self.n_factors)
                b = item_factors_u.T @ ratings_u
                self.user_factors[u, :] = np.linalg.solve(A, b)

            # 固定用户因子，更新物品因子
            for i in range(n_items):
                # 找到评分过物品 i 的用户
                rating_users = R[:, i] > 0
                if not np.any(rating_users):
                    continue

                # 提取相关的用户因子和评分
                user_factors_i = self.user_factors[rating_users, :]
                ratings_i = R[rating_users, i]

                # 求解最小二乘问题
                A = user_factors_i.T @ user_factors_i + self.regularization * np.eye(self.n_factors)
                b = user_factors_i.T @ ratings_i
                self.item_factors[i, :] = np.linalg.solve(A, b)

        print(f"训练完成！用时 {time.time() - start_time:.2f} 秒")

    def predict_rating(self, user_idx: int, movie_idx: int) -> float:
        """预测用户对电影的评分"""
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[movie_idx])
        return np.clip(prediction, 0.5, 5.0)

    def recommend_for_user(self, user_id: int, top_n: int = 10,
                          exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """为指定用户推荐电影"""
        if user_id not in self.user_ids:
            raise ValueError(f"用户 {user_id} 不在训练数据中")

        user_idx = self.user_ids.get_loc(user_id)

        # 计算该用户对所有电影的预测评分
        predictions = np.dot(self.user_factors[user_idx], self.item_factors.T)
        predictions = np.clip(predictions, 0.5, 5.0)

        # 排除已评分电影
        if exclude_rated:
            rated_mask = self.user_movie_matrix.iloc[user_idx] > 0
            predictions[rated_mask] = -1

        # 获取 top-N 推荐
        top_indices = np.argsort(predictions)[::-1][:top_n]
        recommendations = [
            (self.movie_ids[idx], predictions[idx])
            for idx in top_indices
        ]

        return recommendations

    def evaluate(self, test_ratings: pd.DataFrame) -> Dict[str, float]:
        """评估 ALS 推荐系统"""
        print("\n正在评估 ALS 推荐系统...")

        predictions = []
        actuals = []

        for _, row in tqdm(test_ratings.iterrows(), total=len(test_ratings), desc="  评估进度"):
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
                continue

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


class ItemKNNRecommender:
    """基于物品的 K 近邻协同过滤推荐系统

    ItemKNN 是经典的推荐算法，具有以下优势：
    - 可解释性强（基于相似物品推荐）
    - 无需训练（直接计算相似度）
    - 效果稳定
    """

    def __init__(self, k: int = 20, similarity_metric: str = 'cosine'):
        """
        初始化 ItemKNN 推荐系统

        Args:
            k: 近邻数量
            similarity_metric: 相似度度量（cosine, pearson）
        """
        self.k = k
        self.similarity_metric = similarity_metric
        self.item_similarity = None
        self.user_movie_matrix = None
        self.user_ids = None
        self.movie_ids = None

    def fit(self, user_movie_matrix: pd.DataFrame) -> None:
        """
        训练 ItemKNN 模型（计算物品相似度矩阵）

        Args:
            user_movie_matrix: 用户-电影评分矩阵
        """
        print(f"\n正在训练 ItemKNN 推荐模型 (k={self.k}, metric={self.similarity_metric})...")
        start_time = time.time()

        self.user_movie_matrix = user_movie_matrix
        self.user_ids = user_movie_matrix.index
        self.movie_ids = user_movie_matrix.columns

        # 计算物品-物品相似度矩阵
        if self.similarity_metric == 'cosine':
            # 使用余弦相似度
            self.item_similarity = cosine_similarity(user_movie_matrix.T)
        elif self.similarity_metric == 'pearson':
            # 使用皮尔逊相关系数
            self.item_similarity = np.corrcoef(user_movie_matrix.T)
            # 处理 NaN
            self.item_similarity = np.nan_to_num(self.item_similarity, 0)
        else:
            raise ValueError(f"不支持的相似度度量: {self.similarity_metric}")

        # 转换为 DataFrame 方便索引
        self.item_similarity_df = pd.DataFrame(
            self.item_similarity,
            index=self.movie_ids,
            columns=self.movie_ids
        )

        print(f"训练完成！用时 {time.time() - start_time:.2f} 秒")
        print(f"物品相似度矩阵: {self.item_similarity.shape}")

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """预测用户对电影的评分"""
        if user_id not in self.user_ids or movie_id not in self.movie_ids:
            return 2.5  # 返回平均评分

        # 获取用户评分过的电影
        user_ratings = self.user_movie_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0]

        if len(rated_movies) == 0:
            return 2.5

        # 获取与目标电影最相似的 k 个电影
        similarities = self.item_similarity_df.loc[movie_id, rated_movies.index]

        # 选择 top-k 相似电影
        top_k_similar = similarities.nlargest(self.k)

        if len(top_k_similar) == 0 or top_k_similar.sum() == 0:
            return 2.5

        # 加权平均预测
        weighted_sum = 0
        similarity_sum = 0

        for similar_movie, similarity in top_k_similar.items():
            if similarity > 0:
                rating = rated_movies[similar_movie]
                weighted_sum += similarity * rating
                similarity_sum += similarity

        if similarity_sum > 0:
            prediction = weighted_sum / similarity_sum
        else:
            prediction = 2.5

        return np.clip(prediction, 0.5, 5.0)

    def recommend_for_user(self, user_id: int, top_n: int = 10,
                          exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """为指定用户推荐电影"""
        if user_id not in self.user_ids:
            raise ValueError(f"用户 {user_id} 不在训练数据中")

        # 获取用户评分过的电影
        user_ratings = self.user_movie_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0]

        # 为所有未评分电影预测评分
        predictions = {}
        for movie_id in self.movie_ids:
            if exclude_rated and movie_id in rated_movies.index:
                continue

            pred_rating = self.predict_rating(user_id, movie_id)
            predictions[movie_id] = pred_rating

        # 排序并返回 top-N
        top_movies = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        return top_movies

    def evaluate(self, test_ratings: pd.DataFrame, sample_size: int = 10000) -> Dict[str, float]:
        """
        评估 ItemKNN 推荐系统（使用采样加速）

        Args:
            test_ratings: 测试集评分数据
            sample_size: 采样大小（默认10000，设为None使用全部数据）

        Returns:
            评估指标字典
        """
        print("\n正在评估 ItemKNN 推荐系统...")

        # 采样加速
        if sample_size and len(test_ratings) > sample_size:
            test_sample = test_ratings.sample(n=sample_size, random_state=42)
            print(f"  (使用 {sample_size}/{len(test_ratings)} 条样本进行评估)")
        else:
            test_sample = test_ratings

        predictions = []
        actuals = []

        print(f"  处理中... ", end='', flush=True)
        processed = 0

        for _, row in test_sample.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']

            try:
                pred_rating = self.predict_rating(user_id, movie_id)
                predictions.append(pred_rating)
                actuals.append(actual_rating)
            except (KeyError, ValueError):
                continue

            # 进度提示
            processed += 1
            if processed % 2000 == 0:
                print(f"{processed}...", end='', flush=True)

        print("完成!")

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


class BaseRecommender:
    """推荐器基类 - 提供统一的评估接口"""

    def evaluate_ranking(self, test_ratings: pd.DataFrame, 
                        k: int = 10, 
                        relevance_threshold: float = 4.0,
                        sample_size: int = 1000) -> Dict[str, float]:
        """
        评估推荐排序质量

        Args:
            test_ratings: 测试集
            k: Top-K 推荐数量
            relevance_threshold: 评分>=threshold视为相关
            sample_size: 采样用户数量（避免太慢）

        Returns:
            Precision@K, Recall@K, NDCG@K, Hit Rate, MRR, MAP
        """
        from src.evaluation import RecommenderEvaluator, create_relevance_set

        print(f"\n正在评估排序质量 (Top-{k})...")

        # 创建相关项目集合
        user_relevant = create_relevance_set(test_ratings, threshold=relevance_threshold)

        # 采样用户（避免太慢）
        sampled_users = list(user_relevant.keys())[:sample_size]
        print(f"  采样 {len(sampled_users)} 个用户进行评估")

        # 为每个用户生成推荐
        user_recommendations = {}
        processed = 0

        for user_id in sampled_users:
            try:
                # 生成 Top-K 推荐
                recommendations = self.recommend_for_user(user_id, top_n=k, exclude_rated=True)
                # 提取电影ID列表
                user_recommendations[user_id] = [movie_id for movie_id, _ in recommendations]

                processed += 1
                if processed % 200 == 0:
                    print(f"  已处理 {processed}/{len(sampled_users)} 个用户...")

            except (KeyError, ValueError):
                continue

        print(f"  完成！共评估 {len(user_recommendations)} 个用户")

        # 计算指标
        evaluator = RecommenderEvaluator(k=k)
        metrics = evaluator.evaluate_recommendations(
            user_recommendations,
            {uid: user_relevant[uid] for uid in user_recommendations if uid in user_relevant},
            k=k
        )

        # 打印结果
        print(f"\n排序质量指标:")
        print(f"  Precision@{k}: {metrics[f'Precision@{k}']:.4f}")
        print(f"  Recall@{k}: {metrics[f'Recall@{k}']:.4f}")
        print(f"  F1@{k}: {metrics[f'F1@{k}']:.4f}")
        print(f"  NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}")
        print(f"  HitRate@{k}: {metrics[f'HitRate@{k}']:.4f}")
        print(f"  MRR: {metrics['MRR']:.4f}")
        print(f"  MAP: {metrics['MAP']:.4f}")

        return metrics
