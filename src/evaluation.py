"""
推荐系统评估指标模块

包含评分预测和排序质量的各类指标
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class RecommenderEvaluator:
    """推荐系统评估器

    支持两类评估：
    1. 评分预测：MAE, RMSE
    2. 排序质量：Precision@K, Recall@K, NDCG@K, Hit Rate, MRR
    """

    def __init__(self, k: int = 10):
        """
        初始化评估器

        Args:
            k: Top-K 推荐数量
        """
        self.k = k

    @staticmethod
    def rating_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
        """
        计算评分预测指标

        Args:
            predictions: 预测评分
            actuals: 真实评分

        Returns:
            包含 MAE 和 RMSE 的字典
        """
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

        return {
            'MAE': mae,
            'RMSE': rmse,
            'n_samples': len(predictions)
        }

    @staticmethod
    def precision_at_k(recommended: List, relevant: Set, k: int = 10) -> float:
        """
        计算 Precision@K

        Precision@K = (推荐列表前K个中相关的数量) / K

        Args:
            recommended: 推荐列表（按分数排序）
            relevant: 相关项目集合（用户真正喜欢的）
            k: 前K个

        Returns:
            Precision@K 值 (0-1)
        """
        recommended_k = recommended[:k]
        relevant_count = len([item for item in recommended_k if item in relevant])
        return relevant_count / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(recommended: List, relevant: Set, k: int = 10) -> float:
        """
        计算 Recall@K

        Recall@K = (推荐列表前K个中相关的数量) / (所有相关项目数量)

        Args:
            recommended: 推荐列表（按分数排序）
            relevant: 相关项目集合
            k: 前K个

        Returns:
            Recall@K 值 (0-1)
        """
        if len(relevant) == 0:
            return 0.0

        recommended_k = recommended[:k]
        relevant_count = len([item for item in recommended_k if item in relevant])
        return relevant_count / len(relevant)

    @staticmethod
    def f1_at_k(recommended: List, relevant: Set, k: int = 10) -> float:
        """
        计算 F1@K

        F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)

        Args:
            recommended: 推荐列表
            relevant: 相关项目集合
            k: 前K个

        Returns:
            F1@K 值 (0-1)
        """
        precision = RecommenderEvaluator.precision_at_k(recommended, relevant, k)
        recall = RecommenderEvaluator.recall_at_k(recommended, relevant, k)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def ndcg_at_k(recommended: List, relevant: Set, k: int = 10) -> float:
        """
        计算 NDCG@K (Normalized Discounted Cumulative Gain)

        NDCG 考虑了推荐位置的重要性，排名越靠前权重越高

        Args:
            recommended: 推荐列表
            relevant: 相关项目集合
            k: 前K个

        Returns:
            NDCG@K 值 (0-1)
        """
        recommended_k = recommended[:k]

        # DCG: 累积折扣增益
        dcg = 0.0
        for i, item in enumerate(recommended_k):
            if item in relevant:
                # 位置 i+1，折扣因子 1/log2(i+2)
                dcg += 1.0 / np.log2(i + 2)

        # IDCG: 理想情况下的 DCG（所有相关项都排在前面）
        idcg = 0.0
        for i in range(min(len(relevant), k)):
            idcg += 1.0 / np.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def hit_rate_at_k(recommended: List, relevant: Set, k: int = 10) -> float:
        """
        计算 Hit Rate@K

        Hit Rate@K = 1 if 至少有一个相关项在前K个推荐中 else 0

        Args:
            recommended: 推荐列表
            relevant: 相关项目集合
            k: 前K个

        Returns:
            1.0 或 0.0
        """
        recommended_k = recommended[:k]
        for item in recommended_k:
            if item in relevant:
                return 1.0
        return 0.0

    @staticmethod
    def mrr(recommended: List, relevant: Set) -> float:
        """
        计算 MRR (Mean Reciprocal Rank)

        MRR = 1 / (第一个相关项的位置)

        Args:
            recommended: 推荐列表
            relevant: 相关项目集合

        Returns:
            MRR 值
        """
        for i, item in enumerate(recommended):
            if item in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def average_precision(recommended: List, relevant: Set) -> float:
        """
        计算 AP (Average Precision)

        AP = (所有相关位置的Precision之和) / 相关项总数

        Args:
            recommended: 推荐列表
            relevant: 相关项目集合

        Returns:
            AP 值
        """
        if len(relevant) == 0:
            return 0.0

        precision_sum = 0.0
        relevant_count = 0

        for i, item in enumerate(recommended):
            if item in relevant:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i

        return precision_sum / len(relevant)

    def evaluate_recommendations(self,
                                 user_recommendations: Dict[int, List[int]],
                                 user_relevant_items: Dict[int, Set[int]],
                                 k: int = None) -> Dict[str, float]:
        """
        批量评估推荐结果

        Args:
            user_recommendations: {user_id: [推荐的item_id列表（按分数排序）]}
            user_relevant_items: {user_id: {相关item_id集合}}
            k: Top-K，默认使用初始化时的 k

        Returns:
            各类指标的平均值
        """
        if k is None:
            k = self.k

        precisions = []
        recalls = []
        f1_scores = []
        ndcgs = []
        hit_rates = []
        mrrs = []
        aps = []

        for user_id in user_recommendations:
            if user_id not in user_relevant_items:
                continue

            recommended = user_recommendations[user_id]
            relevant = user_relevant_items[user_id]

            if len(relevant) == 0:
                continue

            precisions.append(self.precision_at_k(recommended, relevant, k))
            recalls.append(self.recall_at_k(recommended, relevant, k))
            f1_scores.append(self.f1_at_k(recommended, relevant, k))
            ndcgs.append(self.ndcg_at_k(recommended, relevant, k))
            hit_rates.append(self.hit_rate_at_k(recommended, relevant, k))
            mrrs.append(self.mrr(recommended, relevant))
            aps.append(self.average_precision(recommended, relevant))

        metrics = {
            f'Precision@{k}': np.mean(precisions) if precisions else 0.0,
            f'Recall@{k}': np.mean(recalls) if recalls else 0.0,
            f'F1@{k}': np.mean(f1_scores) if f1_scores else 0.0,
            f'NDCG@{k}': np.mean(ndcgs) if ndcgs else 0.0,
            f'HitRate@{k}': np.mean(hit_rates) if hit_rates else 0.0,
            'MRR': np.mean(mrrs) if mrrs else 0.0,
            'MAP': np.mean(aps) if aps else 0.0,
            'n_users': len(precisions)
        }

        return metrics

    @staticmethod
    def catalog_coverage(all_recommendations: List[int],
                        catalog_size: int) -> float:
        """
        计算推荐目录覆盖率

        Coverage = (被推荐过的不同物品数) / (总物品数)

        Args:
            all_recommendations: 所有推荐的物品ID列表（可重复）
            catalog_size: 物品目录总大小

        Returns:
            覆盖率 (0-1)
        """
        unique_items = set(all_recommendations)
        return len(unique_items) / catalog_size if catalog_size > 0 else 0.0

    @staticmethod
    def diversity(recommendations: List[List[int]],
                  similarity_matrix: np.ndarray = None) -> float:
        """
        计算推荐多样性

        如果提供相似度矩阵：Diversity = 1 - 平均相似度
        否则：Diversity = 不同物品数 / 总推荐数

        Args:
            recommendations: 推荐列表的列表
            similarity_matrix: 物品相似度矩阵（可选）

        Returns:
            多样性分数
        """
        if similarity_matrix is not None:
            # 基于相似度的多样性
            total_dissimilarity = 0.0
            count = 0

            for rec_list in recommendations:
                for i in range(len(rec_list)):
                    for j in range(i + 1, len(rec_list)):
                        if i < len(similarity_matrix) and j < len(similarity_matrix):
                            dissimilarity = 1 - similarity_matrix[rec_list[i], rec_list[j]]
                            total_dissimilarity += dissimilarity
                            count += 1

            return total_dissimilarity / count if count > 0 else 0.0
        else:
            # 简单多样性：不同物品比例
            all_items = []
            for rec_list in recommendations:
                all_items.extend(rec_list)

            if len(all_items) == 0:
                return 0.0

            return len(set(all_items)) / len(all_items)


def create_relevance_set(test_ratings: pd.DataFrame,
                        threshold: float = 4.0) -> Dict[int, Set[int]]:
    """
    从测试集创建相关项目集合

    Args:
        test_ratings: 测试集 DataFrame (userId, movieId, rating)
        threshold: 评分阈值，>=threshold 视为相关

    Returns:
        {user_id: {相关电影ID集合}}
    """
    relevance = defaultdict(set)

    for _, row in test_ratings.iterrows():
        if row['rating'] >= threshold:
            relevance[row['userId']].add(row['movieId'])

    return dict(relevance)
