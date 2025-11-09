"""
MovieLens 数据加载和预处理模块
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class MovieLensLoader:
    """MovieLens 20M 数据集加载器"""

    def __init__(self, data_dir: str = './data'):
        """
        初始化数据加载器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.ratings = None
        self.movies = None
        self.tags = None
        self.genome_scores = None

    def load_data(self, sample_size: Optional[int] = None) -> None:
        """
        加载 MovieLens 数据

        Args:
            sample_size: 如果指定，则只加载部分数据（用于快速测试）
        """
        print("正在加载数据...")

        # 加载评分数据
        ratings_path = os.path.join(self.data_dir, 'ml-20m', 'ratings.csv')
        if sample_size:
            self.ratings = pd.read_csv(ratings_path, nrows=sample_size)
        else:
            self.ratings = pd.read_csv(ratings_path)
        print(f"已加载 {len(self.ratings)} 条评分记录")

        # 加载电影数据
        movies_path = os.path.join(self.data_dir, 'ml-20m', 'movies.csv')
        self.movies = pd.read_csv(movies_path)
        print(f"已加载 {len(self.movies)} 部电影")

        # 加载标签数据（可选）
        try:
            tags_path = os.path.join(self.data_dir, 'ml-20m', 'tags.csv')
            self.tags = pd.read_csv(tags_path)
            print(f"已加载 {len(self.tags)} 条标签")
        except FileNotFoundError:
            print("标签文件未找到，跳过")

        # 加载 genome scores（可选）
        try:
            genome_path = os.path.join(self.data_dir, 'ml-20m', 'genome-scores.csv')
            if sample_size:
                self.genome_scores = pd.read_csv(genome_path, nrows=sample_size*100)
            else:
                self.genome_scores = pd.read_csv(genome_path)
            print(f"已加载 {len(self.genome_scores)} 条 genome scores")
        except FileNotFoundError:
            print("Genome scores 文件未找到，跳过")

    def preprocess_for_recommendation(self,
                                     min_user_ratings: int = 20,
                                     min_movie_ratings: int = 20) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        为推荐系统预处理数据

        Args:
            min_user_ratings: 用户最少评分数
            min_movie_ratings: 电影最少评分数

        Returns:
            user_movie_matrix: 用户-电影评分矩阵
            user_movie_sparse: 稀疏矩阵形式
        """
        print("\n正在预处理数据用于推荐系统...")

        # 过滤活跃用户和热门电影
        user_counts = self.ratings['userId'].value_counts()
        movie_counts = self.ratings['movieId'].value_counts()

        active_users = user_counts[user_counts >= min_user_ratings].index
        popular_movies = movie_counts[movie_counts >= min_movie_ratings].index

        filtered_ratings = self.ratings[
            (self.ratings['userId'].isin(active_users)) &
            (self.ratings['movieId'].isin(popular_movies))
        ]

        print(f"过滤后: {len(filtered_ratings)} 条评分, "
              f"{filtered_ratings['userId'].nunique()} 个用户, "
              f"{filtered_ratings['movieId'].nunique()} 部电影")

        # 创建用户-电影矩阵
        user_movie_matrix = filtered_ratings.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )

        return user_movie_matrix, filtered_ratings

    def create_movie_features(self, use_genome: bool = True) -> pd.DataFrame:
        """
        创建电影特征矩阵用于聚类

        Args:
            use_genome: 是否使用 genome scores 作为特征

        Returns:
            movie_features: 电影特征矩阵
        """
        print("\n正在创建电影特征...")

        if use_genome and self.genome_scores is not None:
            # 使用 genome scores 作为特征（更丰富的特征）
            genome_pivot = self.genome_scores.pivot_table(
                index='movieId',
                columns='tagId',
                values='relevance',
                fill_value=0
            )

            # 只保留在 movies 中存在的电影
            common_movies = list(set(genome_pivot.index) & set(self.movies['movieId']))
            movie_features = genome_pivot.loc[common_movies]

            print(f"创建了 {movie_features.shape[0]} 部电影的 {movie_features.shape[1]} 维特征")

        else:
            # 使用类型（genres）作为特征
            print("使用类型作为特征...")

            # 提取所有类型
            all_genres = set()
            for genres in self.movies['genres']:
                if genres != '(no genres listed)':
                    all_genres.update(genres.split('|'))
            all_genres = sorted(list(all_genres))

            # 创建 one-hot 编码
            genre_features = []
            movie_ids = []

            for _, row in self.movies.iterrows():
                genres = row['genres'].split('|')
                feature = [1 if g in genres else 0 for g in all_genres]
                genre_features.append(feature)
                movie_ids.append(row['movieId'])

            movie_features = pd.DataFrame(
                genre_features,
                index=movie_ids,
                columns=all_genres
            )

            print(f"创建了 {movie_features.shape[0]} 部电影的 {movie_features.shape[1]} 维特征（基于类型）")

        return movie_features

    def get_movie_info(self, movie_ids):
        """获取电影信息"""
        return self.movies[self.movies['movieId'].isin(movie_ids)]
