#!/bin/bash
# MovieLens 20M 数据集下载脚本

echo "正在下载 MovieLens 20M 数据集..."
echo "注意：由于数据集较大 (约 190MB)，下载可能需要一些时间"

# 创建数据目录
mkdir -p data
cd data

# 下载数据集
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip

# 解压
echo "正在解压数据集..."
unzip ml-20m.zip

# 清理
rm ml-20m.zip

echo "数据下载完成！"
echo "数据位置: ./data/ml-20m/"
