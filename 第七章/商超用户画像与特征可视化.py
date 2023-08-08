# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:35:57 2023

@author: 23170
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 加载数据
data = pd.read_csv('your_data.csv')  # 替换为你的数据文件路径

# 提取购买数据和用户属性
#购买数据
X = data.iloc[:, :-4].values
#用户属性：性别、年龄、会员等级、地址（根据实际情况调整数据读取的格式）
attributes = data.iloc[:, -4:].values

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用PCA降维（降维类别设置为2，可根据需要调整）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 使用K-Means聚类
n_clusters = 3  # 设置聚类簇数，可根据实际情况调整
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 将聚类结果合并回原数据
data['Cluster'] = labels

# 可视化展示（可视化展示的属性可以根据实际情况调整）
plt.figure(figsize=(10, 6))
for cluster in range(n_clusters):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['Age'], cluster_data['Gender'], label=f'Cluster {cluster}')
plt.xlabel('Age')
plt.ylabel('Gender')
plt.title('Customer Segmentation by K-Means')
plt.legend()
plt.show()
