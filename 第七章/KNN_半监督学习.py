# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:03:20 2023

@author: 23170
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import classification_report

# 加载经典样例Iris数据集
iris = load_iris()
X, y = iris.data, iris.target

# 将数据集分成有标签和无标签部分
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, test_size=0.7, stratify=y, random_state=42)

# 使用Label Propagation对无标签数据打标签
lp_model = LabelPropagation(kernel='knn', n_neighbors=10)#临近数量可以根据需要调整
lp_model.fit(X_labeled, y_labeled)
y_unlabeled_pred = lp_model.predict(X_unlabeled)

# 将打标签的无标签数据加入到有标签数据中
X_combined = np.vstack((X_labeled, X_unlabeled))
y_combined = np.concatenate((y_labeled, y_unlabeled_pred))

# 使用KNN进行分类训练
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_combined, y_combined)

# 对测试集进行预测
y_pred = knn_classifier.predict(X)

# 输出分类报告
print(classification_report(y, y_pred, target_names=iris.target_names))
