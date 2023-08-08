# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:17:16 2023

@author: 23170
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 载入数据集
data = load_iris()
X = data.data
y = data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用预剪枝
pre_pruned_model = DecisionTreeClassifier(max_depth=3)
pre_pruned_model.fit(X_train, y_train)

# 使用后剪枝
post_pruned_model = DecisionTreeClassifier()
post_pruned_model.fit(X_train, y_train)
post_pruned_model.cost_complexity_pruning_path(X_train, y_train)

# 预测并评估模型
y_pred_pre = pre_pruned_model.predict(X_test)
accuracy_pre = accuracy_score(y_test, y_pred_pre)

y_pred_post = post_pruned_model.predict(X_test)
accuracy_post = accuracy_score(y_test, y_pred_post)

print("Accuracy with Pre-Pruning:", accuracy_pre)
print("Accuracy with Post-Pruning:", accuracy_post)
