# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 13:31:27 2023

@author: 23170
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample

# 读取Excel表格
data = pd.read_excel('your_excel_file.xlsx')

# 将非数值属性进行独热编码
data_encoded = pd.get_dummies(data, columns=['Non_numeric_column1', 'Non_numeric_column2'])

# 划分特征和标签
X = data_encoded.drop(columns=['Label_column'])
y = data_encoded['Label_column']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 重抽样处理不平衡数据集
# 重抽样违约用户
defaulters = data_encoded[data_encoded['Label_column'] == 1]
defaulters_resampled = resample(defaulters, replace=True, n_samples=len(defaulters)*19, random_state=42)
# 合并重抽样后的数据集
data_resampled = pd.concat([data_encoded[data_encoded['Label_column'] == 0], defaulters_resampled])

X_resampled = data_resampled.drop(columns=['Label_column'])
y_resampled = data_resampled['Label_column']

# 构建决策树模型，使用成本敏感框架
sample_weights = y_resampled.apply(lambda x: 10 if x == 0 else 100)
clf = DecisionTreeClassifier(criterion='gini', class_weight='balanced')
clf.fit(X_resampled, y_resampled, sample_weight=sample_weights)

# 预测测试集
y_pred = clf.predict(X_test)

# 展示分类报告
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)

# 计算ROI（根据实际情况修改）
def calculate_roi(y_true, y_pred):
    true_positive = sum((y_true == 1) & (y_pred == 1))
    false_positive = sum((y_true == 0) & (y_pred == 1))
    total_cost = false_positive * 100
    total_revenue = true_positive * 10
    roi = (total_revenue - total_cost) / total_cost
    return roi

roi = calculate_roi(y_test, y_pred)
print("ROI:", roi)
