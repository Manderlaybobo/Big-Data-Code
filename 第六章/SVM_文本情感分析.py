# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:20:20 2023

@author: 23170
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 加载数据
data = pd.read_csv('your_data.csv')  # 替换为你的数据文件路径
texts = data['文本内容'].tolist()
labels = data['标签'].tolist()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 创建TF-IDF向量化器
tfidf_vectorizer = TfidfVectorizer(max_features=10000)  # 控制词汇表的大小

# 将文本转换为TF-IDF特征向量
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 创建SVM模型并训练
# 使用不同的分类准则和核函数进行尝试
svm_linear = SVC(kernel='linear')  # 线性核
svm_poly = SVC(kernel='poly', degree=3)  # 多项式核
svm_rbf = SVC(kernel='rbf')  # 高斯径向基核

svm_linear.fit(X_train_tfidf, y_train)
svm_poly.fit(X_train_tfidf, y_train)
svm_rbf.fit(X_train_tfidf, y_train)

# 在测试集上进行预测
y_pred_linear = svm_linear.predict(X_test_tfidf)
y_pred_poly = svm_poly.predict(X_test_tfidf)
y_pred_rbf = svm_rbf.predict(X_test_tfidf)

# 输出评估结果
print("Linear Kernel:")
print(classification_report(y_test, y_pred_linear))

print("Polynomial Kernel:")
print(classification_report(y_test, y_pred_poly))

print("RBF Kernel:")
print(classification_report(y_test, y_pred_rbf))
