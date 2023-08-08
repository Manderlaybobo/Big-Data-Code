# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:35:05 2023

@author: 23170
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# 读取泰坦尼克号数据集（请替换为您的数据路径）
data = pd.read_csv('titanic.csv')

# 数据预处理，处理缺失值等
# ...

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = data[features]
y = data['Survived']

# 将非数值属性进行独热编码
X_encoded = pd.get_dummies(X, columns=['Sex', 'Embarked'])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 构建朴素贝叶斯模型
naive_bayes_model = GaussianNB()

# 训练朴素贝叶斯模型
naive_bayes_model.fit(X_train, y_train)

# 预测测试集
y_pred = naive_bayes_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 展示分类报告
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)
