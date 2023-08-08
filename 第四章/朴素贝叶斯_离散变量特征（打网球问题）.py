# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:37:18 2023

@author: 23170
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import classification_report

# 打网球问题数据
data = {
    'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

# 创建DataFrame
df = pd.DataFrame(data)

# 将离散变量进行独热编码
df_encoded = pd.get_dummies(df.drop(columns=['PlayTennis']))

# 划分特征和标签
X = df_encoded
y = df['PlayTennis']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建朴素贝叶斯模型
nb_model = CategoricalNB()

# 训练模型
nb_model.fit(X_train, y_train)

# 预测测试集
y_pred = nb_model.predict(X_test)

# 展示分类报告
classification_rep = classification_report(y_test, y_pred)
print("Naive Bayes Classifier - Classification Report:\n", classification_rep)
