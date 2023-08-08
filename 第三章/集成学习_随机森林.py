# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:12:38 2023

@author: 23170
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 读取Excel表格
data = pd.read_excel('your_excel_file.xlsx')

# 将非数值属性进行独热编码
data_encoded = pd.get_dummies(data, columns=['Non_numeric_column1', 'Non_numeric_column2'])

# 划分特征和标签
X = data_encoded.drop(columns=['Label_column'])
y = data_encoded['Label_column']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建随机森林模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 预测测试集
y_pred = clf.predict(X_test)

# 展示分类报告
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)
