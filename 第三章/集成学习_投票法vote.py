# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:29:03 2023

@author: 23170
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
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

# 构建随机森林模型作为基学习器
random_forest_model1 = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model2 = RandomForestClassifier(n_estimators=100, random_state=42)

# 构建决策树模型作为基学习器
decision_tree_model = DecisionTreeClassifier(random_state=42)

# 使用投票法集成学习
voting_model = VotingClassifier(estimators=[('rf1', random_forest_model1), 
                                            ('rf2', random_forest_model2),
                                            ('dt', decision_tree_model)],
                                voting='hard')
voting_model.fit(X_train, y_train)

# 预测测试集并展示分类报告
y_pred = voting_model.predict(X_test)
classification_rep = classification_report(y_test, y_pred)
print("Voting Classifier - Classification Report:\n", classification_rep)
