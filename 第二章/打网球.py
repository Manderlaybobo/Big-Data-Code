# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:10:38 2023

@author: 23170
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 示例数据(根据需要修改)
data = {
    'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 将分类变量转换为数值
df_encoded = pd.get_dummies(df, columns=['Weather', 'Temperature', 'Wind', 'Humidity'], drop_first=True)

# 准备训练数据
X = df_encoded.drop('PlayTennis', axis=1)
y = df_encoded['PlayTennis']

# 创建决策树模型
model = DecisionTreeClassifier(criterion='entropy')  # 使用信息熵作为划分标准
model.fit(X, y)

# 可视化决策树
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()

# 输出每个属性划分的信息增益
importances = model.feature_importances_
for col, importance in zip(X.columns, importances):
    print(f"{col}: {importance:.4f}")
