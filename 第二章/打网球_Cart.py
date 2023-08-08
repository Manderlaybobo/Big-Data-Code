# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:16:10 2023

@author: 23170
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 创建数据
data = {
    'Temperature(C)': [25, 28, 26, 20, 18, 22, 27, 21, 20, 25],
    'Humidity': [70, 65, 90, 75, 60, 80, 70, 85, 80, 70],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

# 转换数据为DataFrame
df = pd.DataFrame(data)

# 将温度和湿度转换为二进制特征（例如：高温、低温、高湿度、低湿度）
df['Temperature(C)'] = pd.cut(df['Temperature(C)'], bins=[0, 20, 25, 30], labels=['Low', 'Moderate', 'High'])
df['Humidity'] = pd.cut(df['Humidity'], bins=[0, 50, 100], labels=['Low', 'High'])

# 使用独热编码
df_encoded = pd.get_dummies(df, columns=['Temperature(C)', 'Humidity'], drop_first=True)

# 创建决策树分类器
X = df_encoded.drop(columns=['PlayTennis'])
y = df_encoded['PlayTennis']
clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1)
clf.fit(X, y)

# 绘制决策树图
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, rounded=True, impurity=True)
plt.title("Decision Tree with Gini Impurity")
plt.show()

# 输出每个属性划分的基尼系数
for feature, importance in zip(X.columns, clf.feature_importances_):
    print(f"{feature}: Gini Importance = {importance}")


