#%%a类缺失训练样本集中有些样本缺少一部分属性
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 创建数据，包括缺失值
data = {
    'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', np.nan, 'Overcast', 'Sunny', 'Sunny', 'Rainy'],
    'Temperature_C': [30, 27, 28, 21, np.nan, 18, 17, 22, 21, 24],  # 使用摄氏度
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
    'Humidity': [85, 90, 86, np.nan, 80, 70, 65, 95, 70, 80],
    'TimeOfDay': ['Morning', 'Evening', 'Morning', 'Afternoon', 'Morning', 'Evening', 'Afternoon', 'Morning', 'Evening', 'Afternoon'],
    'DayOfWeek': ['Weekday', 'Weekend', 'Weekend', 'Weekday', 'Weekday', 'Weekend', 'Weekday', 'Weekend', 'Weekend', 'Weekday'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

# 转换数据为DataFrame
df = pd.DataFrame(data)

# 将连续变量离散化
df['Temperature'] = pd.cut(df['Temperature_C'], bins=[0, 20, 25, 100], labels=['Low', 'Moderate', 'High'])  # 使用标签
df['Humidity'] = pd.cut(df['Humidity'], bins=[0, 70, 90, 100], labels=['Low', 'Moderate', 'High'])  # 使用标签

# 使用标签编码
label_encoder = LabelEncoder()
df_encoded = df.apply(label_encoder.fit_transform)

# 填充缺失值
df_encoded.fillna(-1, inplace=True)

# 创建决策树分类器
X = df_encoded.drop(columns=['PlayTennis'])
y = df_encoded['PlayTennis']
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# 绘制决策树图
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns.astype(str), class_names=label_encoder.classes_, filled=True, rounded=True, impurity=True)
plt.title("Decision Tree with Missing Values as Special Category")
plt.show()


#%%b类缺失. 对于已经选择某属性作为分裂属性，但某些样本缺少该属性值的情况，可以考虑将这些样本分配到占比较大的子树中。
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 创建数据，包括缺失值
data = {
    'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', np.nan, 'Overcast', 'Sunny', 'Sunny', 'Rainy'],
    'Temperature_C': [30, 27, 28, 21, np.nan, 18, 17, 22, 21, 24],  # 使用摄氏度
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
    'Humidity': [85, 90, 86, np.nan, 80, 70, 65, 95, 70, 80],
    'TimeOfDay': ['Morning', 'Evening', 'Morning', 'Afternoon', 'Morning', 'Evening', 'Afternoon', 'Morning', 'Evening', 'Afternoon'],
    'DayOfWeek': ['Weekday', 'Weekend', 'Weekend', 'Weekday', 'Weekday', 'Weekend', 'Weekday', 'Weekend', 'Weekend', 'Weekday'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

# 转换数据为DataFrame
df = pd.DataFrame(data)

# 将连续变量离散化
df['Temperature'] = pd.cut(df['Temperature_C'], bins=[0, 20, 25, 100], labels=['Low', 'Moderate', 'High'])  # 使用标签
df['Humidity'] = pd.cut(df['Humidity'], bins=[0, 70, 90, 100], labels=['Low', 'Moderate', 'High'])  # 使用标签

# 使用标签编码
label_encoder = LabelEncoder()
df_encoded = df.apply(label_encoder.fit_transform)

# 填充缺失值
df_encoded.fillna(-1, inplace=True)

# 创建决策树分类器
X = df_encoded.drop(columns=['PlayTennis'])
y = df_encoded['PlayTennis']
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# 对于已选择的属性分支，处理缺失值样本的分配
def assign_samples_to_majority_branch(node, sample):
    if node.feature == -2:  # 叶子节点
        return node.value.argmax()
    feature_val = sample[node.feature]
    if feature_val == -1:  # 属性值缺失
        if node.value[0, 1] > node.value[0, 0]:
            return assign_samples_to_majority_branch(node.children_left, sample)
        else:
            return assign_samples_to_majority_branch(node.children_right, sample)
    if feature_val <= node.threshold:
        return assign_samples_to_majority_branch(node.children_left, sample)
    else:
        return assign_samples_to_majority_branch(node.children_right, sample)

# 对缺失值样本进行预测
missing_samples = X[X['Temperature'] == -1]
for index, row in missing_samples.iterrows():
    prediction = assign_samples_to_majority_branch(clf.tree_, row)
    print(f"Predicted class for missing sample {index}: {label_encoder.inverse_transform([prediction])[0]}")
















