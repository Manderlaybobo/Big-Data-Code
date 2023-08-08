#%%%%预剪枝
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 创建数据
data = {
    'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'TimeOfDay': ['Morning', 'Evening', 'Morning', 'Afternoon', 'Morning', 'Evening', 'Afternoon', 'Morning', 'Evening', 'Afternoon'],
    'DayOfWeek': ['Weekday', 'Weekend', 'Weekend', 'Weekday', 'Weekday', 'Weekend', 'Weekday', 'Weekend', 'Weekend', 'Weekday'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

# 转换数据为DataFrame
df = pd.DataFrame(data)

# 使用独热编码
df_encoded = pd.get_dummies(df, columns=['Weather', 'Temperature', 'Wind', 'Humidity', 'TimeOfDay', 'DayOfWeek'], drop_first=True)

# 创建预剪枝的决策树分类器
X = df_encoded.drop(columns=['PlayTennis'])
y = df_encoded['PlayTennis']
clf_prepruning = DecisionTreeClassifier(criterion='gini', max_depth=3)  # 设置最大深度进行预剪枝
clf_prepruning.fit(X, y)

# 绘制预剪枝的决策树图
plt.figure(figsize=(12, 8))
plot_tree(clf_prepruning, feature_names=X.columns, class_names=clf_prepruning.classes_, filled=True, rounded=True, impurity=True)
plt.title("Decision Tree with Pre-pruning")
plt.show()

#%%%%后剪枝
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 创建数据
data = {
    'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'TimeOfDay': ['Morning', 'Evening', 'Morning', 'Afternoon', 'Morning', 'Evening', 'Afternoon', 'Morning', 'Evening', 'Afternoon'],
    'DayOfWeek': ['Weekday', 'Weekend', 'Weekend', 'Weekday', 'Weekday', 'Weekend', 'Weekday', 'Weekend', 'Weekend', 'Weekday'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

# 转换数据为DataFrame
df = pd.DataFrame(data)

# 使用独热编码
df_encoded = pd.get_dummies(df, columns=['Weather', 'Temperature', 'Wind', 'Humidity', 'TimeOfDay', 'DayOfWeek'], drop_first=True)

# 创建决策树分类器
X = df_encoded.drop(columns=['PlayTennis'])
y = df_encoded['PlayTennis']
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(X, y)

# 后剪枝
from sklearn.tree import _tree

def prune_index(tree, index, threshold):
    if tree.children_left[index] != _tree.TREE_LEAF:
        prune_index(tree, tree.children_left[index], threshold)
        prune_index(tree, tree.children_right[index], threshold)

        left_child = tree.children_left[index]
        right_child = tree.children_right[index]

        if tree.value[left_child][0, 1] / (tree.value[left_child][0, 0] + tree.value[left_child][0, 1]) < threshold and \
           tree.value[right_child][0, 1] / (tree.value[right_child][0, 0] + tree.value[right_child][0, 1]) < threshold:
            tree.children_left[index] = _tree.TREE_LEAF
            tree.children_right[index] = _tree.TREE_LEAF

prune_index(clf.tree_, 0, 0.2)  # prune with threshold

# 绘制后剪枝的决策树图
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, rounded=True, impurity=True)
plt.title("Decision Tree with Post-pruning")
plt.show()

