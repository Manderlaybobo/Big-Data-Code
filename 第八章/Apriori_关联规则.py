# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:10:03 2023

@author: 23170
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 加载商超POS数据，假设数据文件名为'supermarket_data.csv'
data = pd.read_csv('supermarket_data.csv')

# 数据预处理，将数据进行独热编码
data_encoded = pd.get_dummies(data, columns=data.columns)

# 使用Apriori算法进行关联分析
frequent_itemsets = apriori(data_encoded, min_support=0.05, use_colnames=True)

# 根据关联规则提取有意义的规则
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.2)

# 根据关联规则给出营销意见
for index, row in rules.iterrows():
    antecedent = row['antecedents']
    consequent = row['consequents']
    support = row['support']
    confidence = row['confidence']
    lift = row['lift']
    if len(antecedent) == 1 and len(consequent) == 1:
        antecedent_name = antecedent[0]
        consequent_name = consequent[0]
        if support > 0.1 and confidence > 0.5 and lift > 1.5:#可根据实际需要进行筛选
            print(f"If customers buy '{antecedent_name}', they are likely to buy '{consequent_name}'. "
                  f"Support: {support:.2f}, Confidence: {confidence:.2f}, Lift: {lift:.2f}")
