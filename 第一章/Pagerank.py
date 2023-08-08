# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:32:11 2023

@author: 23170
"""

import pandas as pd
import numpy as np

# 读取Excel文件
df = pd.read_excel('Net.xlsx')

# 创建网页名称列表
webpages = df['Net'].tolist()

# 创建链接关系字典
links = {}
for index, row in df.iterrows():
    links[row['Net']] = row['linkto'].split(',')

# 创建初始的PageRank值
num_webpages = len(webpages)
initial_pagerank = 1 / num_webpages
pagerank = {webpage: initial_pagerank for webpage in webpages}

# 迭代计算PageRank
damping_factor = 0.85
num_iterations = 100
for _ in range(num_iterations):
    new_pagerank = {webpage: (1 - damping_factor) / num_webpages for webpage in webpages}
    for webpage, linked_webpages in links.items():
        num_links = len(linked_webpages)
        if num_links > 0:
            link_contribution = damping_factor * pagerank[webpage] / num_links
            for linked_webpage in linked_webpages:
                new_pagerank[linked_webpage] += link_contribution
    pagerank = new_pagerank

# 输出每个网页的PageRank值
for webpage, rank in pagerank.items():
    print(f"{webpage}: PageRank = {rank:.4f}")
