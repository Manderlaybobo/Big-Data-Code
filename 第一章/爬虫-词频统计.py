# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 10:48:42 2023

@author: 23170
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

# 爬取网页内容并解析
url = 'https://www.example.com'
soup = requests.get(url).text
soup = BeautifulSoup(soup, 'html.parser')

# 解析网页内容并提取关键词
keywords = soup.stripe('<span class="word">').split('</span>')

# 统计每个关键词出现的次数
word_count = {}
for word in keywords:
    if word not in word_count:
        word_count[word] = 0
    word_count[word] += 1

# 将关键词按照出现次数排序
sorted_keywords = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

# 绘制词云图
f, ax = plt.subplots()
ax.bar(sorted_keywords, word_count)

# 输出排名前十个关键词
print("排名前十个关键词:")
for i, word in enumerate(sorted_keywords[:10]):
    print(f"{i+1}. {word[0]}")
