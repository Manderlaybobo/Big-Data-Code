# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:12:48 2023

@author: 23170
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 加载数据
data = pd.read_csv('your_data.csv')  # 替换为你的数据文件路径
texts = data['文本内容'].tolist()
labels = data['标签'].tolist()

# 创建分词器
max_words = 10000  # 词汇表的最大词数
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)

# 将文本转换为序列
sequences = tokenizer.texts_to_sequences(texts)
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# 将标签转换为数值
label_mapping = {'积极': 1, '消极': 0}
labels = [label_mapping[label] for label in labels]

# 划分训练集和测试集
split_ratio = 0.8
split_index = int(len(padded_sequences) * split_ratio)

X_train = padded_sequences[:split_index]
y_train = labels[:split_index]
X_test = padded_sequences[split_index:]
y_test = labels[split_index:]

# 构建RNN模型
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_sequence_length),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))

# 使用模型进行情感预测
new_texts = ['这个产品真的很好用', '质量太差，不值这个价']
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
predictions = model.predict(new_padded_sequences)

for i, prediction in enumerate(predictions):
    sentiment = '积极' if prediction >= 0.5 else '消极'
    print(f'文本内容: {new_texts[i]}，情感预测: {sentiment}')
