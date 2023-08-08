# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:03:24 2023

@author: 23170
"""

import numpy as np

# 定义输入数据
X = np.array([[0.1, 0.2]])

# 定义真实标签
y_true = np.array([[0.5]])

# 初始化权重和偏置
np.random.seed(0)
weights = np.random.rand(2, 1)
bias = np.random.rand(1)

# 定义激活函数和其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 定义学习率
learning_rate = 0.1

# 定义迭代次数
epochs = 10000

# 开始训练
for epoch in range(epochs):
    # 前向传播
    z = np.dot(X, weights) + bias
    y_pred = sigmoid(z)
    
    # 计算损失
    error = y_true - y_pred
    
    # 反向传播
    d_loss = error * sigmoid_derivative(y_pred)
    d_weights = np.dot(X.T, d_loss)
    d_bias = np.sum(d_loss)
    
    # 更新权重和偏置
    weights += learning_rate * d_weights
    bias += learning_rate * d_bias
    
    # 打印每1000次迭代的损失
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f'Epoch {epoch}, Loss: {loss:.6f}')

# 输出最终预测结果
print("Final Predicted Output:", y_pred)
