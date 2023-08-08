# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:07:32 2023

@author: 23170
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 加载MNIST数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 数据预处理
X_train = np.expand_dims(X_train, axis=-1).astype('float32') / 255.0
X_test = np.expand_dims(X_test, axis=-1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 构建CNN模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, batch_size=64, epochs=5, validation_split=0.2)

# 使用模型预测新的手写数字图片
# 假设你有一张图片数据为new_image，维度为(28, 28)
new_image = np.expand_dims(new_image, axis=-1).astype('float32') / 255.0
predicted_digit = model.predict(np.array([new_image]))
predicted_label = np.argmax(predicted_digit)

print("Predicted Digit:", predicted_label)
