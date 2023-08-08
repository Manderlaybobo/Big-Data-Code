# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 10:47:57 2023

@author: 23170
"""

import pygame
import numpy as np
import math

# 定义屏幕大小和波浪高度
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
WAVELING_HEIGHT = 200

# 初始化pygame
pygame.init()

# 创建屏幕对象
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# 创建波浪背景图
wave_background = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=np.uint8)

# 创建文本对象
wave_text = "                           ^", font=pygame.font.SysFont((65, 30), 30), True, (255, 255, 255))

# 循环,直到用户退出程序
while True:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            break

    # 获取鼠标点击位置
    mouse_pos = pygame.mouse.get_pos()

    # 计算波浪宽度和高度
    wave_width = int(SCREEN_WIDTH / 16)
    wave_height = int(WAVELING_HEIGHT)

    # 绘制背景波浪
    screen.set_alpha(0.5)
    for x in range(0, SCREEN_WIDTH - wave_width, 2):
        for y in range(0, SCREEN_HEIGHT - wave_height, 2):
            wave_background[y, x] = 128

    # 绘制文本
    screen.set_alpha(1)
    wave_text_pos = (SCREEN_WIDTH / 2, 0)
    screen.blit(wave_text, wave_text_pos)

    # 更新显示
    pygame.display.flip()