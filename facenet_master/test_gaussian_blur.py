# -*- coding: utf-8 -*-
# @Time : 2021/3/11 13:25
# @Author : xiaojie
# @File : test_gaussian_blur.py
# @Software: PyCharm

# encoding:utf-8
import cv2

# 读取图片
img = cv2.imread('tmp/pig.jpg')
source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 高斯滤波
result = cv2.GaussianBlur(source, (7, 7), 0)  # 可以更改核大小

cv2.imshow("Before", cv2.cvtColor(source, cv2.COLOR_RGB2BGR))
cv2.imshow("After", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

