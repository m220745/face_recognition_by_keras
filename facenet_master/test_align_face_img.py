# -*- coding: utf-8 -*-
# @Time : 2021/3/4 14:02
# @Author : xiaojie
# @File : test_align_face_img.py
# @Software: PyCharm

import cv2
import numpy as np
from config import *
import utils
from mtcnn_master.mtcnn import MTCnnDetector

# 测试人脸对齐校正功能
if __name__ == "__main__":
    # 设置3个阈值
    threshold = [0.5, 0.7, 0.8]

    m = MTCnnDetector(p_net_md_path, r_net_md_path, o_net_md_path)

    img = cv2.imread('tmp/zhouxingchi_input.png')
    # 转换图片通道
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 检测人脸
    rectangles = m.detectFace(img, threshold)

    # _img = img.copy()

    # 高斯滤波处理
    img = cv2.GaussianBlur(img, (5, 5), 0)  # 可以更改核大小

    # 转化成正方形
    # FaceNet模型的输入是160x160x3的数据
    rectangles = utils.rect2square(np.array(rectangles))

    for rectangle in rectangles:
        # 截取图像返回关键点坐标信息
        landmark = np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])
        crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]

        cv2.namedWindow("Align Before", 0)
        cv2.resizeWindow("Align Before", 300, 300)
        cv2.imshow("Align Before", cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

        # 根据双眼特征点进行水平对齐
        crop_img, _ = utils.Alignment_1(crop_img, landmark)
        cv2.namedWindow("Align After", 0)
        cv2.resizeWindow("Align After", 300, 300)
        cv2.imshow("Align After", cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
