# -*- coding: utf-8 -*-
# @Time : 2021/3/3 15:12
# @Author : xiaojie
# @File : test_mtcnn_detect.py
# @Software: PyCharm
import cv2
from mtcnn_master.mtcnn import MTCnnDetector

# 测试人脸检测并画出人脸框功能
if __name__ == '__main__':
    m = MTCnnDetector()
    # 设置3个阈值
    threshold = [0.5, 0.6, 0.7]
    img = cv2.imread('tmp/img0.jpg')
    temp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rectangles = m.detectFace(temp_img, threshold)
    draw = img.copy()
    for rectangle in rectangles:
        # 画图边框大小
        W = int(rectangle[2]) - int(rectangle[0])
        H = int(rectangle[3]) - int(rectangle[1])

        # 在原图画出人脸框
        cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (0, 0, 255),
                      2)

        # 画出5个特征点的位置
        for i in range(5, 15, 2):
            cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 1, (255, 0, 0), 4)

    cv2.imwrite("tmp/result0.jpg", draw)
    cv2.imshow("cv2show", draw)
    c = cv2.waitKey(0)
