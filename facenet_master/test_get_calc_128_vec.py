# -*- coding: utf-8 -*-
# @Time : 2021/3/4 16:58
# @Author : xiaojie
# @File : test_get_calc_128_vec.py
# @Software: PyCharm
import time
import cv2
import numpy as np
from config import *
import utils
from inception_resnet_v1 import InceptionResNetV1
from mtcnn_master.mtcnn import MTCnnDetector
import dataset_utils.DBUtils as DBUtils

# 初始化
# 设置3个阈值
threshold = [0.5, 0.7, 0.8]
m = MTCnnDetector(p_net_md_path, r_net_md_path, o_net_md_path)
# 加载FaceNet的inception_v1模型
facenet_model = InceptionResNetV1()
facenet_model.load_weights(model_h5_path)
db = DBUtils.MysqlDatabase()


def save_to_dataset(name, img):
    # 转换图片通道
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 检测人脸
    rectangles = m.detectFace(img, threshold)

    # _img = img.copy()

    # 转化成正方形
    # FaceNet模型的输入是160x160x3的数据
    rectangles = utils.rect2square(np.array(rectangles))

    for rectangle in rectangles:
        # 截取图像返回关键点坐标信息
        landmark = np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])
        crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]

        # 根据双眼特征点进行水平对齐
        crop_img, _ = utils.Alignment_1(crop_img, landmark)

        crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)

        # 提取FaceNet的128个特征值
        pre = utils.calc_128_vec(facenet_model, crop_img)

        print(pre)
        db.saveFace128Vec(data={"name": name, "face128vec": pre.tolist()})
    pass


# 通过摄像头截取图片录入人脸数据
if __name__ == '__main__':
    print("请输入录入的人脸名称...\n")
    name = input()
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, img = video_capture.read()
        # img = cv2.imread("img/zhouxingchi_input.png")
        # 检测人脸
        rectangles = m.detectFace(img, threshold)

        _img = img.copy()

        # 高斯滤波处理
        img = cv2.GaussianBlur(img, (5, 5), 0)  # 可以更改核大小

        for rectangle in rectangles:
            # 画图边框大小
            W = int(rectangle[2]) - int(rectangle[0])
            H = int(rectangle[3]) - int(rectangle[1])

            # 在原图画出人脸框
            cv2.rectangle(_img, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                          (0, 0, 255), 2)

        cv2.imshow('Camera:0', _img)
        # 截图
        if cv2.waitKey(20) & 0xFF == ord('q'):
            # 保存图片
            save_name = "img/{}.jpg".format(time.strftime("%Y%m%d%H%M%S", time.localtime()))
            # cv2.imwrite(save_name, _img)
            cv2.imencode('.jpg', img)[1].tofile(save_name)

            # 保存128个特征点信息到数据库
            save_to_dataset(name, img)

            break

    video_capture.release()
    cv2.destroyAllWindows()
