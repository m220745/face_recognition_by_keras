# -*- coding: utf-8 -*-
# @Time : 2021/3/4 15:52
# @Author : xiaojie
# @File : face_model.py
# @Software: PyCharm

import os, sys

# 获取当前路径
rootpath = os.path.dirname(os.path.realpath(sys.argv[0]))

print(rootpath)
import cv2
import numpy as np
from config import *
import utils
from facenet_master.inception_resnet_v1 import InceptionResNetV1
from mtcnn_master.mtcnn import MTCnnDetector
import dataset_utils.DBUtils as DBUtils

if not os.path.exists(p_net_md_path):
    p_net_md_path = rootpath + "/test_model/Pnet.h5"
    r_net_md_path = rootpath + "/test_model/Rnet.h5"
    o_net_md_path = rootpath + "/test_model/Onet.h5"
    model_h5_path = rootpath + "/test_model/facenet_keras.h5"


class Model():
    def __init__(self):
        # 初始化基本参数

        # 初始化mtcnn的人脸检测模型
        print("2 p_net_md_path:", p_net_md_path)
        self.mtcnn_detector = MTCnnDetector(p_net_md_path, r_net_md_path, o_net_md_path)
        # 设置3个阈值（用于3个mtcnn网络的解码过程）
        self.threshold = [0.5, 0.6, 0.8]

        # 加载FaceNet的inception_v1模型，检测128个特征向量
        self.facenet_model = InceptionResNetV1()

        # 加载预训练好的模型权重信息
        self.facenet_model.load_weights(model_h5_path)

        self.db = DBUtils.MysqlDatabase()
        global known_face
        self.known_face = known_face = self.db.getKnownFaceToDict()

    def draw_face(self, img):
        temp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rectangles = self.mtcnn_detector.detectFace(temp_img, self.threshold)
        draw = img.copy()
        for rectangle in rectangles:
            # 在原图画出人脸框
            cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                          (0, 0, 255),
                          2)
        return draw

    def discern(self, img):
        """
        识别人脸信息

        :param img:
        :return:
        """
        height, width, _ = np.shape(img)
        # 灰度处理图片
        tempimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 定位人脸框
        rectangles = self.mtcnn_detector.detectFace(tempimg, self.threshold)

        if len(rectangles) == 0:
            return img

        # 转化成正方形
        rectangles = utils.rect2square(np.array(rectangles, dtype=np.int32))
        rectangles[:, [0, 2]] = np.clip(rectangles[:, [0, 2]], 0, width)
        rectangles[:, [1, 3]] = np.clip(rectangles[:, [1, 3]], 0, height)

        face_encodings = []
        for rectangle in rectangles:
            # 截取画出的人脸框图像
            landmark = np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])
            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]

            # 通过眼睛水平人脸对齐
            crop_img, _ = utils.Alignment_1(crop_img, landmark)
            crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)

            # 提取FaceNet的128个特征值
            face_encoding = utils.calc_128_vec(self.facenet_model, crop_img)
            face_encodings.append(face_encoding)

        face_names = []
        face_data = {}
        for face_encoding in face_encodings:

            # 与数据库中所有的人脸进行比对
            matches = utils.compare_faces(self.known_face["known_face_encodings"], face_encoding, tolerance=0.9)
            name = "未知人脸"

            # 找出距离最近的人脸
            face_distances = utils.face_distance(self.known_face["known_face_encodings"], face_encoding)

            # 取出距离最近人脸数据
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face["known_face_names"][best_match_index]
            face_names.append(name)
        face_data["face_names"] = face_names
        rectangles = rectangles[:, 0:4]

        for (left, top, right, bottom), name in zip(rectangles, face_names):
            # 画出人脸框
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            # 在框内写上名字信息

            img = utils.cv2ImgAddText(img, name, left + 5, bottom - 25)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img, name, (left, bottom - 15), font, 0.75, (255, 255, 255), 2)

        return img, face_data

    def get_calc_128_vec(self, img):
        # 转换图片通道
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 高斯滤波处理
        img = cv2.GaussianBlur(img, (5, 5), 0)  # 可以更改核大小

        # 检测人脸
        rectangles = self.mtcnn_detector.detectFace(img, self.threshold)
        # 转化成正方形
        # FaceNet模型的输入是160x160x3的数据
        rectangles = utils.rect2square(np.array(rectangles))
        for rectangle in rectangles:
            # 截取图像返回关键点坐标信息
            landmark = np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])
            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]

            # cv2.imshow("对齐之前人脸截图", cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

            # 根据双眼特征点进行水平对齐
            crop_img, _ = utils.Alignment_1(crop_img, landmark)
            # cv2.imshow("对齐后效果", cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

            crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)

            # 提取FaceNet的128个特征值
            pre = utils.calc_128_vec(self.facenet_model, crop_img)
            return pre
        pass


if __name__ == "__main__":
    model = Model()
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        _img, face_data = model.discern(img)
        cv2.imshow('Camera:0', _img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
