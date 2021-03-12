# -*- coding: utf-8 -*-
# @Time : 2021/3/2 13:37
# @Author : xiaojie
# @File : mtcnn_master.py
# @Software: PyCharm

# 参考： https://github.com/davidsandberg/facenet
# 参考： https://github.com/zipengbo/keras-mtcnn

import cv2
import numpy as np
# from config import *

from keras.layers import (Conv2D, Dense, Flatten, Input, MaxPool2D, Permute)
from keras.layers.advanced_activations import PReLU
from keras.models import Model

import utils


# Pnet的全称为Proposal Network，其基本的构造是一个全卷积网络
def create_Pnet(weight_path):
    """
    粗提取(初步特征提取与标定人脸框)\n
    输入是12×12×3，经过卷积和最大池化，最后的feature map是1×1×32，然后接上一个全卷积得到三个输出，分别是1×1×2,1×1×4和1×1*10。\n
    这里PNet的输出还是一个feature map，维度是（N，W，H，C）。

    :param weight_path: 权重信息模型路径
    :return:
    """
    inputs = Input(shape=[None, None, 3])

    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    model = Model([inputs], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model


# Rnet全称为Refine Network，其基本的构造是一个卷积神经网络，相对于Pnet来说，增加了一个全连接层，因此对于输入数据的筛选会更加严格。
def create_Rnet(weight_path):
    """
    细提取(进一步提取人脸框特征信息，过滤大量效果比较差的候选框，修正Pnet得到的粗略框)\n
    输入是24×24×3，经过卷积和最大池化，最后接上全连接层得到128维的向量，\n
    然后得到三个输出，维度是（N，C），C是16通道，包括2通道的人脸分类，4通道的bounding box偏移量以及10通道的landmark(特征点)偏移量。

    :param weight_path:
    :return:
    """
    inputs = Input(shape=[24, 24, 3])
    # x 维度变化(套用维度计算公式) [24, 24, 3]
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    # x -> [22,22,28]
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # x -> [11,11,28]

    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    # x -> [9,9,48]
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)
    # x -> [4, 4, 48]

    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    # x -> [3,3,64]
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)

    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)
    # x-> [64,3,3]

    x = Dense(128, name='conv4')(x)
    x = PReLU(name='prelu4')(x)

    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = Dense(4, name='conv5-2')(x)

    model = Model([inputs], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model


# Onet全称为Output Network，基本结构是一个较为复杂的卷积神经网络，相对于Rnet来说多了一个卷积层。
def create_Onet(weight_path):
    """
    精提取(通过更多的监督来识别面部的区域，而且会对人的面部特征点进行回归，最终输出五个人脸面部特征点，画出人脸框)
    输入是48×48×3，经过卷积和最大池化，最后接上全连接层得到256维的向量，然后得到三个输出，维度是（N，C）
    :param weight_path:
    :return:
    """
    inputs = Input(shape=[48, 48, 3])
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)

    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)

    x = Dense(256, name='conv5')(x)
    x = PReLU(name='prelu5')(x)

    classifier = Dense(2, activation='softmax', name='conv6-1')(x)
    bbox_regress = Dense(4, name='conv6-2')(x)
    landmark_regress = Dense(10, name='conv6-3')(x)

    model = Model([inputs], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)
    return model


class MTCnnDetector():
    def __init__(self, p_net_md_path="/test_model/Pnet.h5", r_net_md_path="/test_model/Rnet.h5",
                 o_net_md_path="/test_model/Onet.h5"):
        self.Pnet = create_Pnet(p_net_md_path)
        self.Rnet = create_Rnet(r_net_md_path)
        self.Onet = create_Onet(o_net_md_path)

    def detectFace(self, img, threshold):
        """
        检测人脸

        :param img: 图像
        :param threshold: 3个阈值列表
        :return:
        """
        # 简单归一化处理
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape

        # 计算原始输入图像每一次缩放的比例
        scales = utils.calculateScales(img)

        out = []

        # 第一步 Pnet 粗选
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = np.expand_dims(scale_img, 0)
            ouput = self.Pnet.predict(inputs)
            ouput = [ouput[0][0], ouput[1][0]]
            out.append(ouput)

        rectangles = []

        # 对每个金字塔的图像预测的输出结果out进行循环
        # 取出每张图片的种类预测和回归预测结果
        for i in range(len(scales)):
            cls_prob = out[i][0][:, :, 1]
            roi = out[i][1]
            # 取出每个缩放后图片的高宽
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            # 解码的过程
            rectangle = utils.detect_face_Pnet(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h,
                                               threshold[0])
            rectangles.extend(rectangle)

        # 计算IoU值并进行非极大值抑制
        rectangles = np.array(utils.NMS(rectangles, 0.7))

        if len(rectangles) == 0:
            return rectangles

        # 第二步 Rnet 细选人脸框
        predict_R_batch = []
        # 处理P-Net的预测结果
        for rectangle in rectangles:
            # 利用获取到的粗略坐标，在原图上进行截取
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            # 将截取到的图片进行resize，调整成24x24的大小
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_R_batch.append(scale_img)

        cls_prob, roi_prob = self.Rnet.predict(np.array(predict_R_batch))
        # 解码的过程，需传入第一步P-Net的预测矩阵
        rectangles = utils.detect_face_Rnet(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])

        if len(rectangles) == 0:
            return rectangles

        # 第三步 Onet部分 精选人脸框
        predict_batch = []
        for rectangle in rectangles:
            # 利用获取到的粗略坐标，在原图上进行截取
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            # 将截取到的图片进行resize，调整成48x48的大小
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)

        cls_prob, roi_prob, pts_prob = self.Onet.predict(np.array(predict_batch))
        # 解码的过程
        rectangles = utils.detect_face_Onet(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])

        return rectangles
