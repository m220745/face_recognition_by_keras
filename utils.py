# -*- coding: utf-8 -*-
# @Time : 2021/3/2 15:21
# @Author : xiaojie
# @File : utils.py
# @Software: PyCharm

import math
import sys
from operator import itemgetter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def calculateScales(img):
    """
    计算原始输入图像每一次缩放的比例

    :param img:
    :return:
    """
    pr_scale = 1.0
    h, w, _ = img.shape

    # 将最大的图像大小进行一个固定
    # 如果图像的短边大于500，则将短边固定为500
    # 如果图像的长边小于500，则将长边固定为500
    if min(w, h) > 500:
        pr_scale = 500.0 / min(h, w)
        w = int(w * pr_scale)
        h = int(h * pr_scale)
    elif max(w, h) < 500:
        pr_scale = 500.0 / max(h, w)
        w = int(w * pr_scale)
        h = int(h * pr_scale)

    # 建立图像金字塔的scales，防止图像的宽高小于12
    scales = []
    # 缩放比例
    factor = 0.709
    factor_count = 0
    minl = min(h, w)
    while minl >= 12:
        scales.append(pr_scale * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales


def rect2square(rectangles):
    """
    将长方形调整为正方形

    :param rectangles: 矩阵
    :return:
    """
    w = rectangles[:, 2] - rectangles[:, 0]
    h = rectangles[:, 3] - rectangles[:, 1]
    l = np.maximum(w, h).T
    rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
    rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T
    return rectangles


def NMS(rectangles, threshold):
    """
    非极大值抑制（Non-Maximum Suppression，NMS），即抑制不是极大值的元素，可以理解为局部最大搜索。

    :param rectangles: 矩阵
    :param threshold: 阈值
    :return:
    """
    if len(rectangles) == 0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    I = np.array(s.argsort())
    pick = []
    while len(I) > 0:
        # 计算两个框的IoU（交并比）值
        # 重叠部分左上右下坐标
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])  # I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        # 计算边界框的宽度和高度
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        # 重叠部分面积
        inter = w * h
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o <= threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle


def detect_face_Pnet(cls_prob, roi, out_side, scale, width, height, threshold):
    """
    对Pnet处理后的结果进行处理

    :param cls_prob: 用于面部分类的softmax特征图
    :param roi: 回归特征偏移量
    :param out_side: 功能图的最大尺寸
    :param scale: 当前输入图像比例尺，多比例尺
    :param width: 图像的原点宽度
    :param height: 图像的原点高度
    :param threshold: 0.6具有99％的召回率
    :return:
    """

    # 计算特征点之间的步长
    stride = 0
    if out_side != 1:
        stride = float(2 * out_side - 1) / (out_side - 1)

    # 获得满足得分阈值的特征点的坐标
    (y, x) = np.where(cls_prob >= threshold)

    # 获得满足得分阈值的特征点得分
    # 最终获得的score的shape为：[num_box, 1]
    score = np.expand_dims(cls_prob[y, x], -1)

    # 将对应的特征点的坐标转换成位于原图上的先验框的坐标
    # 利用回归网络的预测结果对先验框的左上角与右下角进行调整
    # 获得对应的粗略预测框
    # 最终获得的boundingbox的shape为：[num_box, 4]
    boundingbox = np.concatenate([np.expand_dims(x, -1), np.expand_dims(y, -1)], axis=-1)

    # 找到对应原图的位置
    top_left = np.fix(stride * boundingbox + 0)
    bottom_right = np.fix(stride * boundingbox + 11)
    boundingbox = np.concatenate((top_left, bottom_right), axis=-1)
    boundingbox = (boundingbox + roi[y, x] * 12.0) * scale

    # 将预测框和得分进行堆叠，并转换成正方形
    # 最终获得的rectangles的shape为：[num_box, 5]
    rectangles = np.concatenate((boundingbox, score), axis=-1)
    rectangles = rect2square(rectangles)

    rectangles[:, [1, 3]] = np.clip(rectangles[:, [1, 3]], 0, height)
    rectangles[:, [0, 2]] = np.clip(rectangles[:, [0, 2]], 0, width)
    return rectangles


def detect_face_Rnet(cls_prob, roi, rectangles, width, height, threshold):
    """
    对Rnet处理后的结果进行处理

    :param cls_prob: 用于面部分类的softmax特征图
    :param roi: 回归特征偏移量
    :param rectangles: Pnet的预测矩阵
    :param width: 图像的原点宽度
    :param height: 图像的原点高度
    :param threshold: 0.6召回率达可到97％
    :return:
    """

    # 利用得分进行筛选
    pick = cls_prob[:, 1] >= threshold

    score = cls_prob[pick, 1:2]
    rectangles = rectangles[pick, :4]
    roi = roi[pick, :]

    # 利用Rnet网络的预测结果对粗略预测框进行调整
    # 最终获得的rectangles的shape为：[num_box, 4]
    w = np.expand_dims(rectangles[:, 2] - rectangles[:, 0], -1)
    h = np.expand_dims(rectangles[:, 3] - rectangles[:, 1], -1)
    rectangles[:, [0, 2]] = rectangles[:, [0, 2]] + roi[:, [0, 2]] * w
    rectangles[:, [1, 3]] = rectangles[:, [1, 3]] + roi[:, [1, 3]] * w

    # 将预测框和得分进行堆叠，并转换成正方形
    # 最终获得的rectangles的shape为：[num_box, 5]
    rectangles = np.concatenate((rectangles, score), axis=-1)
    rectangles = rect2square(rectangles)

    rectangles[:, [1, 3]] = np.clip(rectangles[:, [1, 3]], 0, height)
    rectangles[:, [0, 2]] = np.clip(rectangles[:, [0, 2]], 0, width)
    return np.array(NMS(rectangles, 0.7))


def detect_face_Onet(cls_prob, roi, pts, rectangles, width, height, threshold):
    """
    对Onet处理后的结果进行处理

    :param cls_prob: 用于面部分类的softmax特征图
    :param roi: 回归特征偏移量
    :param pts: landmark(特征点)偏移量
    :param rectangles: Rnet的预测矩阵
    :param width: 图像的原点宽度
    :param height: 图像的原点高度
    :param threshold: 0.7可以产生94％的召回率
    :return:
    """

    # 利用得分进行筛选
    pick = cls_prob[:, 1] >= threshold

    score = cls_prob[pick, 1:2]
    rectangles = rectangles[pick, :4]
    pts = pts[pick, :]
    roi = roi[pick, :]

    w = np.expand_dims(rectangles[:, 2] - rectangles[:, 0], -1)
    h = np.expand_dims(rectangles[:, 3] - rectangles[:, 1], -1)

    # 利用Onet网络的预测结果对预测框进行调整
    # 通过解码获得人脸关键点与预测框的坐标
    # 最终获得的face_marks的shape为：[num_box, 10]
    # 最终获得的rectangles的shape为：[num_box, 4]
    face_marks = np.zeros_like(pts)
    face_marks[:, [0, 2, 4, 6, 8]] = w * pts[:, [0, 1, 2, 3, 4]] + rectangles[:, 0:1]
    face_marks[:, [1, 3, 5, 7, 9]] = h * pts[:, [5, 6, 7, 8, 9]] + rectangles[:, 1:2]
    rectangles[:, [0, 2]] = rectangles[:, [0, 2]] + roi[:, [0, 2]] * w
    rectangles[:, [1, 3]] = rectangles[:, [1, 3]] + roi[:, [1, 3]] * w

    # 将预测框和得分进行堆叠
    # 最终获得的rectangles的shape为：[num_box, 15]
    rectangles = np.concatenate((rectangles, score, face_marks), axis=-1)

    rectangles[:, [1, 3]] = np.clip(rectangles[:, [1, 3]], 0, height)
    rectangles[:, [0, 2]] = np.clip(rectangles[:, [0, 2]], 0, width)
    return np.array(NMS(rectangles, 0.3))


# 人脸对齐
# 参考 https://blog.csdn.net/Code_Mart/article/details/100044071
def Alignment_1(img, landmark):
    """
    通过双眼坐标进行旋正对齐

    :param img:
    :param landmark:
    :return:
    """
    x = None
    y = None
    if landmark.shape[0] == 68:  # use left_eye and right_eye location
        x = landmark[36, 0] - landmark[45, 0]
        y = landmark[36, 1] - landmark[45, 1]
    elif landmark.shape[0] == 5:
        x = landmark[0, 0] - landmark[1, 0]
        y = landmark[0, 1] - landmark[1, 1]

    if x == 0:
        angle = 0
    else:
        angle = math.atan(y / x) * 180 / math.pi

    center = (img.shape[1] // 2, img.shape[0] // 2)

    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    # 仿射函数
    new_img = cv2.warpAffine(img, RotationMatrix, (img.shape[1], img.shape[0]))

    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(RotationMatrix[0, 0] * landmark[i, 0] + RotationMatrix[0, 1] * landmark[i, 1] + RotationMatrix[0, 2])
        pts.append(RotationMatrix[1, 0] * landmark[i, 0] + RotationMatrix[1, 1] * landmark[i, 1] + RotationMatrix[1, 2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark


def Alignment_2(img, std_landmark, landmark):
    """
    通过矩阵运算求解仿射矩阵进行旋正

    :param img:
    :param std_landmark:
    :param landmark:
    :return:
    """

    # https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
    def Transformation(std_landmark, landmark):
        std_landmark = np.matrix(std_landmark).astype(np.float64)
        landmark = np.matrix(landmark).astype(np.float64)

        c1 = np.mean(std_landmark, axis=0)
        c2 = np.mean(landmark, axis=0)
        std_landmark -= c1
        landmark -= c2

        s1 = np.std(std_landmark)
        s2 = np.std(landmark)
        std_landmark /= s1
        landmark /= s2

        U, S, Vt = np.linalg.svd(std_landmark.T * landmark)
        R = (U * Vt).T

        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

    Trans_Matrix = Transformation(std_landmark, landmark)  # Shape: 3 * 3
    Trans_Matrix = Trans_Matrix[:2]
    Trans_Matrix = cv2.invertAffineTransform(Trans_Matrix)
    new_img = cv2.warpAffine(img, Trans_Matrix, (img.shape[1], img.shape[0]))

    Trans_Matrix = np.array(Trans_Matrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(Trans_Matrix[0, 0] * landmark[i, 0] + Trans_Matrix[0, 1] * landmark[i, 1] + Trans_Matrix[0, 2])
        pts.append(Trans_Matrix[1, 0] * landmark[i, 0] + Trans_Matrix[1, 1] * landmark[i, 1] + Trans_Matrix[1, 2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark


def Alignment_3(std_img, std_landmark, img, landmark):
    """
    通过最小二乘法求解仿射矩阵进行旋正

    :param std_img:
    :param std_landmark:
    :param img:
    :param landmark:
    :return:
    """

    def Transformation(std_landmark, landmark):
        Trans_Matrix = np.float32([[1, 0, 0], [0, 1, 0]])
        n_pts = landmark.shape[0]
        ones = np.ones((n_pts, 1), landmark.dtype)
        landmark_ = np.hstack([landmark, ones])
        std_landmark_ = np.hstack([std_landmark, ones])

        A, res, rank, s = np.linalg.lstsq(landmark_, std_landmark_, rcond=-1)

        if rank == 3:
            Trans_Matrix = np.float32([
                [A[0, 0], A[1, 0], A[2, 0]],
                [A[0, 1], A[1, 1], A[2, 1]]
            ])
        elif rank == 2:
            Trans_Matrix = np.float32([
                [A[0, 0], A[1, 0], 0],
                [A[0, 1], A[1, 1], 0]
            ])

        return Trans_Matrix

    # def Transformation(std_landmark,landmark):
    #     from matlab_cp2tform import get_similarity_transform_for_cv2
    #     return get_similarity_transform_for_cv2(landmark, std_landmark, False)

    Trans_Matrix = Transformation(std_landmark, landmark)  # Shape: 2 * 3
    new_img = cv2.warpAffine(img, Trans_Matrix, (img.shape[1], img.shape[0]))

    Trans_Matrix = np.array(Trans_Matrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(Trans_Matrix[0, 0] * landmark[i, 0] + Trans_Matrix[0, 1] * landmark[i, 1] + Trans_Matrix[0, 2])
        pts.append(Trans_Matrix[1, 0] * landmark[i, 0] + Trans_Matrix[1, 1] * landmark[i, 1] + Trans_Matrix[1, 2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark


# 图片预处理
def pre_process(x):
    """
    高斯归一化分布处理

    :param x:
    :return:
    """
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y


def l2_normalize(x, axis=-1, epsilon=1e-10):
    """
    l2标准化

    :param x:
    :param axis:
    :param epsilon:
    :return:
    """
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def calc_128_vec(model, img):
    """
    计算128特征值

    :param model:
    :param img:
    :return:
    """

    face_img = pre_process(img)
    pre = model.predict(face_img)
    pre = l2_normalize(np.concatenate(pre))
    pre = np.reshape(pre, [128])
    return pre


def face_distance(face_encodings, face_to_compare):
    """
    计算人脸距离(欧氏距离)\n
    给定一组面部编码，将它们与已知的面部编码进行比较，得到欧氏距离。对于每一个比较的脸，欧氏距离代表了这些脸有多相似。

    :param face_encodings:
    :param face_to_compare:
    :return:
    """
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


# 比较人脸
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    比较人脸\n
    比较脸部编码列表和候选编码，看看它们是否匹配。

    :param known_face_encodings: 已知的人脸编码列表
    :param face_encoding_to_check: 待进行对比的单张人脸编码数据
    :param tolerance: 两张脸之间有多少距离才算匹配。该值越小对比越严格，0.6是典型的最佳值
    :return:
    """
    dis = face_distance(known_face_encodings, face_encoding_to_check)
    return list(dis <= tolerance)


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
