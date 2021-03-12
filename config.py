# -*- coding: utf-8 -*-
# @Time : 2021/1/21 11:06
# @Author : xiaojie
# @File : config.py
# @Software: PyCharm


import os
# import tensorflow as tf
# 以下两行代码以使用tf1.x版本
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf = tf

# mtcnn 相关
abspath = os.path.dirname(os.path.abspath(__file__))
p_net_md_path = abspath + "/mtcnn_master/test_model/Pnet.h5"
r_net_md_path = abspath + "/mtcnn_master/test_model/Rnet.h5"
o_net_md_path = abspath + "/mtcnn_master/test_model/Onet.h5"

# face_net 相关
checkpoint_path = abspath + "/facenet_master/test_model/model-20180408-102900.ctpk"
checkpoint_meta_path = abspath + "/facenet_master/test_model/model-20180408-102900.meta"
checkpoint_dir = os.path.dirname(checkpoint_path)
model_pb_path = abspath + "/facenet_master/test_model/20180408-102900.pb"
model_pb_dir = os.path.dirname(model_pb_path)
model_h5_path = abspath + "/facenet_master/test_model/facenet_keras.h5"

# 数据库中已经保存的人脸图片的128个特征数据
known_face = {}

