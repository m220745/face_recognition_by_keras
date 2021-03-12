# -*- coding: utf-8 -*-
# @Time : 2021/3/5 11:10
# @Author : xiaojie
# @File : service.py
# @Software: PyCharm

import re
import base64
import json
import asyncio
import websockets
import numpy as np
import cv2

from facenet_master.face_model import Model

model_api = Model()


def getMatch(match, str, returnType="bool"):
    pattern = re.search(r'{}'.format(match), str)
    if pattern:
        returnS = pattern.group()
        returnB = True
        # print("search --> pattern.group() : {}".format(returnS))
    else:
        # print("No match!!")
        returnS = ""
        returnB = False
    if returnType == "bool":
        return returnB
    elif returnType == "str":
        return returnS


def base64_to_image(base64_code):
    # base64解码
    img_data = base64.b64decode(base64_code)
    # 转换为np数组
    img_array = np.fromstring(img_data, np.uint8)
    # 转换成opencv可用格式
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    return img


def image_to_base64(image_np):
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code


# 检测客户端权限，用户名密码通过才能退出循环
async def check_permit(websocket):
    while True:
        recv_str = await websocket.recv()
        cred_dict = recv_str.split(":")
        if cred_dict[0] == "admin" and cred_dict[1] == "123456":
            response_str = "检验通过..."
            await websocket.send(response_str)
            return True
        else:
            response_str = "账号或密码有误，请重新校验..."
            await websocket.send(response_str)


# 接收客户端消息并处理，这里把前端传来的base64图片进行转换并检测处理后返回
async def recv_msg(websocket):
    while True:
        recv_text = await websocket.recv()
        response_content = f"{recv_text}"
        d_map = {}
        r_map = {}

        try:
            d_map = json.loads(response_content)
        except Exception as e:
            r_map["data"] = {}
            r_map["img"] = ""
            r_map["type"] = 404
            pass

        if "data" in d_map.keys() and getMatch(
                "^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{4}|[A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)$", d_map["data"]):
            r_map["type"] = d_map["type"]
            img = base64_to_image(d_map["data"])
            # 画出人脸框
            if d_map["type"] == 0:
                img = model_api.draw_face(img)
                r_map["data"] = {}
            elif d_map["type"] == 1:  # 识别人脸
                img, face_data = model_api.discern(img)
                r_map["data"] = face_data
            elif d_map["type"] == 2:  # 录入到数据库
                pre = model_api.get_calc_128_vec(img)
                res = model_api.db.saveFace128Vec(data={"name": d_map["name"], "face128vec": pre.tolist()})
                model_api.known_face = model_api.db.getKnownFaceToDict()
                print(res)
                r_map["data"] = "录入成功"
                pass

            r_map["img"] = "data:image/png;base64," + image_to_base64(img)

        response_content = json.dumps(r_map, ensure_ascii=False)
        # print(response_content)
        await websocket.send(response_content)


# 服务器端主逻辑
# websocket和path是该函数被回调时自动传过来的，不需要自己传
async def main_logic(websocket, path):
    # 检验账号密码
    # await check_permit(websocket)

    await recv_msg(websocket)


# 把ip换成自己本地的ip
start_server = websockets.serve(main_logic, '0.0.0.0', 5678)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
