# -*- coding: utf-8 -*-
# @Time : 2021/3/5 11:52
# @Author : xiaojie
# @File : test_rtsm.py
# @Software: PyCharm

import time
import multiprocessing as mp
import threading
from queue import Queue
import cv2
import numpy as np
import asyncio
import websockets
from websockets import ConnectionClosed

frame = None

"""
推送摄像头的RTSM流媒体DEMO
"""

def websocket_process(img_dict):
    # 服务器端主逻辑
    async def main_logic(websocket, path):
        await recv_msg(websocket, img_dict)

    # new_loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(new_loop)
    start_server = websockets.serve(main_logic, '0.0.0.0', 5678)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


async def recv_msg(websocket, img_dict):
    recv_text = await websocket.recv()
    if recv_text == 'begin':
        while True:
            frame = img_dict['img']
            if isinstance(frame, np.ndarray):
                enconde_data = cv2.imencode('.png', frame)[1]
                enconde_str = enconde_data.tostring()
                try:
                    await websocket.send(enconde_str)
                except Exception as e:
                    print(e)
                    return True


def image_put(q, user, pwd, ip):
    rtsp_url = 'rtsp://{0}:{1}@{2}:554/h265/ch1/main/av_stream'.format(user, pwd, ip)
    cap = cv2.VideoCapture(0)
    i = 0
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (500, 500))
            q.put(frame)
            q.get() if q.qsize() > 1 else time.sleep(0.01)


def image_get(q, img_dict):
    while True:
        frame = q.get()
        if isinstance(frame, np.ndarray):
            img_dict['img'] = frame


def run_single_camera(user_name, user_pwd, camera_ip):
    mp.set_start_method(method='spawn')  # init
    queue = mp.Queue(maxsize=3)
    m = mp.Manager()
    img_dict = m.dict()
    Processes = [mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)),
                 mp.Process(target=image_get, args=(queue, img_dict)),
                 mp.Process(target=websocket_process, args=(img_dict,))]

    [process.start() for process in Processes]
    [process.join() for process in Processes]


def run():
    run_single_camera('admin', 'admin', '0.0.0.0')


if __name__ == '__main__':
    run()
    print("aaaa")