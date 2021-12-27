# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 12月 23日 星期四 15:55:34 CST
@Description: 用来处理图片的读取,使之对于不同的读取有统一的接口
'''
from logging import exception
import os
from copy import deepcopy
from typing import Union

import cv2
import torch
from torchvision import transforms


def get_all_file_path(file_dir: str, filter_: tuple = ('.jpg', )) -> list:
    #遍历文件夹下所有的file
    return [os.path.join(maindir,filename) for maindir,_,file_name_list in os.walk(file_dir) \
            for filename in file_name_list \
            if os.path.splitext(filename)[1] in filter_ ]


class DataSequence(object):
    def __init__(self,
                 file_path: str,
                 filter_: tuple = (".jpg", ),
                 video_size: tuple = (1280, 720)):
        self.stop_flag = False
        self._count = 0
        self._type = "None"
        if os.path.isdir(file_path):
            self._data_list = get_all_file_path(file_path,
                                                filter_)  # type: ignore
            self._type = "image"
        else:
            if file_path.startswith("cam:"):
                self._data_list = cv2.VideoCapture(int(file_path.split(":")[-1]))
                self._data_list.set(cv2.CAP_PROP_FRAME_WIDTH, video_size[0])
                self._data_list.set(cv2.CAP_PROP_FRAME_HEIGHT, video_size[1])
                self._type = "video"

                for i in range(10):
                    self._data_list.read()

            else:
                raise ValueError("file_path must be dir or video num,video num should like \"cam:0\"")

    @property
    def type(self):
        return self._type

    def __call__(self):
        try:
            while True:
                if self.stop_flag:
                    break
                if isinstance(self._data_list, list):
                    if self._count>=len(self._data_list):
                        raise StopIteration
                    current_img_path = self._data_list[
                        self._count]  # type: ignore
                    img = cv2.imread(current_img_path)
                    yield img, current_img_path
                else:
                    ret, frame = self._data_list.read()
                    if ret:
                        yield frame, self._count
                    else:
                        print("video over,can not read frame")
                        break
                self._count += 1
        except StopIteration:
            print("seqs over")
            pass

    def stop(self):
        self.stop_flag = True


_BASIC_FRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def img_deal(img, img_resize: tuple, basic_transform=_BASIC_FRANSFORM):
    if isinstance(img, str):
        img = cv2.imread(img)
    ori_img = deepcopy(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_hw = img.shape[:2]

    h_resize_rate = img_resize[0] / img.shape[0]
    w_resize_rate = img_resize[1] / img.shape[1]

    resize_rate = min(h_resize_rate, w_resize_rate)
    img = cv2.resize(img, dsize=None, fx=resize_rate, fy=resize_rate)

    w_reminder = int(img_resize[1] - img.shape[1])
    h_reminder = int(img_resize[0] - img.shape[0])

    if w_reminder > 0:
        img = cv2.copyMakeBorder(img,
                                 0,
                                 0,
                                 0,
                                 w_reminder,
                                 cv2.BORDER_CONSTANT,
                                 value=[255, 255, 255])
    if h_reminder > 0:
        img = cv2.copyMakeBorder(img,
                                 0,
                                 h_reminder,
                                 0,
                                 0,
                                 cv2.BORDER_CONSTANT,
                                 value=[255, 255, 255])

    img = basic_transform(img)
    return img.unsqueeze(0), resize_rate, ori_hw


if __name__ == "__main__":
    a = DataSequence(
        "/data/own_dataset/indoor_meter/VA_indoor_meter/20211207/")
    for i in a():
        frame, idx = i
        cv2.imshow("frame", frame)
        print(idx)
        key = cv2.waitKey(30)
        if ord("q") == key:
            break
