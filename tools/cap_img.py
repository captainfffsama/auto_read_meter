# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 12月 27日 星期一 11:06:06 CST
@Description: 用于采集数据
'''

import os
import argparse
import cv2
from core.meter_det.darknet.d_yolo import DarknetDet
from core.file_reader import DataSequence
from tqdm import tqdm

import debug_tools as D


def parse_args():
    parser = argparse.ArgumentParser(description="infer args set")
    parser.add_argument(
        "--i",
        type=str,
        default="cam:0",
        help="",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=
        "/home/chiebotgpuhq/MyCode/python/meter_auto_read/model_weight/meter_det/indoor.cfg",
        help="",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=
        "/home/chiebotgpuhq/MyCode/python/meter_auto_read/model_weight/meter_det/indoor.names",
        help="",
    )
    parser.add_argument(
        "--weight",
        type=str,
        default=
        "/home/chiebotgpuhq/MyCode/python/meter_auto_read/model_weight/meter_det/indoor.weights",
        help="",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/data/own_dataset/indoor_meter/VA_indoor_meter/ex4",
        help="",
    )
    args = parser.parse_args()
    return args


def main(img_path, det_model, save_dir=""):
    data_loader = DataSequence(img_path)
    count = 0
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    for data in data_loader():
        ori_img, img_path = data
        r, image = det_model.debug(ori_img)
        cv2.imshow("frame", image)
        inputkey = cv2.waitKey(20)
        if ord("q") == inputkey:
            data_loader.stop()
        if ord("w") == inputkey:
            file_name = str(count)
            for idx, r_i in enumerate(r):
                pos = [i for i in r_i[1:]]
                img = ori_img[pos[1]:pos[3], pos[0]:pos[2], :]
                cv2.imwrite(
                    os.path.join(save_dir,
                                 file_name + "_" + str(idx) + ".jpg"), img)
            count += 1


if __name__ == "__main__":
    args = parse_args()
    det_model = DarknetDet(args.cfg,
                           args.name,
                           args.weight,
                           thr=0.5,
                           class_filter={
                               "meter_square",
                           })
    main(args.i, det_model, args.save_dir)
