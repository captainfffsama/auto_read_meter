
import os
import math
import argparse
from collections import defaultdict
from typing import List
from pprint import pprint
import random
from collections import namedtuple, defaultdict
from functools import partial
from copy import deepcopy

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

from core.pt_detection.pt_det_net import PointDetectNet
from core.utils import get_circle_center, get_angle, get_dist, find_normal_line, find_cross_pt
from core.config.base_config import get_cfg_defaults
from core.ocr.pp_ocr import OCRModel
from core.meter_det.darknet.d_yolo import DarknetDet

from main import  init_model,get_meter_img,filter_cm,d_map,fix_pt_detect_result,get_meter_center,fix_ocr_scale,get_num

def parse_args():
    parser = argparse.ArgumentParser(description="infer args set")
    parser.add_argument(
        "--cfg",
        type=str,
        default="./config/cfgv1.yaml",
        help="",
    )
    args = parser.parse_args()
    return args

def get_frame():
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    # warmup
    for i in range(10):
        cap.read()
    while True:
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            break

def draw_frame(frame,num,**kwargs):
    font = cv2.FONT_HERSHEY_PLAIN
    if isinstance(num,float):
        num="{}".format(num)
    if not isinstance(num,str):
        num=str(num)
    cv2.putText(frame,num , (100,100), font, 5,
                            (0,0,255), 3)
    if "meter_det" in kwargs.keys():
        cv2.rectangle(frame,tuple(kwargs["meter_det"][:2]),tuple(kwargs["meter_det"][2:]),(255,0,0),3)
    color_map = {
        "min_scale": (0, 0, 255),
        "max_scale": (0, 255, 0),
        "pointer": (255, 0, 0),
        "small_label": (255, 255, 0),
        "big_label": (0, 255, 255),
    }
    for k,v in kwargs.get("pt_result",{}).items():
        for v_i in v:
            pos = (int(v_i[0]), int(v_i[1]))
            cv2.circle(frame, pos, 3, color_map[k], -1)

            cv2.putText(frame, "{}".format(k), pos, font, 1,
                        color_map[k], 1)

    for v in kwargs.get("ocr_result",[]):
        cv2.circle(frame, v[0], 3, (10,150,10), -1)

        cv2.putText(frame, "{}".format(v[-1]), v[0], font, 2,
                    (10,150,10), 2)


    if "c_pt" in kwargs.keys():
        cv2.circle(frame, kwargs["c_pt"], 3, (0,100,0), -1)

    return frame

def img_deal(img, basic_transform, img_resize):
    if isinstance(img,str):
        img = cv2.imread(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

def fix_pt_pos(pt,fix_v):
    return int(pt[0]+fix_v[0]),int(pt[1]+fix_v[1])

def get_r(model,ocr,meter_det,ori_img,basic_transform):

    draw_info={}
    meter_det_result=meter_det(ori_img)
    if len(meter_det_result)<1:
        return "can no find meter",ori_img,draw_info
    if not all(meter_det_result[0]):
        return "det some thing wrong",ori_img,draw_info
    draw_info["meter_det"]=meter_det_result[0][1:]
    meter_img,fix_pos=get_meter_img(ori_img,meter_det_result)
    try:
        img, resize_rate, ori_hw = img_deal(meter_img, basic_transform,
                                        (416, 416))
    except:
        breakpoint()
    scm, srm, pmcm, pmrm, rcm, rrm = model(img.cuda())

    pmcm = filter_cm(pmcm)
    scm = filter_cm(scm)
    rcm = filter_cm(rcm)
    result = d_map(scm, srm, pmcm, pmrm, rcm, rrm,
                model.downsample_rate)

    # TODO:因为这里只有一张图片,简单处理
    pt_result = fix_pt_detect_result(result[0], resize_rate)
    draw_info["pt_result"]={k:[fix_pt_pos(x,fix_pos) for x in v] for k,v in pt_result.items()}
    ocr_result = ocr(meter_img)
    draw_info["ocr_result"]=[(fix_pt_pos(x[0],fix_pos),x[1]) for x in ocr_result]
    if len(ocr_result) < 2:
        return "ocr no detect",ori_img,draw_info
    circle_pt, radius = get_meter_center(pt_result, ori_hw)

    if radius<0:
        return "ocr no detect",ori_img,draw_info
    draw_info["c_pt"]=fix_pt_pos(circle_pt,fix_pos)
    try:
        ocr_angle, base_angle = fix_ocr_scale(
            ocr_result, circle_pt, pt_result["min_scale"][0][:2],
            pt_result["max_scale"][0][:2])
    except:
        return "scale generate failed",ori_img,draw_info

    num = get_num(ocr_angle, base_angle, circle_pt, pt_result)
    print(num)
    return num,ori_img,draw_info

def main(args):
    model, ocr,meter_det = init_model(args)
    model.eval()

    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        for ori_img in get_frame():

            num,ori_img,draw_info=get_r(model,ocr,meter_det,ori_img,basic_transform)
            ori_img=draw_frame(ori_img,num,**draw_info)
            cv2.imshow("frame",ori_img)
            inputkey=cv2.waitKey(25)
            # if ord("w")==inputkey:
            #     cv2.imwrite(os.path.join(save_path,str(count)+'b.jpg'),roi)
            #     count+=1
            if ord("q")==inputkey:
                break





if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    main(cfg)