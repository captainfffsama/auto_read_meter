"""
@Author: CaptainHu
@Date: 2021年 09月 23日 星期四 16:58:29 CST
@Description: 用来做前项推理
"""
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
from core.ocr.pp_ocr import PPOCR
from core.utils import get_circle_center, get_angle,get_dist,find_normal_line,find_cross_pt

import debug_tools as D


def parse_args():
    parser = argparse.ArgumentParser(description="infer args set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/experiments_data/point_weight/weight/49.ckpt",
        help="",
    )
    parser.add_argument("--data_dir",
                        type=str,
                        default="/data/own_dataset/indoor_meter/test3")
    args = parser.parse_args()
    return args


def init_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PointDetectNet(3).to(device)
    if os.path.exists(args.checkpoint):
        params = torch.load(args.checkpoint)
        model.load_state_dict(params, strict=False)

    ocr = PPOCR()

    return model, ocr


def img_deal(img_path, basic_transform, img_resize):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
    return img.unsqueeze(0), resize_rate


def get_all_file_path(file_dir: str, filter_: tuple = (".jpg", )) -> list:
    # 遍历文件夹下所有的file
    return [
        os.path.join(maindir, filename)
        for maindir, _, file_name_list in os.walk(file_dir)
        for filename in file_name_list
        if os.path.splitext(filename)[1] in filter_
    ]


def filter_cm(cls_map):
    cls_map_p = F.max_pool2d(cls_map, kernel_size=11, stride=1, padding=5)
    cls_map[cls_map != cls_map_p] = 0
    # cls_map[cls_map<=0.3]=0
    return cls_map


def d_map(scm, srm, pmcm, pmrm, d_rate):
    result: List[dict] = []

    for one_img_scm, one_img_srm, one_img_pmcm, one_img_pmrm in zip(
            scm, srm, pmcm, pmrm):
        ori_cm_shape = one_img_scm.shape[1:]
        one_img_result = defaultdict(list)

        for i in range(2):
            pt_idx = one_img_scm[i].argmax()
            pt_yi = pt_idx // ori_cm_shape[-1]
            pt_xi = pt_idx % ori_cm_shape[0]
            pt_score = one_img_scm[i, pt_yi, pt_xi]

            pt_x = one_img_srm[2 * i, pt_yi, pt_xi]
            pt_y = one_img_srm[2 * i + 1, pt_yi, pt_xi]
            one_img_result[str(i + 1)].append(
                (pt_x.item(), pt_y.item(), pt_score.item()))

        pt5_idx = one_img_pmcm[0].argmax()
        pt5_yi = pt5_idx // ori_cm_shape[-1]
        pt5_xi = pt5_idx % ori_cm_shape[0]
        pt5_score = one_img_pmcm[0, pt5_yi, pt5_xi]

        pt5_x = one_img_pmrm[0, pt5_yi, pt5_xi]
        pt5_y = one_img_pmrm[1, pt5_yi, pt5_xi]
        one_img_result["3"].append(
            (pt5_x.item(), pt5_y.item(), pt5_score.item()))
        result.append(one_img_result)

    return result


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range * 255


def vis_img(img, result,num):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy().squeeze()
        img = np.transpose(img, (1, 2, 0))
        img = normalization(img)
        img = img.astype(np.uint8).copy()
    font = cv2.FONT_HERSHEY_PLAIN

    color_map = {
        "1": (0, 255, 0),
        "2": (0, 0, 255),
        "3": (255, 0, 0),
        # "5":(255,255,0)
    }

    for k, v in result.items():
        for v_i in v:
            pos = (int(v_i[0]), int(v_i[1]))
            cv2.circle(img, pos, 3, color_map[k], -1)

            cv2.putText(img, "{}/{:.2f}".format(k, v_i[-1]), pos, font, 1,
                        color_map[k], 1)

    cv2.putText(img, str(num),(40,40),font,2,(255,255,255),2)

    return img[:, :-1]


def fix_pt_detect_result(result_ori: dict, resize_rate: float):
    result=deepcopy(result_ori)
    for k in result.keys():
        result[k] = [(x[0] / resize_rate, x[1] / resize_rate, x[2])
                     for x in result[k]]
    return result


def find_center(ocr_result):
    assert len(ocr_result) > 2, "orc pt is too less"
    find_center_pts = [x[0] for x in ocr_result[:3]]
    circle_center = get_circle_center(find_center_pts)
    return circle_center


def get_num(ocr_r: list, pt_det_r):
    ocr_r.sort(key=lambda x: x[-1])
    difference_dict = defaultdict(list)
    for i in range(1, len(ocr_r)):
        difference_dict[ocr_r[i][-1] - ocr_r[i - 1][-1]].append((i - 1, i))
    diff = max(difference_dict.items(), key=lambda x: len(x[-1]))
    
    diff=diff[0]
    insert_time = 0
    for k, v in difference_dict.items():
        if k > diff:
            for idx_pair in v:
                need_insert_idx = idx_pair[0] + insert_time
                ocr_r.insert(
                    need_insert_idx+1,
                    (
                        (
                            (ocr_r[need_insert_idx][0][0] +
                             ocr_r[need_insert_idx + 1][0][0]) / 2,
                            (ocr_r[need_insert_idx][0][1] +
                             ocr_r[need_insert_idx + 1][0][1]) / 2,
                        ),
                        diff + ocr_r[need_insert_idx][-1],
                    ),
                )
                insert_time += 1

    spt1=pt_det_r["1"][0][:2]
    spt2=pt_det_r["2"][0][:2]

    pt=pt_det_r["3"][0][:2]
    
    spt1_s_compare_func=partial(get_dist,*spt1)
    spt1_scale=min(ocr_r,key=lambda x:spt1_s_compare_func(*x[0]))[-1]
    spt2_s_compare_func=partial(get_dist,*spt2)
    spt2_scale=min(ocr_r,key=lambda x:spt2_s_compare_func(*x[0]))[-1]

    if spt1_scale==spt2_scale:
        return spt1_scale
    
    big_pt=spt1 if spt1_scale>spt2_scale else spt2
    big_s=spt1_scale if spt1_scale>spt2_scale else spt2_scale
    small_pt=spt1 if spt1_scale<spt2_scale else spt2
    small_s=spt1_scale if spt1_scale<spt2_scale else spt2_scale
    print("大刻度:{},小刻度:{}".format(big_s,small_s))


    scale_line_k,scale_line_b=find_normal_line(*spt1,*spt2)
    
    if scale_line_k is None:
        pt2s_lk=0
        pt2s_lb=pt[1]
    elif scale_line_k ==0:
        pt2s_lk=None
        pt2s_lb=pt[0]
    else:
        pt2s_lk=-1/scale_line_k
        pt2s_lb=pt[1]-pt2s_lk*pt[0]

    cross=find_cross_pt(scale_line_k,scale_line_b,pt2s_lk,pt2s_lb)

    bs_line_len=get_dist(*big_pt,*small_pt)
    pt2b_line_len=get_dist(*big_pt,*pt)
    pt2s_line_len=get_dist(*small_pt,*pt)

    bs_scale_diff=big_s-small_s
    if pt2b_line_len>bs_line_len and pt2b_line_len>pt2s_line_len:
        return small_s-(bs_scale_diff*pt2s_line_len)/bs_line_len
    elif pt2s_line_len>bs_line_len and pt2s_line_len>pt2b_line_len:
        return big_s+(bs_scale_diff*pt2b_line_len)/bs_line_len
    else:
        return small_s+(bs_scale_diff*pt2s_line_len)/bs_line_len

def main(args):
    model, ocr = init_model(args)
    model.eval()

    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    imgs_path = get_all_file_path(args.data_dir)

    with torch.no_grad():
        for img_path in imgs_path:
            print(img_path)
            img, resize_rate = img_deal(img_path, basic_transform, (416, 416))
            scm, srm, pmcm, pmrm = model(img.cuda())

            pmcm = filter_cm(pmcm)
            scm = filter_cm(scm)
            result = d_map(scm, srm, pmcm, pmrm, model.downsample_rate)

            # TODO:因为这里只有一张图片,简单处理
            pt_result = fix_pt_detect_result(result[0], resize_rate)
            ocr_result = ocr(img_path)
            num=get_num(ocr_result,pt_result)
            pprint(ocr_result)
            print(num)
            visimg=vis_img(img,result[0],num)
            D.show_img(visimg,cvreader=False)
            


if __name__ == "__main__":
    args = parse_args()
    main(args)
