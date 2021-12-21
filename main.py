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
from core.utils import get_circle_center, get_angle, get_dist, find_normal_line, find_cross_pt

import debug_tools as D


def parse_args():
    parser = argparse.ArgumentParser(description="infer args set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/experiments_data/point_weight/3pt.ckpt",
        help="",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=
        # "/data/own_dataset/indoor_meter/VA_indoor_meter/change_label/test")
        "/data/own_dataset/indoor_meter/debug_test")
        # "/home/chiebotgpuhq/Share/win_share/22222")
        # "/data/own_dataset/indoor_meter/自己拍摄_电压表")
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


def decode_feature_map(c_feature_map, r_feature_map, label_list, result_dict):
    ori_cm_shape = c_feature_map.shape[1:]
    for i, label in enumerate(label_list):
        pt_idx = c_feature_map[i].argmax()
        pt_yi = pt_idx // ori_cm_shape[-1]
        pt_xi = pt_idx % ori_cm_shape[0]
        pt_score = c_feature_map[i, pt_yi, pt_xi]

        pt_x = r_feature_map[2 * i, pt_yi, pt_xi]
        pt_y = r_feature_map[2 * i + 1, pt_yi, pt_xi]
        result_dict[label].append(
            (pt_x.item(), pt_y.item(), 0, pt_score.item()))


def d_map(scm, srm, pmcm, pmrm, rcm, rrm, d_rate):
    result: List[dict] = []

    for one_img_scm, one_img_srm, one_img_pmcm, one_img_pmrm, one_img_rcm, one_img_rrm in zip(
            scm, srm, pmcm, pmrm, rcm, rrm):
        ori_cm_shape = one_img_scm.shape[1:]
        one_img_result = defaultdict(list)
        decode_feature_map(one_img_scm, one_img_srm,
                           ("min_scale", "max_scale"), one_img_result)
        decode_feature_map(one_img_pmcm, one_img_pmrm, ("pointer", ),
                           one_img_result)
        decode_feature_map(one_img_rcm, one_img_rrm,
                           ("small_label", "big_label"), one_img_result)

        result.append(one_img_result)

    return result


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range * 255


def vis_img(img, result, num):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy().squeeze()
        img = np.transpose(img, (1, 2, 0))
        img = normalization(img)
        img = img.astype(np.uint8).copy()
    font = cv2.FONT_HERSHEY_PLAIN

    color_map = {
        "min_scale": (0, 0, 255),
        "max_scale": (0, 255, 0),
        "pointer": (255, 0, 0),
        "small_label": (255, 255, 0),
        "big_label": (0, 255, 255),
    }

    for k, v in result.items():
        for v_i in v:
            pos = (int(v_i[0]), int(v_i[1]))
            cv2.circle(img, pos, 3, color_map[k], -1)

            cv2.putText(img, "{}/{:.2f}".format(k, v_i[-1]), pos, font, 1,
                        color_map[k], 1)

    cv2.putText(img, str(num), (40, 40), font, 2, (255, 255, 255), 2)

    return img[:, :-1]


def fix_pt_detect_result(result_ori: dict, resize_rate: float):
    result = deepcopy(result_ori)
    for k in result.keys():
        result[k] = [(x[0] / resize_rate, x[1] / resize_rate, x[2])
                     for x in result[k]]
    return result


def find_center(ocr_result):
    assert len(ocr_result) > 2, "orc pt is too less"
    find_center_pts = [x[0] for x in ocr_result[:3]]
    circle_center = get_circle_center(find_center_pts)
    return circle_center


def find_dup_pt(pt, pt_det_r):
    if get_dist(*pt, *(pt_det_r["min_scale"][0][:2])) < 5:
        return "min_scale"
    if get_dist(*pt, *(pt_det_r["max_scale"][0][:2])) < 5:
        return "max_scale"

    return "none"


def get_meter_center(pt_det_r, ori_hw):
    compute_pts = []
    compute_pts.append(pt_det_r["min_scale"][0][:2])
    compute_pts.append(pt_det_r["max_scale"][0][:2])

    label_concurrent = []
    if "none" == find_dup_pt(pt_det_r["small_label"][0][:2], pt_det_r):
        label_concurrent.append(pt_det_r["small_label"][0])
    if "none" == find_dup_pt(pt_det_r["big_label"][0][:2], pt_det_r):
        label_concurrent.append(pt_det_r["big_label"][0])

    if len(label_concurrent) > 1:
        good_pt = max(label_concurrent, key=lambda x: x[-1])[:2]
        compute_pts.append(good_pt)
    else:
        compute_pts.append(label_concurrent[0][:2])

    circle_pt = get_circle_center(compute_pts)
    if not (0 < circle_pt[1] < ori_hw[0] and 0 < circle_pt[0] < ori_hw[1]):
        # FIXME:
        breakpoint()
    radius = (get_dist(*circle_pt, *(pt_det_r["min_scale"][0][:2])) +
              get_dist(*circle_pt, *(pt_det_r["max_scale"][0][:2]))) / 2
    return circle_pt, radius


def fix_ocr_scale(ocr_r, meter_center, min_scale_pt, max_scale_pt):
    # TODO: need add extern fix
    ocr_r.sort(key=lambda x: x[-1])

    difference_dict = defaultdict(list)
    for i in range(1, len(ocr_r)):
        difference_dict[round(ocr_r[i][-1] - ocr_r[i - 1][-1], 3)].append(
            (i - 1, i))
    # FIXME:这里可能会有概率引起bug
    diff = max(difference_dict.items(), key=lambda x: len(x[-1]))

    ocr_angle = [(get_angle(*i[0], *meter_center, True), i[-1]) for i in ocr_r]
    base_angle = min(ocr_angle, key=lambda x: x[0])[0]
    ocr_angle = [(int(i[0] - base_angle), i[1]) for i in ocr_angle]
    angle_diff = 360
    for i in diff[-1]:
        if abs(ocr_angle[i[0]][0] - ocr_angle[i[1]][0]) < angle_diff:
            angle_diff = abs(ocr_angle[i[0]][0] - ocr_angle[i[1]][0])

    print("original ocr_angle:",ocr_angle)

    scale_diff = diff[0]
    # 添加头尾刻度修
    min_scale_ag = get_angle(*min_scale_pt, *meter_center, True) - base_angle
    max_scale_ag = get_angle(*max_scale_pt, *meter_center, True) - base_angle
    zero_scale_is_exist = False
    max_scale_is_exist = False
    max_scale = 0.0
    for ag_scale in ocr_angle:
        if ag_scale[-1] == 0:
            zero_scale_is_exist = True
        if abs(ag_scale[0] - max_scale_ag) < 5:
            max_scale_is_exist = True
        if ag_scale[-1] > max_scale:
            max_scale = ag_scale[-1]

    # pprint(ocr_angle)
    # pprint(max_scale_ag)

    if not max_scale_is_exist:
        ocr_angle.append((max_scale_ag, max_scale + round((max_scale_ag-ocr_angle[-1][0])/angle_diff)*scale_diff))
    # TODO: 由于大部分的表在接近零的地方不是线性变化,这里最好单独处理
    if not zero_scale_is_exist:
        ocr_angle.insert(0, (min_scale_ag, 0))
        base_angle = base_angle + min_scale_ag
        # NOTE: 去除负角度,但是这样要求表的量程差不能大于180度
        ocr_angle = [(x[0] - min_scale_ag, x[1]) for x in ocr_angle]
        for k,v in difference_dict.items():
            difference_dict[k]=[(x[0]+1,x[1]+1) for x in v]
        difference_dict[round(ocr_angle[1][-1]-ocr_angle[0][-1], 3)].append((0,1))
        if not max_scale_is_exist:
            print("add max!!!")
            difference_dict[round(ocr_angle[-1][-1]-ocr_angle[-2][-1],3)].append((len(ocr_angle)-2,len(ocr_angle)-1))
    print("add 0 and max:",ocr_angle)

    insert_time = 0
    for k, v in difference_dict.items():
        current_insert_time = k // scale_diff
        for idx_pair in v:
            for i in range(1, int(current_insert_time)):
                need_insert_idx = idx_pair[0] + insert_time
                ocr_angle.insert(
                    need_insert_idx + 1,
                    (
                        ocr_angle[need_insert_idx][0] + angle_diff * i,
                        scale_diff * i + ocr_angle[need_insert_idx][-1],
                    ),
                )
                insert_time += 1

    print("fix over:",ocr_angle)

    return ocr_angle, base_angle


def get_num(ocr_angle: list, base_angle, circle_pt, pt_det_r):
    # s2:fill ocr result
    # s3:find matching between pt det and orc
    # s4:get num

    spt1 = pt_det_r["small_label"][0][:2]
    spt2 = pt_det_r["big_label"][0][:2]
    spt1_angle = get_angle(*spt1, *circle_pt, True) - base_angle
    spt2_angle = get_angle(*spt2, *circle_pt, True) - base_angle

    pt = pt_det_r["pointer"][0][:2]
    pt_angle = get_angle(*pt, *circle_pt, True) - base_angle

    spt1_scale = min(ocr_angle, key=lambda x: abs(x[0] - spt1_angle))[-1]
    spt2_scale = min(ocr_angle, key=lambda x: abs(x[0] - spt2_angle))[-1]

    if spt1_scale == spt2_scale:
        print("detect some thing wrong")
        return spt1_scale

    big_anlge = spt1_angle if spt1_scale > spt2_scale else spt2_angle
    big_s = spt1_scale if spt1_scale > spt2_scale else spt2_scale
    small_angle = spt1_angle if spt1_scale < spt2_scale else spt2_angle
    small_s = spt1_scale if spt1_scale < spt2_scale else spt2_scale
    print("大刻度:{},小刻度:{}".format(big_s, small_s))

    num = (pt_angle - small_angle) / (big_anlge - small_angle) * (
        big_s - small_s) + small_s
    return num


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
            img, resize_rate, ori_hw = img_deal(img_path, basic_transform,
                                                (416, 416))
            scm, srm, pmcm, pmrm, rcm, rrm = model(img.cuda())

            pmcm = filter_cm(pmcm)
            scm = filter_cm(scm)
            rcm = filter_cm(rcm)
            result = d_map(scm, srm, pmcm, pmrm, rcm, rrm,
                           model.downsample_rate)

            # TODO:因为这里只有一张图片,简单处理
            pt_result = fix_pt_detect_result(result[0], resize_rate)
            ocr_result = ocr(img_path)
            if len(ocr_result)<2:
                pprint(ocr_result)
                print("ocr result too less")
                continue

            circle_pt, radius = get_meter_center(pt_result, ori_hw)
            ocr_angle, base_angle = fix_ocr_scale(
                ocr_result, circle_pt, pt_result["min_scale"][0][:2],
                pt_result["max_scale"][0][:2])

            num = get_num(ocr_angle, base_angle, circle_pt, pt_result)
            visimg = vis_img(img, result[0], num)
            D.show_img(visimg, cvreader=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
