# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 12月 23日 星期四 16:45:54 CST
@Description: 前项推理专用,包含了一些特殊的逻辑策略
'''
from functools import partial
from collections import defaultdict
from pprint import pprint
from copy import deepcopy

import torch
import numpy as np

from core.tools.utils import get_circle_center, get_angle, get_dist, find_normal_line, find_cross_pt
from core.file_reader import img_deal
from core.tools.info_container import ImageInfoContainer, MeterInfoContainer, pos_fix
from core.pt_detection.pt_det_net import PtDetInfer
from core.ocr.pp_ocr import OCRModel
from core.meter_det.darknet.d_yolo import DarknetDet
from core.status_enum import OCRStatus

import debug_tools as D


class InferToolsMixin(object):
    def __init__(self):
        self.debug_info_container = ImageInfoContainer()

    def get_meter_img(self, img, meter_det_result):
        _, x1, y1, x2, y2 = meter_det_result
        meter_img = img[y1:y2, x1:x2, :]
        return meter_img, (x1, y1)

    def find_center(self, ocr_result):
        assert len(ocr_result) > 2, "orc pt is too less"
        find_center_pts = [x[0] for x in ocr_result[:3]]
        circle_center = get_circle_center(find_center_pts)
        return circle_center

    def find_dup_pt(self, pt, pt_det_r):
        if get_dist(*pt, *(pt_det_r["min_scale"][0][:2])) < 5:
            return "min_scale"
        if get_dist(*pt, *(pt_det_r["max_scale"][0][:2])) < 5:
            return "max_scale"

        return "none"

    def get_meter_center(self, pt_det_r, ori_hw):
        compute_pts = []
        compute_pts.append(pt_det_r["min_scale"][0][:2])
        compute_pts.append(pt_det_r["max_scale"][0][:2])

        label_concurrent = []
        if "none" == self.find_dup_pt(pt_det_r["small_label"][0][:2],
                                      pt_det_r):
            label_concurrent.append(pt_det_r["small_label"][0])
        if "none" == self.find_dup_pt(pt_det_r["big_label"][0][:2], pt_det_r):
            label_concurrent.append(pt_det_r["big_label"][0])

        if len(label_concurrent) > 1:
            good_pt = max(label_concurrent, key=lambda x: x[-1])[:2]
            compute_pts.append(good_pt)
        else:
            compute_pts.append(label_concurrent[0][:2])

        circle_pt = get_circle_center(compute_pts)
        if not (0 < circle_pt[1] < ori_hw[0] and 0 < circle_pt[0] < ori_hw[1]):
            return (-1, -1), -1
        radius = (get_dist(*circle_pt, *(pt_det_r["min_scale"][0][:2])) +
                  get_dist(*circle_pt, *(pt_det_r["max_scale"][0][:2]))) / 2
        return circle_pt, radius

    def get_dict_diff(self, diff_dict: dict):
        diff_count_list = [(k, len(v)) for k, v in diff_dict.items()]
        diff_count_list.sort(key=lambda x: x[-1], reverse=True)
        try:
            candicate_k = [
                diff_count_list[0][0],
            ]
        except Exception as e:
            raise e
            # breakpoint()
        for i in range(1, len(diff_count_list)):
            if diff_count_list[i][-1] == diff_count_list[0][-1]:
                candicate_k.append(diff_count_list[i][0])
            else:
                break
        candicate_k_t = [x for x in candicate_k if x > 0]
        if not candicate_k_t:
            # breakpoint()
            raise ValueError("candicate should not be empty")
        diff_k = min(candicate_k_t)
        return diff_k, diff_dict[diff_k]

    def fix_ocr_scale(self, ocr_r, meter_center, min_scale_pt, max_scale_pt):
        # TODO: need add extern fix
        ocr_r.sort(key=lambda x: x[-1])

        ocr_angle = [(get_angle(*i[0], *meter_center, True), i[-1])
                     for i in ocr_r]
        base_angle = min(ocr_angle, key=lambda x: x[0])[0]
        ocr_angle = [(int(i[0] - base_angle), i[1]) for i in ocr_angle]

        print("original ocr_angle:", ocr_angle)
        # 添加头尾刻度修
        min_scale_ag = get_angle(*min_scale_pt, *meter_center,
                                 True) - base_angle
        max_scale_ag = get_angle(*max_scale_pt, *meter_center,
                                 True) - base_angle
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
            ocr_angle.append((max_scale_ag, None))
        # TODO: 由于大部分的表在接近零的地方不是线性变化,这里最好单独处理
        if not zero_scale_is_exist:
            ocr_angle.insert(0, (min_scale_ag, 0))
            base_angle = base_angle + min_scale_ag
            # NOTE: 去除负角度,但是这样要求表的量程差不能大于180度
            ocr_angle = [(x[0] - min_scale_ag, x[1]) for x in ocr_angle
                         if (x[0] - min_scale_ag) >= 0]

        difference_dict = defaultdict(list)
        # TODO: 这里逻辑需要再次梳理
        start_tmp = 1
        end_tmp = len(ocr_angle) if max_scale_is_exist else len(ocr_angle) - 1
        for i in range(start_tmp, end_tmp):
            difference_dict[round(ocr_angle[i][-1] - ocr_angle[i - 1][-1],
                                  3)].append((i - 1, i))
        # XXX: 改了策略,但是不能应对所有情况
        if not difference_dict:
            print("difference_dict should not be empty")
            raise ValueError("difference_dict should not be empty")
        diff = self.get_dict_diff(difference_dict)

        angle_diff = 360
        for i in diff[-1]:
            if abs(ocr_angle[i[0]][0] - ocr_angle[i[1]][0]) < angle_diff:
                angle_diff = abs(ocr_angle[i[0]][0] - ocr_angle[i[1]][0])

        scale_diff = diff[0]
        print("scale diff is {}".format(scale_diff))
        pprint(difference_dict)
        print("angle diff is {}".format(angle_diff))
        if not max_scale_is_exist:
            # TODO: 这里补全的规则还要考量一下,目前补全方式也不够科学
            ocr_angle[-1] = (ocr_angle[-1][0], max_scale + round(
                (ocr_angle[-1][0] - ocr_angle[-2][0]) / angle_diff) *
                             scale_diff)
            difference_dict[round(ocr_angle[-1][-1] - ocr_angle[-2][-1],
                                  3)].append(
                                      (len(ocr_angle) - 2, len(ocr_angle) - 1))
        print("add 0 and max:", ocr_angle)

        # 中间补足
        insert_time = 0
        # difference_dict k为刻度差,v中每个元素都是一对索引,指示了ocr_angle中对应的元素
        for k, v in difference_dict.items():
            current_insert_time = (k // scale_diff)
            for idx_pair in v:
                #NOTE: 这样插值更加准确
                angle_diff=(ocr_angle[idx_pair[1]][0] - ocr_angle[idx_pair[0]][0])/current_insert_time
                for i in range(1, int(current_insert_time)):
                    need_insert_idx = idx_pair[0] + insert_time
                    ocr_angle.insert(
                        need_insert_idx + 1,
                        (
                            ocr_angle[need_insert_idx][0] + angle_diff,
                            scale_diff+ ocr_angle[need_insert_idx][-1],
                        ),
                    )
                    insert_time += 1
        # if not zero_scale_is_exist:
        #     ocr_angle.insert(1, ((ocr_angle[0][0] + ocr_angle[1][0]) / 2,
        #                          (ocr_angle[0][1] + ocr_angle[1][1]) / 2))
        print("fix over:", ocr_angle)

        return ocr_angle, base_angle

    def get_num(self, ocr_angle: list, base_angle, circle_pt, pt_det_r):
        spt1 = pt_det_r["small_label"][0][:2]
        spt2 = pt_det_r["big_label"][0][:2]
        spt1_angle = get_angle(*spt1, *circle_pt, True) - base_angle
        spt2_angle = get_angle(*spt2, *circle_pt, True) - base_angle
        #两者重合了
        #TODO: 这里处理还需要细化
        if abs(spt1_angle - spt2_angle) < 1:
            print("spt1 spt2 is same,spt1 angle is{},spt2 angle is {}".format(
                spt1_angle, spt2_angle))
            spt2_scale = min(ocr_angle,
                             key=lambda x: abs(x[0] - spt2_angle))[-1]
            return spt2_scale

        pt = pt_det_r["pointer"][0][:2]
        pt_angle = get_angle(*pt, *circle_pt, True) - base_angle

        spt1_scale_sort_list = sorted(ocr_angle,
                                      key=lambda x: abs(x[0] - spt1_angle))
        spt2_scale_sort_list = sorted(ocr_angle,
                                      key=lambda x: abs(x[0] - spt2_angle))
        print("sp1 angle:",spt1_angle)
        print("spt2 angle:",spt2_angle)
        if spt1_scale_sort_list[0] == spt2_scale_sort_list[0]:
            if abs(spt1_scale_sort_list[0][0] -
                   spt1_angle) > abs(spt2_scale_sort_list[0][0] - spt2_angle):
                spt1_scale = spt1_scale_sort_list[1][-1]
                spt2_scale = spt2_scale_sort_list[0][-1]
            else:
                spt1_scale = spt1_scale_sort_list[0][-1]
                spt2_scale = spt2_scale_sort_list[1][-1]
        else:
            spt1_scale = spt1_scale_sort_list[0][-1]
            spt2_scale = spt2_scale_sort_list[0][-1]

        # 若两者过近则认为是指到了同一刻度
        if abs(spt1_angle - spt2_angle) < 3:
            return spt1_scale

        # if spt1_scale == spt2_scale:
        #     print("detect some thing wrong")
        #     return spt1_scale

        big_anlge = spt1_angle if spt1_scale > spt2_scale else spt2_angle
        big_s = spt1_scale if spt1_scale > spt2_scale else spt2_scale
        small_angle = spt1_angle if spt1_scale < spt2_scale else spt2_angle
        small_s = spt1_scale if spt1_scale < spt2_scale else spt2_scale
        print("大刻度:{},小刻度:{}".format(big_s, small_s))

        num = (pt_angle - small_angle) / (big_anlge - small_angle) * (
            big_s - small_s) + small_s
        return num


class Infer(InferToolsMixin):
    def __init__(self,
                 meter_det: DarknetDet,
                 ocr_model: OCRModel,
                 pt_model: PtDetInfer,
                 img_resize=(416, 416),
                 base_trans=None):
        self.meter_det = meter_det
        self.ocr_model = ocr_model
        self.pt_model = pt_model
        self.pt_model_input_precess = partial(
            img_deal, **dict(img_resize=img_resize,
                             basic_transform=base_trans))
        self.debug_info_container = ImageInfoContainer()

    def __call__(self, ori_img: np.ndarray):
        self.debug_info_container.clear()
        final_result = []
        with torch.no_grad():
            meter_det_result, _ = self.meter_det(ori_img)
            # DEBUG:
            print("meter_det_result:  ", meter_det_result)
            if len(meter_det_result) < 1:
                self.debug_info_container.global_info.append(
                    "can no find meter")
                return None, self.debug_info_container
            for one_md_result in meter_det_result:
                obj_debug_info = MeterInfoContainer()
                obj_debug_info.meter_rect = tuple(one_md_result[1:])
                if not all(one_md_result[1:]):
                    obj_debug_info.messages.append("one meter dect failed")
                    self.debug_info_container.meters_info.append(
                        obj_debug_info)
                    continue
                meter_img, fix_pos = self.get_meter_img(ori_img, one_md_result)

                # NOTE: 很奇怪的会出现一些shape为0的 meter_img
                if not all(meter_img.shape):
                    continue
                img, resize_rate, ori_hw = self.pt_model_input_precess(
                    meter_img)
                pt_result = self.pt_model(img, resize_rate)
                obj_debug_info.pt_result = {
                    k: [(*pos_fix(x[:2], fix_pos), x[-1]) for x in v]
                    for k, v in pt_result.items()
                }
                ocr_result, ocr_status = self.ocr_model(meter_img)

                obj_debug_info.ocr_result = [(pos_fix(x[0], fix_pos), x[-1])
                                             for x in ocr_result]
                try:
                    circle_pt, radius = self.get_meter_center(pt_result, ori_hw)
                except Exception as e:
                    print("can not find meter center")
                    continue
                obj_debug_info.circle_pt = circle_pt

                if ocr_status != OCRStatus.OK:
                    pprint(ocr_result)
                    obj_debug_info.messages.append("ocr result too less")
                    self.debug_info_container.meters_info.append(
                        obj_debug_info)
                    continue

                if radius < 0:
                    obj_debug_info.messages.append(
                        "can not find circle center")
                    self.debug_info_container.meters_info.append(
                        obj_debug_info)
                    continue
                if ocr_status == OCRStatus.OK:
                    try:
                        ocr_angle, base_angle = self.fix_ocr_scale(
                            ocr_result, circle_pt, pt_result["min_scale"][0][:2],
                            pt_result["max_scale"][0][:2])
                        num = self.get_num(ocr_angle, base_angle, circle_pt,
                                       pt_result)
                    except Exception as e:
                        num=-1
                        pass
                    obj_debug_info.num = num
                else:
                    num = -1
                final_result.append((tuple(one_md_result[1:]), num))
                self.debug_info_container.meters_info.append(obj_debug_info)

        return final_result, self.debug_info_container
