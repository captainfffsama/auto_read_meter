# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2022年 03月 16日 星期三 16:32:38 CST
@Description:
'''
import os
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Sequence, Mapping

import cv2

import torch
from torchvision import transforms

from core.config.base_config import get_cfg_defaults

from core.pt_detection.pt_det_net import PtDetInfer, PointDetectNet
from core.ocr.ocr_3 import OCRModel
# from core.ocr.pp_ocr import OCRModel
from core.meter_det.darknet.d_yolo import DarknetDet

from core.file_reader import DataSequence
from core.inference import Infer
from core.tools.vis import draw_frame

from generate_xml import parse_xml_info

import debug_tools as D


def parse_args():
    parser = argparse.ArgumentParser(description="infer args set")
    parser.add_argument(
        "--cfg",
        type=str,
        default="./config/cfgv5.yaml",
        help="",
    )
    args = parser.parse_args()
    return args


def init_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PointDetectNet(3).to(device)
    if os.path.exists(args.PT_DET.ckpt):
        params = torch.load(args.PT_DET.ckpt)
        model.load_state_dict(params, strict=False)
    model = PtDetInfer(model)

    ocr = OCRModel(**args.OCR_TEST)

    meter_det = DarknetDet(**args.METER_DET)

    return model, ocr, meter_det


class VideoRecorder(object):
    def __init__(self, save_path, fps=10, size=(1280, 720)):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.recoder = cv2.VideoWriter(save_path, fourcc, fps, size)

    def __enter__(self):
        return self.recoder

    def __exit__(self, type, value, traceback):
        self.recoder.release()


def main(args):
    model, ocr, meter_det = init_model(args)
    data_loader = DataSequence(args.data_path)

    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    model_iner = Infer(meter_det,
                       ocr,
                       model,
                       img_resize=(416, 416),
                       base_trans=basic_transform)

    all_det_tp = 0
    all_det_fp = 0
    all_det_fn = 0
    all_read_tp = 0
    all_read_fp = 0
    for data in data_loader():
        img, img_path = data
        final_result, draw_info_container = model_iner(img)
        _, obj_info = parse_xml_info(img_path.replace(".jpg", ".xml"))
        print("=========================================")
        det_tp, det_fp, det_fn, read_tp, read_fp = get_tpfpfn(
            final_result, obj_info)

        all_det_tp += det_tp
        all_det_fp += det_fp
        all_det_fn += det_fn
        all_read_tp += read_tp
        all_read_fp += read_fp

    det_precision, det_recall, read_precision, total_precision, total_recall = count_benchmark(
        all_det_tp, all_det_fp, all_det_fn, all_read_tp, all_read_fp)

    print("det_precision: {}".format(det_precision))
    print("det_recall: {}".format(det_recall))
    print("read_precision: {}".format(read_precision))
    print("total_precision: {}".format(total_precision))
    print("total_recall : {}".format(total_recall))

def get_iou(r1, r2):
    r1_a = (r1[2] - r1[0]) * (r1[3] - r1[1])
    r2_a = (r2[2] - r2[0]) * (r2[3] - r2[1])

    i = (max(r1[0], r2[0]), max(r1[1], r2[1]), min(r1[2],
                                                   r2[2]), min(r1[3], r2[3]))
    if (i[2] - i[0])<0 or (i[3] - i[1])<0:
        return 0
    i_a = (i[2] - i[0]) * (i[3] - i[1])
    return i_a / (r1_a + r2_a-i_a)


def get_tpfpfn(final_result: Sequence[Tuple[Tuple[int, int, int, int], float]],
               obj_info: Mapping[str, Sequence[Tuple[int, int, int, int]]],
               thr: float = 0.3):
    det_tp = 0
    det_fp = 0
    det_fn = 0
    read_tp = 0
    read_fp = 0
    gt_info = []
    for k, v in obj_info.items():
        gt_info.extend([(a, k) for a in v])
    if not final_result:
        final_result =[]

    for obj_pre in final_result:
        for gt in gt_info[:]:
            iou = get_iou(obj_pre[0], gt[0])
            if iou > thr:
                det_tp += 1
                reduce_v = (abs(obj_pre[1] - float(gt[1]))+0.000001) / (float(gt[1])+0.000001)
                if reduce_v > 0.05:
                    read_fp += 1
                else:
                    read_tp += 1
                gt_info.remove(gt)
            elif iou<=thr and iou > 0:
                det_fp+=1
            else:
                pass
    det_fn += len(gt_info)

    return det_tp, det_fp, det_fn, read_tp, read_fp


def count_benchmark(det_tp, det_fp, det_fn, read_tp, read_fp):
    det_precision = det_tp / (det_tp + det_fp)
    det_recall = det_tp / (det_tp + det_fn)

    read_precision = read_tp / (read_tp + read_fp)

    total_precision = read_tp / (det_tp + det_fp)
    total_recall = read_tp / (det_tp + det_fn)

    return det_precision, det_recall, read_precision, total_precision, total_recall


if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    main(cfg)
