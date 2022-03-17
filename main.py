"""
@Author: CaptainHu
@Date: 2021年 09月 23日 星期四 16:58:29 CST
@Description: 用来做前项推理
"""
import os
import argparse
from collections import defaultdict

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

from generate_xml import dump_xml

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
    def __init__(self,save_path,fps=10,size=(1280,720)):
        fourcc=cv2.VideoWriter_fourcc(*'MJPG')
        self.recoder=cv2.VideoWriter(save_path,fourcc,fps,size)

    def __enter__(self):
        return self.recoder

    def __exit__(self,type, value, traceback):
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

    with VideoRecorder("/data/1.avi") as v:
        for data in data_loader():
            img, img_path = data
            final_result, draw_info_container = model_iner(img)
            # save_xml(img,img_path,final_result)
            result_img = draw_frame(img, draw_info_container,True)
            # v.write(result_img)
            if "image" == data_loader.type:
                D.show_img(result_img)
            else:
                cv2.imshow("frame", result_img)
                inputkey = cv2.waitKey(20)
                if ord("q") == inputkey:
                    data_loader.stop()

def save_xml(img,img_path,result):
    if not result:
        result=[]
    obj_info=defaultdict(list)
    for box_info in result:
        obj_info["{:.2f}".format(box_info[-1])].append(box_info[0])
    out_path=img_path.replace('.jpg','.xml')
    img_name=os.path.basename(img_path)
    h,w,c=img.shape
    dump_xml([img_name,w,h,c],obj_info,out_path)


if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    main(cfg)
