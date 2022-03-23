import streamlit as st


import os
import argparse
from collections import defaultdict

import cv2
import numpy as np

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


config_path="/home/chiebotgpuhq/MyCode/python/meter_auto_read/config/cfgv5.yaml"



@st.cache(allow_output_mutation=True)
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

@st.cache(allow_output_mutation=True)
def get_inference_model():

    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_path)
    cfg.freeze()

    model, ocr, meter_det = init_model(cfg)

    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    model_infer = Infer(meter_det,
                        ocr,
                        model,
                        img_resize=(416, 416),
                        base_trans=basic_transform)
    return model_infer

upload_file=st.file_uploader("choose a jpg file")
model_infer=get_inference_model()
if upload_file is not None:
    bytes_data = upload_file.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    st.write("输入图片")
    st.image(img[:,:,::-1])
    final_result, draw_info_container = model_infer(img)
    # save_xml(img,img_path,final_result)
    result_img = draw_frame(img, draw_info_container,True)
    st.write("输出图片")
    st.image(result_img[:,:,::-1])

