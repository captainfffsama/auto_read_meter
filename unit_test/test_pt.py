# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 12月 27日 星期一 17:12:08 CST
@Description: 用来测试pt部分网络性能
'''

import base64
import os
import json
from copy import deepcopy

import torch
from collections import defaultdict
import cv2
from core.pt_detection.pt_det_net import PtDetInfer, PointDetectNet
from core.file_reader import DataSequence,img_deal
from core.tools.info_container import ImageInfoContainer, MeterInfoContainer
from core.tools.vis import draw_frame
from tqdm import tqdm
from pt_label_template import PT_LABEL_TMP

import debug_tools as D

def init_model(ckpt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PointDetectNet(3).to(device)
    if os.path.exists(ckpt):
        params = torch.load(ckpt)
        model.load_state_dict(params, strict=False)
    model = PtDetInfer(model)

    return model

def generate_label(img_path,save_dir,result):
    json_content=deepcopy(PT_LABEL_TMP)
    json_content["imagePath"]=os.path.basename(img_path)
    h,w,c=cv2.imread(img_path).shape
    json_content["imageHeight"]=h
    json_content["imageWidth"]=w
    with open(img_path, "rb") as f:
        base64_data = base64.b64encode(f.read())
    base64_data=str(base64_data).replace("b'","").replace("'\"","\"")
    json_content["imageData"]=base64_data
    for k,v in result.items():
        content={"label":k,"points":[list(i[:2]) for i in v],"group_id": None,"shape_type":"point","flags":{}}
        json_content["shapes"].append(content)

    with open(os.path.join(save_dir,json_content["imagePath"].replace(".jpg",".json")),"w") as fw:
        json.dump(json_content,fw,indent=2,ensure_ascii=True)

    print("{} save done!".format(img_path))

def main(img_path,det_model,save_dir="",save_crop=False,show=True,gen_label=False):
    image_container=ImageInfoContainer()
    data_loader=DataSequence(img_path)
    for data in tqdm(data_loader()):
        image_container.clear()
        meter_container=MeterInfoContainer()
        ori_img,img_path=data
        img, resize_rate, ori_hw = img_deal(ori_img,(416,416))
        r=det_model(img,resize_rate)
        print(r)
        meter_container.pt_result=r
        image_container.meters_info.append(meter_container)
        result_img = draw_frame(ori_img,image_container)
        if show:
            if "image" == data_loader.type:
                D.show_img(result_img)
            else:
                cv2.imshow("frame",result_img)
                inputkey = cv2.waitKey(20)
                if ord("q") == inputkey:
                    data_loader.stop()
        if save_dir and gen_label:
            generate_label(img_path,save_dir,r)
            print("{} label save done!".format(img_path))

if __name__ == "__main__":
    ckpt="/data/experiments_data/point_weight/3pt_1.ckpt"
    det_model = init_model(ckpt)
    img_path="/data/own_dataset/indoor_meter/VA_indoor_meter/ex3/another"
    # img_path="/data/own_dataset/indoor_meter/VA_indoor_meter/20211207"
    # img_path="/home/chiebotgpuhq/Pictures/摄像头"
    save_dir="/data/own_dataset/indoor_meter/VA_indoor_meter/ex3/another"
    main(img_path,det_model,save_dir,False,False,True)
