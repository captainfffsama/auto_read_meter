# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 12月 21日 星期二 16:00:40 CST
@Description:
'''
import argparse

from paddleocr import PaddleOCR,draw_ocr
from PIL import Image
import numpy as np
import cv2
import numpy as np
import paddleocr as pocr
from tools.infer import predict_system
from ppstructure.utility import init_args

def get_ppocr_default_args():
    parser=init_args()
    parser.add_argument("--lang", type=str, default='ch')
    parser.add_argument("--det", type=bool, default=True)
    parser.add_argument("--rec", type=bool, default=True)
    parser.add_argument("--type", type=str, default='ocr')
    return parser.parse_args()

class ChiebotOCR(predict_system.TextSystem):
    def __init__(self,det_model_dir,rec_model_dir,rec_image_shape,rec_char_dict_path,rec_char_type=None,**kwargs):
        args=get_ppocr_default_args()
        args.__dict__["det_model_dir"]=det_model_dir
        args.__dict__["rec_image_shape"]=rec_image_shape
        args.__dict__["rec_model_dir"]=rec_model_dir
        args.__dict__["rec_char_type"]=rec_char_type
        args.__dict__["rec_char_dict_path"]=rec_char_dict_path
        args.__dict__.update(**kwargs)
        super().__init__(args)

    def __call__(self,img,cls=True):
        if isinstance(img, str):
            with open(img, 'rb') as f:
                np_arr = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                print("ERROR:img is empty")
                return None
        if isinstance(img, np.ndarray) and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        dt_boxes, rec_res = super().__call__(img, cls)
        return [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]


class OCRModel (object):
    def __init__(self,*args,**kwargs):
        self.ocr_engine=ChiebotOCR(*args,**kwargs)
        self.h_thr=8

    def filter_result(self,result,thr=0.7):
        # 按照内容过滤
        filter_result=[]
        for content in result:
            point_num=0
            for c in content[-1][0]:
                if c==".":
                    point_num+=1
            if point_num>1:
                continue
            num_str:str=content[-1][0].replace(".","").strip()
            if num_str.isnumeric() and (content[-1][1]>thr) and (len(content[-1][0])<6):
                x=0
                y=0
                h=(abs(content[0][0][1]-content[0][3][1])+abs(content[0][1][1]-content[0][2][1]))/2
                for i in range(len(content[0])):
                    x+=content[0][i][0]
                    y+=content[0][i][1]

                x=int(x//4)
                y=int(y//4)

                if 0<float(content[-1][0].strip())<500:
                    filter_result.append(((x,y),float(content[-1][0].strip()),h))

        mean=np.mean([x[-1] for x in filter_result])
        final_result=[x[:2] for x in filter_result if abs(x[-1]-mean)<self.h_thr]
        return final_result

    def __call__(self,img,no_filter=False):
        result=self.ocr_engine(img,cls=False)
        if no_filter:
            return result
        return self.filter_result(result)

if __name__ == "__main__":
    DET_MODEL_DIR= "/home/chiebotgpuhq/MyCode/python/meter_auto_read/model_weight/ppocr/detec"
    REC_MODEL_DIR= "/home/chiebotgpuhq/MyCode/python/meter_auto_read/model_weight/ppocr/reg/2"
    REC_CHAR_TYPE= "en"
    REC_IMG_SHAPE= "3,32,100"
    REC_CHAR_DICT_PATH="/home/chiebotgpuhq/MyCode/python/paddle/PaddleOCR/ppocr/utils/ppocr_keys_v1.txt"
    a=ChiebotOCR(DET_MODEL_DIR,REC_MODEL_DIR,REC_IMG_SHAPE,REC_CHAR_DICT_PATH,REC_IMG_SHAPE)
