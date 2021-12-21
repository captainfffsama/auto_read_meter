# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 12月 09日 星期四 11:11:37 CST
@Description: 使用ppocr进行检测
'''

from paddleocr import PaddleOCR,draw_ocr
from PIL import Image
import numpy as np
class PPOCR(object):
    def __init__(self):
        self.ocr_engine=PaddleOCR(use_angle_cls=False,lang="en")
        self.h_thr=4

    def filter_result(self,result,thr=0.89):
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

    def __call__(self,img):
        result=self.ocr_engine.ocr(img,cls=False)
        return self.filter_result(result)

def test(img_path):
    ocr = PaddleOCR(use_angle_cls=False, lang="en")  # need to run only once to download and load model into memory
    result:list = ocr.ocr(img_path, cls=False)
    for line in result:
        print(line)

    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='/home/chiebotgpuhq/MyCode/python/paddle/PaddleOCR/doc/fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.show()

if __name__=="__main__":
    img_path="/data/own_dataset/indoor_meter/debug_test/0db115efc8c73ba0c156763b3d8961c5.jpg"
    test(img_path)

