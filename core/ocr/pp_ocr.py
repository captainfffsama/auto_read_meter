# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 12月 09日 星期四 11:11:37 CST
@Description: 使用ppocr进行检测
'''

from paddleocr import PaddleOCR,draw_ocr
from PIL import Image

class PPOCR(object):
    def __init__(self):
        self.ocr_engine=PaddleOCR(use_angle_cls=False,lang="en")

    def filter_result(self,result,thr=0.85):
        final_result=[]
        for content in result:
            num_str:str=content[-1][0].replace("-","").replace(".","").strip()
            if num_str.isnumeric() and (content[-1][1]>thr):
                final_result.append((content[0][0],float(content[-1][0].strip())))

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
    img_path="/data/own_dataset/indoor_meter/VA_indoor_meter/change_label/train/add1/a4b1736ffaddcd1b4d9beccaa206fc92.jpg"
    test(img_path)

