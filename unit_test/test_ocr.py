
import os
import math
import argparse
from pprint import pprint

from paddleocr import PaddleOCR, draw_ocr
# 显示结果
from PIL import Image
from core.config.base_config import get_cfg_defaults
from core.ocr.pp_ocr import OCRModel
from core.file_reader import get_all_file_path

def parse_args():
    parser = argparse.ArgumentParser(description="infer args set")
    parser.add_argument(
        "--cfg",
        type=str,
        default="./config/cfgv2.yaml",
        help="",
    )
    args = parser.parse_args()
    return args

def draw_ocr_result(ocr_model,img_path):
    result,status = ocr_model(img_path, no_filter=True)

    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./doc/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.show(title=img_path)

def main(args):
    ocr =OCRModel(**args.OCR)

    imgs_path = get_all_file_path(args.data_path)

    for img_path in imgs_path:
        print(img_path)
        pprint(ocr(img_path))
        draw_ocr_result(ocr,img_path)
        breakpoint()
if __name__ == '__main__':
    args = parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    main(cfg)