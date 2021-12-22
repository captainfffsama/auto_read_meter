# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 12月 21日 星期二 15:42:50 CST
@Description: yacs用的基础配置,不要改
'''

import os
from yacs.config import CfgNode as CN

CURRENT_FILE_DIR=os.path.dirname(os.path.realpath(__file__))
_C=CN()
_C.METER_DET=CN()

_C.OCR=CN()
_C.OCR.DET=CN()
_C.OCR.DET.DET_MODEL_DIR=""

_C.OCR.REG=CN()
_C.OCR.REG.REC_MODEL_DIR=""
_C.OCR.REG.REC_CHAR_TYPE=""
_C.OCR.REG.REC_IMAGE_SHAPE=""
# _C.OCR.REG.REC_CHAR_DICT_PATH="/home/chiebotgpuhq/MyCode/python/paddle/PaddleOCR/ppocr/utils/ppocr_keys_v1.txt"
_C.OCR.REG.REC_CHAR_DICT_PATH=os.path.join(CURRENT_FILE_DIR,"../../en_dict.txt")

_C.PT_DET=CN()
_C.PT_DET.CKPT=""

_C.IMG_DIR=""

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()