# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 12月 21日 星期二 15:42:50 CST
@Description: yacs用的基础配置,不要改
'''

import os
from yacs.config import CfgNode as CN

CURRENT_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
_C = CN()
_C.METER_DET = CN()
_C.METER_DET.config_file = ""
_C.METER_DET.names_file = ""
_C.METER_DET.weights = ""
_C.METER_DET.thr = 0.5
_C.METER_DET.batch_size = 1
_C.METER_DET.class_filter = ()

_C.OCR = CN()
_C.OCR.det_model_dir = ""

_C.OCR.rec_model_dir = ""
_C.OCR.rec_char_type = ""
_C.OCR.rec_image_shape = " 3,32,100"
_C.OCR.rec_char_dict_path = os.path.join(CURRENT_FILE_DIR,
                                             "../../en_dict.txt")
_C.OCR.h_thr=8
_C.OCR.filter_thr=0.7

_C.PT_DET = CN()
_C.PT_DET.ckpt = ""

_C.data_path = ""


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
