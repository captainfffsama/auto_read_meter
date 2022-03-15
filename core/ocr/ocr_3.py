from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys
from pprint import pprint

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import json
import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program
import numpy as np
from tools.infer.predict_rec import TextRecognizer
from ppstructure.utility import init_args
from core.meter_det.darknet.d_yolo import DarknetDet
from core.status_enum import OCRStatus

import debug_tools as D


def get_ppocr_default_args():
    parser = init_args()
    parser.add_argument("--lang", type=str, default='ch')
    parser.add_argument("--det", type=bool, default=True)
    parser.add_argument("--rec", type=bool, default=True)
    parser.add_argument("--type", type=str, default='ocr')
    return parser.parse_args()


class ChiebotOCR(object):
    def __init__(self,
                 config_file,
                 names_file,
                 weights,
                 thr,
                 class_filter,
                 rec_model_dir,
                 rec_image_shape,
                 rec_char_dict_path,
                 rec_char_type=None,
                 **kwargs):
        args = get_ppocr_default_args()
        args.__dict__["rec_image_shape"] = rec_image_shape
        args.__dict__["rec_model_dir"] = rec_model_dir
        args.__dict__["rec_char_type"] = rec_char_type
        args.__dict__["rec_char_dict_path"] = rec_char_dict_path
        args.__dict__.update(**kwargs)
        self.det_model = DarknetDet(config_file,
                                    names_file,
                                    weights,
                                    thr,
                                    class_filter=class_filter)
        # self.rec_model = TextRecognizer(args)

    def rect2pt(self, rect):
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        return [[rect[0], rect[1]], [rect[0] + w, rect[1]],
                [rect[0]+w, rect[1] + h], [rect[0], rect[1] + h]]

    def __call__(self, img, cls=True):
        if isinstance(img, str):
            with open(img, 'rb') as f:
                np_arr = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                print("ERROR:img is empty")
                return None
        if isinstance(img, np.ndarray) and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        det_result, det_all_info = self.det_model(img)
        # rec_imgs = []
        # # TODO: 这里需要修改
        # for obj in det_result:
        #     rec_imgs.append(img[obj[2]:obj[4], obj[1]:obj[3], :])
        # rec_result, _ = self.rec_model(rec_imgs)
        return [[self.rect2pt(x[1:]), (x[0],float(y[1])*0.01)] for x,y in zip(det_result,det_all_info)]


class OCRModel(object):
    def __init__(self, *args, **kwargs):
        self.ocr_engine = ChiebotOCR(*args, **kwargs)
        self.h_thr = 8
        self.filter_thr = 0.7
        self._init_ocrmodel_arg(kwargs, ["h_thr", "filter_thr"])

    def _init_ocrmodel_arg(self, kwargs: dict, ocrmodel_arg: list):
        for arg in ocrmodel_arg:
            if arg in kwargs.keys():
                setattr(self, arg, kwargs.pop(arg))

    def filter_result(self, result):
        # 按照内容过滤
        filter_result = []
        for content in result:
            point_num = 0
            for c in content[-1][0]:
                if c == ".":
                    point_num += 1
            if point_num > 1:
                continue
            num_str: str = content[-1][0].replace(".", "").strip()
            if num_str.isnumeric() and (content[-1][1] > self.filter_thr) and (
                    len(content[-1][0]) < 6):
                x = 0
                y = 0
                h = (abs(content[0][0][1] - content[0][3][1]) +
                     abs(content[0][1][1] - content[0][2][1])) / 2
                for i in range(len(content[0])):
                    x += content[0][i][0]
                    y += content[0][i][1]

                x = int(x // 4)
                y = int(y // 4)

                if 0 < float(content[-1][0].strip()) < 500:
                    filter_result.append(
                        ((x, y), float(content[-1][0].strip()), h))

        mean = np.mean([x[-1] for x in filter_result])
        final_result = [
            x[:2] for x in filter_result if abs(x[-1] - mean) < self.h_thr
        ]
        return final_result

    def __call__(self, img, no_filter=False):
        result = self.ocr_engine(img, cls=False)
        if no_filter:
            final_result = result
        else:
            final_result = self.filter_result(result)
        status = OCRStatus.OK
        print("final_result: ", final_result)
        print("ocr real result: ", result)
        if not result:
            status = OCRStatus.NO_DETECT
        if len(final_result) < 2:
            status = OCRStatus.LESS_DETCT
        # if status !=OCRStatus.OK:
        #     breakpoint()
        return final_result, status

def draw_det_res(dt_boxes, config, img, img_name, save_path):
    if len(dt_boxes) > 0:
        import cv2
        src_im = img
        for box in dt_boxes:
            box = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, os.path.basename(img_name))
        cv2.imwrite(save_path, src_im)
        logger.info("The detected Image saved in {}".format(save_path))



@paddle.no_grad()
def main():
    global_config = config['Global']

    # build model
    model = build_model(config['Architecture'])

    load_model(config, model)
    # build post process
    post_process_class = build_post_process(config['PostProcess'])

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)

    ops = create_operators(transforms, global_config)

    save_res_path = config['Global']['save_res_path']
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.eval()
    with open(save_res_path, "wb") as fout:
        for file in get_image_file_list(config['Global']['infer_img']):
            logger.info("infer_img: {}".format(file))
            with open(file, 'rb') as f:
                img = f.read()
                data = {'image': img}
            batch = transform(data, ops)

            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            images = paddle.to_tensor(images)
            preds = model(images)
            post_result = post_process_class(preds, shape_list)

            src_img = cv2.imread(file)

            dt_boxes_json = []
            # parser boxes if post_result is dict
            if isinstance(post_result, dict):
                breakpoint()
                det_box_json = {}
                for k in post_result.keys():
                    boxes = post_result[k][0]['points']
                    dt_boxes_list = []
                    for box in boxes:
                        tmp_json = {"transcription": ""}
                        tmp_json['points'] = box.tolist()
                        dt_boxes_list.append(tmp_json)
                    det_box_json[k] = dt_boxes_list
                    save_det_path = os.path.dirname(config['Global'][
                        'save_res_path']) + "/det_results_{}/".format(k)
                    draw_det_res(boxes, config, src_img, file, save_det_path)
            else:
                boxes = post_result[0]['points']
                dt_boxes_json = []
                # write result
                for box in boxes:
                    tmp_json = {"transcription": ""}
                    tmp_json['points'] = box.tolist()
                    dt_boxes_json.append(tmp_json)
                save_det_path = os.path.dirname(config['Global'][
                    'save_res_path']) + "/det_results/"
                draw_det_res(boxes, config, src_img, file, save_det_path)
            otstr = file + "\t" + json.dumps(dt_boxes_json) + "\n"
            fout.write(otstr.encode())

    logger.info("success!")


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()
