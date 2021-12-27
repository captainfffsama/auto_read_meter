from copy import deepcopy
import cv2
import numpy as np
import debug_tools as D

from . import darknet_base as darknet


class DarknetDet(object):
    def __init__(self,
                 config_file,
                 names_file,
                 weights,
                 thr=0.1,
                 batch_size=1,
                 class_filter=None):
        self.network, self.class_names, self.class_colors = darknet.load_network(
            config_file, names_file, weights, batch_size=batch_size)
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        self.thr = thr
        if class_filter is None:
            self.class_filter = set([])
        else:
            if isinstance(class_filter, str):
                self.class_filter = set(class_filter.strip().split(","))
            else:
                self.class_filter = set(class_filter)

    def _img_precess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_hw = img.shape[:2]

        h_resize_rate = self.height / img.shape[0]
        w_resize_rate = self.width / img.shape[1]

        resize_rate = min(h_resize_rate, w_resize_rate)
        img = cv2.resize(img, dsize=None, fx=resize_rate, fy=resize_rate)

        w_reminder = int(self.width - img.shape[1])
        h_reminder = int(self.height - img.shape[0])

        if w_reminder > 0:
            img = cv2.copyMakeBorder(img,
                                     0,
                                     0,
                                     0,
                                     w_reminder,
                                     cv2.BORDER_CONSTANT,
                                     value=[255, 255, 255])
        if h_reminder > 0:
            img = cv2.copyMakeBorder(img,
                                     0,
                                     h_reminder,
                                     0,
                                     0,
                                     cv2.BORDER_CONSTANT,
                                     value=[255, 255, 255])
        return img, resize_rate

    def result_process(self, det_result, resize_rate):
        final_result = []
        for obj in det_result:
            if obj[0] in self.class_filter:
                cx, cy,w,h = [x/resize_rate for x in obj[-1]]
                x1=int(cx-w/2)
                y1=int(cy-h/2)
                x2=int(cx+w/2)
                y2=int(cy+h/2)
                final_result.append((obj[0], x1, y1, x2, y2))
        return final_result

    def __call__(self, image):
        img, resize_rate = self._img_precess(image)
        _darknet_image = darknet.make_image(self.width, self.height, 3)
        darknet.copy_image_from_bytes(_darknet_image, img.tobytes())
        detections = darknet.detect_image(self.network,
                                          self.class_names,
                                          _darknet_image,
                                          thresh=self.thr)
        darknet.free_image(_darknet_image)
        final_result=self.result_process(detections, resize_rate)
        return final_result,detections

    def debug(self, img):
        result,r = self(img)
        show_result=[]
        for rr in result:
            pos=((rr[1]+rr[3])//2,(rr[2]+rr[4])//2,rr[3]-rr[1],rr[4]-rr[2])
            show_result.append((rr[0],"1",pos))
        image = darknet.draw_boxes(show_result, img, self.class_colors)

        return result, image


if __name__ == "__main__":
    img = cv2.imread("/home/chiebotgpuhq/Share/win_share/22222/2.jpg")
    det_model = DarknetDet(
        "/home/chiebotgpuhq/MyCode/python/meter_auto_read/model_weight/meter_det/indoor.cfg",
        "/home/chiebotgpuhq/MyCode/python/meter_auto_read/model_weight/meter_det/indoor.names",
        "/home/chiebotgpuhq/MyCode/python/meter_auto_read/model_weight/meter_det/indoor.weights",
        class_filter={
            "meter_square",
        })
    r,image = det_model.debug(img)
    print(r)
    cv2.imshow(image)
    cv2.waitKey()