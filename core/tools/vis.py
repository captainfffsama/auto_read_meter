from typing import List
from copy import deepcopy
import cv2
from core.tools.info_container import ImageInfoContainer

COLOR_MAP = {
    "general": (100, 100, 100),
    "2": (200, 200, 200),
    "3": (150, 150, 150),
    "4": (10, 150, 10),
    "min_scale": (0, 0, 255),
    "max_scale": (0, 255, 0),
    "pointer": (255, 0, 0),
    "small_label": (255, 255, 0),
    "big_label": (0, 255, 255),
}


def draw_frame(img, img_info: ImageInfoContainer):
    font = cv2.FONT_HERSHEY_PLAIN
    frame = deepcopy(img)
    h, w, c = frame.shape
    for idx, info in enumerate(img_info.global_info):
        cv2.putText(frame, info, (20, 30 + (idx * 30)), font, 2,
                    COLOR_MAP["general"], 2)

    for obj_idx, meter_info in enumerate(img_info.meters_info):
        meter_tl = meter_info.meter_rect[:2]
        if meter_tl[-1] > 60:
            base_pos = meter_tl
        else:
            base_pos = meter_info.meter_rect[-2:]

        cv2.rectangle(frame, meter_tl, meter_info.meter_rect[-2:],
                      COLOR_MAP["2"], 3)
        if meter_info.num is not None:
            cv2.putText(frame, str(meter_info.num),
                        (meter_tl[0] - 30, meter_tl[1] - 30), font, 2,
                        COLOR_MAP["3"], 2)
        # TODO: draw messages
        if meter_info.circle_pt[0] < 0:
            cv2.circle(frame, meter_info.circle_pt, 3, COLOR_MAP["3"], -1)

        for k, v in meter_info.pt_result.items():
            for v_i in v:
                pos = (int(v_i[0]), int(v_i[1]))
                cv2.circle(frame, pos, 3, COLOR_MAP[k], -1)

                # cv2.putText(frame, "{}".format(k), pos, font, 3, COLOR_MAP[k],
                #             2)

        for v in meter_info.ocr_result:
            cv2.circle(frame, v[0], 3, COLOR_MAP["4"], -1)

            cv2.putText(frame, "{}".format(v[-1]), v[0], font, 3,
                        COLOR_MAP["4"], 2)
    return frame