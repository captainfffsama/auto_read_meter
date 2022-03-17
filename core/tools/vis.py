from typing import List
from copy import deepcopy
import cv2
from core.tools.info_container import ImageInfoContainer

COLOR_MAP = {
    "general": (100, 100, 100),
    "2": (200, 200, 200),
    "3": (0, 0,100),
    "4": (10, 150, 10),
    "min_scale": (0, 0, 255),
    "max_scale": (0, 255, 0),
    "pointer": (255, 0, 0),
    "small_label": (255, 255, 0),
    "big_label": (0, 255, 255),
}


def draw_frame(img, img_info: ImageInfoContainer,detailed=True):
    font = cv2.FONT_HERSHEY_PLAIN
    frame = deepcopy(img)
    h, w, c = frame.shape
    base_size=max(1,min(h,w)//200)
    for idx, info in enumerate(img_info.global_info):
        cv2.putText(frame, info, (20, 30 + (idx * 30)), font, base_size,
                    COLOR_MAP["general"], 2)

    for obj_idx, meter_info in enumerate(img_info.meters_info):
        meter_tl = meter_info.meter_rect[:2]

        cv2.rectangle(frame, meter_tl, meter_info.meter_rect[-2:],
                      COLOR_MAP["2"], base_size)
        if meter_info.num is not None:
            cv2.putText(frame, "{:.2f}".format(meter_info.num),
                        (meter_tl[0] + 50, meter_tl[1] + 50), font, base_size,
                        COLOR_MAP["3"], 2)
        if detailed:
            if meter_tl[-1] > 60:
                base_pos = meter_tl
            else:
                base_pos = meter_info.meter_rect[-2:]
            # TODO: draw messages
            for i,msg in enumerate(meter_info.messages):
                cv2.putText(frame, str(msg),
                            (meter_tl[0] + i*10, meter_tl[1]+i*10), font, max(1,base_size//2),
                            COLOR_MAP["3"], 1)
            if meter_info.circle_pt[0] < 0:
                cv2.circle(frame, meter_info.circle_pt, 2*base_size, COLOR_MAP["3"], -1)

            for k, v in meter_info.pt_result.items():
                for v_i in v:
                    pos = (int(v_i[0]), int(v_i[1]))
                    cv2.circle(frame, pos, 2*base_size, COLOR_MAP[k], -1)

                    cv2.putText(frame, "{}".format(k), pos, font, base_size, COLOR_MAP[k],
                                2)

            for v in meter_info.ocr_result:
                cv2.circle(frame, v[0], 2*base_size, COLOR_MAP["4"], -1)

                cv2.putText(frame, "{}".format(v[-1]), v[0], font, base_size,
                            COLOR_MAP["4"], 2)
    return frame