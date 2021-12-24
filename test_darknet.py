import cv2
from core.meter_det.darknet.d_yolo import DarknetDet


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
    cv2.imshow("frame",image)
    cv2.waitKey()