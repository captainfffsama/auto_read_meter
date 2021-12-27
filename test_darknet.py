import os
import cv2
from core.meter_det.darknet.d_yolo import DarknetDet
from core.file_reader import DataSequence
from tqdm import tqdm

import debug_tools as D

def padding(img):
    ret = cv2.copyMakeBorder(img, 2000, 2000, 2000, 2000, cv2.BORDER_CONSTANT, value=(0,0,0))
    return ret

def main(img_path,det_model,save_dir="",show=True):
    data_loader=DataSequence(img_path)
    for data in data_loader():
        ori_img,img_path=data
        img=padding(ori_img)
        r,image=det_model.debug(img)
        if show:
            if "image" == data_loader.type:
                D.show_img(image)
            else:
                cv2.imshow("frame",image)
                inputkey = cv2.waitKey(20)
                if ord("q") == inputkey:
                    data_loader.stop()
        if save_dir:
            file_name=os.path.basename(img_path).split(".")[0]
            try:
                for idx,r_i in enumerate(r):
                    pos=[i-2000 for i in r_i[1:]]
                    img=ori_img[pos[1]:pos[3],pos[0]:pos[2],:]
                    cv2.imwrite(os.path.join(save_dir,file_name+str(idx)+".jpg"),img)
            except:
                pass






if __name__ == "__main__":
    det_model = DarknetDet(
        "/home/chiebotgpuhq/MyCode/python/meter_auto_read/model_weight/meter_det/indoor.cfg",
        "/home/chiebotgpuhq/MyCode/python/meter_auto_read/model_weight/meter_det/indoor.names",
        "/home/chiebotgpuhq/MyCode/python/meter_auto_read/model_weight/meter_det/indoor.weights",
        thr=0.5,
        class_filter={
            "meter_square",
        })

    img_path="/data/own_dataset/indoor_meter/VA_indoor_meter/ex2"
    # img_path="/data/own_dataset/indoor_meter/VA_indoor_meter/20211207"
    # img_path="/home/chiebotgpuhq/Pictures/摄像头"
    save_dir="/data/own_dataset/indoor_meter/VA_indoor_meter/ex3"
    main(img_path,det_model,save_dir,False)