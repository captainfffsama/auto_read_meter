# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 12月 27日 星期一 17:12:36 CST
@Description: 用来测试darknet 的性能
'''

import os
from collections import defaultdict
import xml.etree.ElementTree as ET
import cv2
from core.meter_det.darknet.d_yolo import DarknetDet
from core.file_reader import DataSequence
from tqdm import tqdm

import debug_tools as D

def padding(img):
    ret = cv2.copyMakeBorder(img, 1000, 1000, 1000, 1000, cv2.BORDER_CONSTANT, value=(0,0,0))
    return ret

def main(img_path,det_model,save_dir="",save_crop=False,show=True,gen_det_xml=False):
    data_loader=DataSequence(img_path)
    for data in tqdm(data_loader()):
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
        if save_dir and save_crop:
            file_name=os.path.basename(img_path).split(".")[0]
            try:
                for idx,r_i in enumerate(r):
                    pos=[i-1000 for i in r_i[1:]]
                    img=ori_img[pos[1]:pos[3],pos[0]:pos[2],:]
                    cv2.imwrite(os.path.join(save_dir,file_name+str(idx)+".jpg"),img)
            except:
                pass
        if save_dir and gen_det_xml:
            generate_xml_label(img_path,r,save_dir)

def dump_xml(img_info, obj_info, out_path):
    '''根据图片信息和目标信息写xml到指定路径

    Args:
        img_info: [list], [img_name, W, H, C]
        obj_info: [dict], {obj_name1: [[xmin,ymin,xmax,ymax], [xmin,ymin,xmax,ymax], ...], obj_name2: ...}
    '''

    assert out_path.split('.')[-1] == 'xml'
    out_dir, xml_name = os.path.split(out_path)
    root = ET.Element('annotation')
    folder = ET.SubElement(root, 'folder')
    folder.text = out_dir
    filename = ET.SubElement(root, 'filename')
    img_ext = img_info[0].split('.')[-1]
    img_name = xml_name.replace('.xml', '.' + img_ext)
    filename.text = img_name
    path = ET.SubElement(root, 'path')
    path.text = os.path.join(out_dir, img_name)
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')
    depth = ET.SubElement(size, 'depth')
    width.text = str(img_info[1])
    height.text = str(img_info[2])
    depth.text = str(img_info[3])

    for obj_name, bbox in obj_info.items():
        for box in bbox:
            object_root = ET.SubElement(root, 'object')
            name = ET.SubElement(object_root, 'name')
            name.text = obj_name
            pose = ET.SubElement(object_root, 'pose')
            pose.text = "Unspecified"
            trunc = ET.SubElement(object_root, 'truncated')
            trunc.text = "0"
            diff = ET.SubElement(object_root, 'difficult')
            diff.text = "0"
            bndbox = ET.SubElement(object_root, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(int(box[0]))
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(int(box[1]))
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(int(box[2]))
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(int(box[3]))
    indent(root)
    tree = ET.ElementTree(root)
    tree.write(out_path)

def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level + 1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i
    return elem

def generate_xml_label(img_path,det_result,save_dir):
    img_name=os.path.basename(img_path)
    final_det_r=defaultdict(list)
    for i in det_result:
        final_det_r[i[0]].append(tuple([x-2000 for x in i[1:]]))
    h,w,c=cv2.imread(img_path).shape
    dump_xml((img_name,w,h,c),final_det_r,os.path.join(save_dir,img_name.replace(".jpg",".xml")))

if __name__ == "__main__":
    det_model = DarknetDet(
        "/home/chiebotgpuhq/MyCode/python/meter_auto_read/model_weight/meter_det/indoor.cfg",
        "/home/chiebotgpuhq/MyCode/python/meter_auto_read/model_weight/meter_det/indoor.names",
        "/home/chiebotgpuhq/MyCode/python/meter_auto_read/model_weight/meter_det/indoor.weights",
        thr=0.8,
        class_filter={
            "meter_square",
        })

    img_path="/home/chiebotgpuhq/Pictures/piccccc"
    # img_path="/data/own_dataset/indoor_meter/VA_indoor_meter/20211207"
    # img_path="/home/chiebotgpuhq/Pictures/摄像头"
    save_dir="/home/chiebotgpuhq/Pictures/saveeeee"
    main(img_path,det_model,save_dir,True,False,False)
