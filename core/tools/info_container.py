# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 12月 22日 星期三 17:22:18 CST
@Description: 用来存储信息的容器类
'''
from typing import List, Tuple,Dict
class InfoContainerMixin:
    def clear(self):
        for attr_name in vars(self):
            obj=getattr(self, attr_name)
            if hasattr(obj, "clear"):
                obj.clear()
            else:
                obj=type(obj)()

    def __str__(self):
        for attr_name in vars(self):
            obj=getattr(self, attr_name)
            return str(obj)

class MeterInfoContainer(InfoContainerMixin):
    def __init__(self):
        self.messages:List[str]=[]
        self.ocr_result:List[Tuple[Tuple[int,int],float]]=[]
        self.pt_result:Dict[str,List[Tuple[int,int,float]]]={}
        self.circle_pt:Tuple[int,int]=(-1,-1)
        self.meter_rect=(0,0,0,0)
        self.num=None


class ImageInfoContainer(InfoContainerMixin):
    def __init__(self):
        self.global_info:List[str]=[]
        self.meters_info:List[MeterInfoContainer]=[]

def pos_fix(pt1,fix_v):
    return (pt1[0]+fix_v[0],pt1[1]+fix_v[1])



