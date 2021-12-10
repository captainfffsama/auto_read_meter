# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 12月 09日 星期四 16:01:55 CST
@Description: 
'''
from typing import List,Tuple
import math

def find_normal_line(x0,y0,x1,y1):
    xc=(x0+x1)//2
    yc=(y0+y1)//2

    if not int(y1-y0):
        return None,xc
    k=-((x1-x0)/(y1-y0))
    b=yc-k*xc

    return k,b

def find_cross_pt(k1,b1,k2,b2):
    if k1==k2:
        return None
    if k1 is None:
        return b1,k2*b1+b2
    if k2 is None:
        return b2,k1*b2+b1

    return (b2-b1)/(k1-k2),(k1*b1-k2*b2)/(k1-k2)

def get_circle_center(pts:List[Tuple[int,int]]):
    line1=find_normal_line(*pts[0],*pts[1])
    line2=find_normal_line(*pts[1],*pts[2])
    return find_cross_pt(*line1,*line2)


def get_angle(x1,y1,x2,y2,fix_angle=False):
    """计算线段角度,x1,y1 为圆上点,x2,y2为圆形

    Args:
        fix_angle: bool
            若是用于正方形表 需要修正
    """
    angle=math.atan2((y2-y1),(x1-x2))/(math.pi/180)
    if fix_angle:
        angle=180-angle if angle>=0 else -180-angle
    return angle

def get_dist(x1,y1,x2,y2) -> float:
    return math.sqrt((x1-x2)**2+(y1-y2)**2)



if __name__=="__main__":
    print(get_angle(1,3,2,2))


