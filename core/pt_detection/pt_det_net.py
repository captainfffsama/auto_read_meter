# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 09月 10日 星期五 16:19:47 CST
@Description: 模型
'''
from typing import List,Dict,Tuple
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet

class NeckLayer(nn.Module):
    def __init__(self,input_channels:List[int]):
        super( NeckLayer, self).__init__()
        self.upsample_rate=2

        self.upsample_layer=nn.PixelShuffle(2)
        i_c=int(input_channels[1]//(2**2)+input_channels[0])
        self.cbr=nn.Sequential(
                nn.Conv2d(i_c,128,3,padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
        )

    def forward(self,x1,x2):
        x1=self.upsample_layer(x1)
        # x1=F.interpolate(x1,scale_factor=2,mode='bilinear',align_corners=False)
        x= torch.cat((x2,x1),dim=1)
        x=self.cbr(x)
        return x

def generate_meshgrid(torch_shape,device,dtype):
    B,C,H,W=torch_shape
    xs=torch.linspace(0,W-1,W,device=device,dtype=dtype)
    ys=torch.linspace(0,H-1,H,device=device,dtype=dtype)
    ys,xs=torch.meshgrid([ys,xs])
    grid=torch.stack([xs,ys],dim=0).repeat(B,C//2,1,1)

    return grid

class ShortCut(nn.Module):
    def __init__(self,i_c,o_c):
        super(ShortCut, self).__init__()
        self.cbr=nn.Sequential(
                nn.Conv2d(i_c,i_c,1,1),
                nn.BatchNorm2d(i_c),
                nn.LeakyReLU(0.01),
        )
        self.conv=nn.Sequential(
            nn.Conv2d(i_c,o_c,1,1),
        )


    def forward(self,x):
        x_i=self.cbr(x)
        x=self.conv(x_i)
        return x



class PointDetectNet(nn.Module):
    def __init__(self,pt_cls_num):
        super(PointDetectNet, self).__init__()
        self.features=resnet.get_resnet('resnet34')
        self.neck=NeckLayer(self.features.features_channle_list[-2:])
        self.scale_cls=ShortCut(128,pt_cls_num-1)
        self.scale_reg=ShortCut(128,(pt_cls_num-1)*2)
        self.range_cls=ShortCut(128,pt_cls_num-1)
        self.range_reg=ShortCut(128,(pt_cls_num-1)*2)
        self.mp_cls=ShortCut(128,1)
        self.mp_reg=ShortCut(128,2)

        self.center_bias=(self.downsample_rate-1)/2

    @property
    def downsample_rate(self):
        return int(self.features.downsample_rate/self.neck.upsample_rate)

    def forward(self,x:torch.Tensor):
        x3,x4=self.features(x)
        x=self.neck(x4,x3)

        x_scale_cls=self.scale_cls(x).sigmoid()
        x_scale_reg=self.scale_reg(x).tanh()

        s_grid=generate_meshgrid(x_scale_reg.shape,x_scale_reg.device,x_scale_reg.dtype).mul(self.downsample_rate)+self.center_bias
        s_coord=s_grid+x_scale_reg*self.center_bias

        x_mp_cls=self.mp_cls(x).sigmoid()
        x_mp_reg=self.mp_reg(x).tanh()

        mp_grid=generate_meshgrid(x_mp_reg.shape,x_mp_reg.device,x_mp_reg.dtype).mul(self.downsample_rate)+self.center_bias
        mp_coord=mp_grid+x_mp_reg*self.center_bias

        x_range_cls=self.range_cls(x).sigmoid()
        x_range_reg=self.range_reg(x).tanh()

        r_grid=generate_meshgrid(x_range_reg.shape,x_range_reg.device,x_range_reg.dtype).mul(self.downsample_rate)+self.center_bias
        r_coord=r_grid+x_range_reg*self.center_bias
        x_scale_cls=self.filter_cm(x_scale_cls)
        x_mp_cls=self.filter_cm(x_mp_cls)
        x_range_cls=self.filter_cm(x_range_cls)
        return x_scale_cls,s_coord,x_mp_cls,mp_coord,x_range_cls,r_coord

    def filter_cm(self,cls_map):
        cls_map_p = F.max_pool2d(cls_map, kernel_size=11, stride=1, padding=5)
        cls_map[cls_map != cls_map_p] = 0
        # cls_map[cls_map<=0.3]=0
        return cls_map


class PtDetInfer:
    def __init__(self,pt_model):
        self.model=pt_model
        self.model.eval()

    def decode_feature_map(self,c_feature_map, r_feature_map, label_list, result_dict):
        ori_cm_shape = c_feature_map.shape[1:]
        for i, label in enumerate(label_list):
            pt_idx = c_feature_map[i].argmax()
            pt_yi = pt_idx // ori_cm_shape[-1]
            pt_xi = pt_idx % ori_cm_shape[0]
            pt_score = c_feature_map[i, pt_yi, pt_xi]

            pt_x = r_feature_map[2 * i, pt_yi, pt_xi]
            pt_y = r_feature_map[2 * i + 1, pt_yi, pt_xi]
            result_dict[label].append(
                (pt_x.item(), pt_y.item(), pt_score.item()))


    def d_map(self,scm, srm, pmcm, pmrm, rcm, rrm, d_rate):
        result: List[dict] = []

        for one_img_scm, one_img_srm, one_img_pmcm, one_img_pmrm, one_img_rcm, one_img_rrm in zip(
                scm, srm, pmcm, pmrm, rcm, rrm):
            ori_cm_shape = one_img_scm.shape[1:]
            one_img_result = defaultdict(list)
            self.decode_feature_map(one_img_scm, one_img_srm,
                            ("min_scale", "max_scale"), one_img_result)
            self.decode_feature_map(one_img_pmcm, one_img_pmrm, ("pointer", ),
                            one_img_result)
            self.decode_feature_map(one_img_rcm, one_img_rrm,
                            ("small_label", "big_label"), one_img_result)

            result.append(one_img_result)

        return result

    def fix_pt_detect_result(self,result: dict, resize_rate: float):
        for k in result.keys():
            result[k] = [(x[0] / resize_rate, x[1] / resize_rate, x[2])
                        for x in result[k]]
        return result

    def __call__(self,img,resize_rate=None) -> Dict[str,List[Tuple[int,int,float]]]:
        with torch.no_grad():
            scm, srm, pmcm, pmrm, rcm, rrm = self.model(img.cuda())
            result = self.d_map(scm, srm, pmcm, pmrm, rcm, rrm,
                                self.model.downsample_rate)
            result=result[0]
            if resize_rate is not None:
                result=self.fix_pt_detect_result(result,resize_rate)

        return result