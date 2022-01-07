# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2022年 01月 07日 星期五 13:12:30 CST
@Description:
'''

from enum import Enum,Flag

# NOTE: 这里枚举不能用mixin多继承

class _StatusMeta(Enum):
    @property
    def code(self):
        """获取状态码"""
        return self.value[0] # type: ignore

    @property
    def msg(self):
        """获取状态码信息"""
        return self.value[1] # type: ignore

class OCRStatus(_StatusMeta):
    OK = (0, '成功')
    NO_DETECT = (-1, '没检测到东西')
    LESS_DETCT=(-2, '过滤后结果太少')
