自动读表
# requirement
paddleocr
pytorch
opencv
yacs
libdarknet.so

# 注意事项
libdarknet.so需要自己编,编完放到`./core/meter_det/darknet/`,名称就叫`libdarknet.so`,权重放到了 <https://dsm.chiebot.com:10000/d/f/661020386273572884>

# 使用说明
cap_img.py 是仅仅用来采集图像的,使用的是室内机的模型,应该仅需要libdarknet.so和opencv就行,没有其他依赖,启动之后,`q`退出,`w`写入,具体参数见脚本

# TODO
- [x] 刻度计算方式,现在的直接计算垂线,角度误差偏大
- [x] OCR 不太行,要增加 OCR 结果的过滤策略
- [x] 调整代码结构,优化,去掉一些 dirty change

# limit
- [ ] 表计最大量程角度和最小量程的角度应该小于180(这条被用来过滤ocr结果了)
- [ ] 表计量程上没有负值
- [ ] 表计量程上大刻度之间间隔的角度应该大于1度
