自动读表
# requirement
paddleocr
pytorch
opencv
yacs

# TODO
- [x] 刻度计算方式,现在的直接计算垂线,角度误差偏大
- [x] OCR 不太行,要增加 OCR 结果的过滤策略
- [ ] 调整代码结构,优化,去掉一些 dirty change

# limit
- [ ] 表计最大量程角度和最小量程的角度应该小于180(这条被用来过滤ocr结果了)
- [ ] 表计量程上没有负值
- [ ] 表计量程上大刻度之间间隔的角度应该大于1度
