

+ Bag of freebies (BoF) 与 Bag of specials (BoS)

  为了提升准确度，可以针对训练过程进行一些优化，比如数据增强、类别不平衡、成本函数、软标注…… 这些改进不会影响推理速度，可被称为「Bag of freebies」。另外还有一些改进可称为「bag of specials」，仅需在推理时间方面做少许牺牲，就能获得优良的性能回报。这类改进包括增大感受野、使用注意力机制、集成跳过连接（skip-connection）或 FPN 等特性、使用非极大值抑制等后处理方法。本文将探讨特征提取器和颈部的设计方式以及那些好用的 BoF 和 BoS 改进策略。

  

+ 空间注意力模块（SAM）

+ 用于骨干部分的 Bag of freebies (BoF)

  用于 YOLOv4 骨干部分的 BoF 特征包括：

  - CutMix 和 Mosaic 数据增强
  - DropBlock 正则化
  - 类别标签平滑化

+ 

目标检测器由用于特征提取的骨干部分（backbone）和用于目标检测的头部（head，下图最右边的模块）构成

