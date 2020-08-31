# YOLO-V3

R-CNN系列的Two-stage算法需要先产生候选区域，再对候选区域做分类和位置坐标的预。近几年，很多研究人员相继提出一系列单阶段的检测算法，只需要一个网络即可同时产生候选区域并预测出物体的类别和位置坐标。

与R-CNN系列算法不同，YOLO-V3使用单个网络结构，**在产生候选区域的同时即可预测出物体类别和位置**，不需要分成两阶段来完成检测任务。另外，YOLO-V3算法产生的预测框数目比Faster R-CNN少很多。Faster R-CNN中每个真实框可能对应多个标签为正的候选区域，而YOLO-V3里面每个真实框只对应一个正的候选区域。这些特性使得YOLO-V3算法具有更快的速度，能到达实时响应的水平。

Joseph Redmon等人在2015年提出YOLO（You Only Look Once，YOLO）算法，通常也被称为YOLO-V1；2016年，他们对算法进行改进，又提出YOLO-V2版本；2018年发展出YOLO-V3版本。

## YOLO-V3 模型设计思想

YOLO-V3算法的基本思想可以分成两部分：

* 按一定规则在图片上**产生**一系列的**候选区域**，然后根据这些**候选区域与图片上物体真实框之间的位置关系对候选区域进行标注**。跟真实框足够接近的那些候选区域会被标注为正样本，同时将真实框的位置作为正样本的位置目标。偏离真实框较大的那些候选区域则会被标注为负样本，负样本不需要预测位置或者类别。
* 使用卷积**神经网络提取图片特征并对候选区域的位置和类别进行预测**。这样每个预测框就可以看成是一个样本，根据真实框相对它的位置和类别进行了标注而获得标签值，通过网络模型预测其位置和类别，将网络预测值和标签值进行比较，就可以建立起损失函数。

YOLO-V3算法训练过程的流程图如 **图8** 所示：

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/f2eb2b75bb5a4e518b86a257e0f931de7377dba3bba44d1e846b307036aed41a" width = "800"></center>
<center><br>图8：YOLO-V3算法训练流程图 </br></center>


* **图8** 左边是输入图片，上半部分所示的过程是使用卷积神经网络对图片提取特征，随着网络不断向前传播，特征图的尺寸越来越小，每个像素点会代表更加抽象的特征模式，直到输出特征图，其尺寸减小为原图的$\frac{1}{32}$。
* **图8** 下半部分描述了生成候选区域的过程，首先将原图划分成多个小方块，每个小方块的大小是$32 \times 32$，然后以每个小方块为中心分别生成一系列锚框，整张图片都会被锚框覆盖到。在每个锚框的基础上产生一个与之对应的预测框，根据锚框和预测框与图片上物体真实框之间的位置关系，对这些预测框进行标注。
* 将上方支路中输出的**特征图**与下方支路中产生的**预测框标签建立关联**，创建损失函数，开启端到端的训练过程。

接下来具体介绍流程中各节点的原理和代码实现。

······

## yolo-v3原理解析

### 产生候选区域

如何产生候选区域，是检测模型的核心设计方案。目前大多数基于卷积神经网络的模型所采用的方式大体如下：

* 按一定的规则在图片上**生成**一系列位置固定的**锚框**，将这些锚框看作是可能的候选区域。
* 对锚框是否包含目标物体进行预测，如果包含目标物体，还需要预测所包含物体的**类别**，以及预测框相对于锚框位置需**要调整的幅度**。

#### 生成锚框

Step1：将原始图片划分成$m\times n$个区域，如下图所示，原始图片高度$H=640$, 宽度$W=480$，如果我们选择小块区域的尺寸为$32 \times 32$，则$m$和$n$分别为：

$$
m = \frac{640}{32} = 20
$$

$$
n = \frac{480}{32} = 15
$$

如 图9 所示，将原始图像分成了20行15列小方块区域。


<center><img src="https://ai-studio-static-online.cdn.bcebos.com/2dd1cbeb53644552a8cb38f3f834dbdda5046a489465454d93cdc88d1ce65ca5" width = "400"></center>
<center><br>图9：将图片划分成多个32x32的小方块 </br></center>

Step2：YOLO-V3算法会**在每个区域的中心，生成一系列锚框**。为了展示方便，我们先在图中第十行第四列的小方块位置附近画出生成的锚框，如 **图10** 所示。

------

**注意：**

这里为了跟程序中的编号对应，最上面的行号是第0行，最左边的列号是第0列**

------


<center><img src="https://ai-studio-static-online.cdn.bcebos.com/6dd42b9138364a379b6231ac2247d3cb449d612e17be4896986bca2703acbb29" width = "400"></center>
<center><br>图10：在第10行第4列的小方块区域生成3个锚框 </br></center>


**图11** 展示在每个区域附近都生成3个锚框，很多锚框堆叠在一起可能不太容易看清楚，但过程跟上面类似，只是需要以每个区域的中心点为中心，分别生成3个锚框。


<center><img src="https://ai-studio-static-online.cdn.bcebos.com/0880c3b5ec2d40edb476f4fcbadd87aa9f37059cd24d4a1a9d37c627ce5f618a" width = "400"></center>
<center><br>图11：在每个小方块区域生成3个锚框 </br></center>



#### 生成预测框

在前面已经指出，锚框的位置都是固定好的，不可能刚好跟物体边界框重合，需要**在锚框的基础上进行位置的微调以生成预测框**。预测框相对于锚框会有不同的中心位置和大小，采用什么方式能得到预测框呢？我们先来考虑如何生成其中心位置坐标。

比如上面图中在第10行第4列的小方块区域中心生成的一个锚框，如绿色虚线框所示。以小方格的宽度为单位长度，

此小方块区域左上角的位置坐标是：
$$
c_x = 4
$$

$$
c_y = 10
$$

此锚框的区域中心坐标是：
$$
center\_x = c_x + 0.5 = 4.5
$$

$$
center\_y = c_y + 0.5 = 10.5
$$

可以通过下面的方式生成预测框的中心坐标：
$$
b_x = c_x + \sigma(t_x)
$$

$$
b_y = c_y + \sigma(t_y)
$$

其中$t_x$和$t_y$为实数，$\sigma(x)$是我们之前学过的Sigmoid函数，其定义如下：

$$
\sigma(x) = \frac{1}{1 + exp(-x)}
$$

由于Sigmoid的函数值在$0 \thicksim 1$之间，因此由上面公式计算出来的预测框的中心点总是落在第十行第四列的小区域内部。

当$t_x=t_y=0$时，$b_x = c_x + 0.5$，$b_y = c_y + 0.5$，预测框中心与锚框中心重合，都是小区域的中心。

锚框的大小是预先设定好的，在模型中可以当作是超参数，下图中画出的锚框尺寸是

$$
p_h = 350
$$

$$
p_w = 250
$$

通过下面的公式生成预测框的大小：

$$
b_h = p_h e^{t_h}
$$

$$
b_w = p_w e^{t_w}
$$

如果$t_x=t_y=0, t_h=t_w=0$，则预测框跟锚框重合。

如果给$t_x, t_y, t_h, t_w$随机赋值如下：

$$
t_x = 0.2,  t_y = 0.3, t_w = 0.1, t_h = -0.12
$$

则可以得到预测框的坐标是(154.98, 357.44, 276.29, 310.42)，如 **图12** 中蓝色框所示。

------

**说明：**
这里坐标采用$xywh$的格式。

-------


<center><img src="https://ai-studio-static-online.cdn.bcebos.com/f4b33522eb5a45f0804b94a5c66b76a0a2d13345d6de499399580a031b6ccc74" width = "400"></center>
<center><br>图12：生成预测框 </br></center>


这里我们会问：当$t_x, t_y, t_w, t_h$取值为多少的时候，预测框能够跟真实框重合？为了回答问题，只需要将上面预测框坐标中的$b_x, b_y, b_h, b_w$设置为真实框的位置，即可求解出$t$的数值。

令：
$$
\sigma(t^*_x) + c_x = gt_x
$$

$$
\sigma(t^*_y) + c_y = gt_y
$$

$$
p_w e^{t^*_w} = gt_h
$$

$$
p_h e^{t^*_h} = gt_w
$$

可以求解出 $(t^*_x, t^*_y, t^*_w, t^*_h)$

如果$t$是网络预测的输出值，将$t^*$作为目标值，以他们之间的差距作为损失函数，则可以建立起一个回归问题，通过学习网络参数，使得$t$足够接近$t^*$，从而能够求解出预测框的位置坐标和大小。

预测框可以看作是在锚框基础上的一个微调，每个锚框会有一个跟它对应的预测框，我们需要确定上面计算式中的$t_x, t_y, t_w, t_h$，从而计算出与锚框对应的预测框的位置和形状。

#### 对候选区域进行标注

每个区域可以产生3种不同形状的锚框，每个锚框都是一个可能的候选区域，对这些候选区域我们需要了解如下几件事情：

- 锚框是否包含物体，这可以看成是一个二分类问题，使用标签objectness来表示。当锚框包含了物体时，objectness=1，表示预测框属于正类；当锚框不包含物体时，设置objectness=0，表示锚框属于负类。

- 如果锚框包含了物体，那么它对应的预测框的中心位置和大小应该是多少，或者说上面计算式中的$t_x, t_y, t_w, t_h$应该是多少，使用location标签。

- 如果锚框包含了物体，那么具体类别是什么，这里使用变量label来表示其所属类别的标签。

选取任意一个锚框对它进行标注，也就是需要确定其对应的objectness, $(t_x, t_y, t_w, t_h)$和label，下面将分别讲述如何确定这三个标签的值。

##### 标注锚框是否包含物体

如 **图13** 所示，这里一共有3个目标，以最左边的人像为例，其真实框是$(40.93, 141.1, 186.06, 374.63)$。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/f21679e68d2b496698ed788a16d4ea2e5bc6f82b253a44ef9508b6a4fc9b6be4" width = "600"></center>
<center><br>图13：选出与真实框中心位于同一区域的锚框 </br></center>


真实框的中心点坐标是：

$$center\_x = 40.93 + 186.06 / 2 = 133.96$$

$$center\_y = 141.1 + 374.63 / 2 = 328.42$$

$$ i = 133.96 / 32 = 4.18625$$

$$ j = 328.42 / 32 = 10.263125$$

它落在了第10行第4列的小方块内，如**图13**所示。此小方块区域可以生成3个不同形状的锚框，其在图上的编号和大小分别是$A_1(116, 90), A_2(156, 198), A_3(373, 326)$。

用这3个不同形状的锚框跟真实框计算IoU，选出IoU最大的锚框。这里为了简化计算，只考虑锚框的形状，不考虑其跟真实框中心之间的偏移，具体计算结果如 **图14** 所示。


<center><img src="https://ai-studio-static-online.cdn.bcebos.com/3008337ea66c44068042c670db54368edc56b1e43ced4b6b811bdc95b64ca3d5" width = "400"></center>
<center><br>图14：选出与真实框与锚框的IoU </br></center>


其中跟真实框IoU最大的是锚框$A_3$，形状是$(373, 326)$，将它所对应的预测框的objectness标签设置为1，其所包括的物体类别就是真实框里面的物体所属类别。

依次可以找出其他几个真实框对应的IoU最大的锚框，然后将它们的预测框的objectness标签也都设置为1。这里一共有$20 \times 15 \times 3 = 900$个锚框，只有3个预测框会被标注为正。

由于每个真实框只对应一个objectness标签为正的预测框，如果有些预测框跟真实框之间的IoU很大，但并不是最大的那个，那么直接将其objectness标签设置为0当作负样本，可能并不妥当。为了避免这种情况，YOLO-V3算法设置了一个IoU阈值iou_threshold，当预测框的objectness不为1，但是其与某个真实框的IoU大于iou_threshold时，就将其objectness标签设置为-1，不参与损失函数的计算。

所有其他的预测框，其objectness标签均设置为0，表示负类。

对于objectness=1的预测框，需要进一步确定其位置和包含物体的具体分类标签，但是对于objectness=0或者-1的预测框，则不用管他们的位置和类别。

##### 标注预测框的位置坐标标签

当锚框objectness=1时，需要确定预测框位置相对于它微调的幅度，也就是锚框的位置标签。

在前面我们已经问过这样一个问题：当$t_x, t_y, t_w, t_h$取值为多少的时候，预测框能够跟真实框重合？其做法是将预测框坐标中的$b_x, b_y, b_h, b_w$设置为真实框的坐标，即可求解出$t$的数值。

令：
$$\sigma(t^*_x) + c_x = gt_x$$
$$\sigma(t^*_y) + c_y = gt_y$$
$$p_w e^{t^*_w} = gt_w$$
$$p_h e^{t^*_h} = gt_h$$

对于$t_x^*$和$t_y^*$，由于Sigmoid的反函数不好计算，我们直接使用$\sigma(t^*_x)$和$\sigma(t^*_y)$作为回归的目标。

$$d_x^* = \sigma(t^*_x) = gt_x - c_x$$

$$d_y^* = \sigma(t^*_y) = gt_y - c_y$$

$$t^*_w = log(\frac{gt_w}{p_w})$$

$$t^*_h = log(\frac{gt_h}{p_h})$$

如果$(t_x, t_y, t_h, t_w)$是网络预测的输出值，将$(d_x^*, d_y^*, t_w^*, t_h^*)$作为$(\sigma(t_x), \sigma(t_y), t_h, t_w)$的目标值，以它们之间的差距作为损失函数，则可以建立起一个回归问题，通过学习网络参数，使得$t$足够接近$t^*$，从而能够求解出预测框的位置。

##### 标注锚框包含物体类别的标签

对于objectness=1的锚框，需要确定其具体类别。正如上面所说，objectness标注为1的锚框，会有一个真实框跟它对应，该锚框所属物体类别，即是其所对应的真实框包含的物体类别。这里使用one-hot向量来表示类别标签label。比如一共有10个分类，而真实框里面包含的物体类别是第2类，则label为$(0,1,0,0,0,0,0,0,0,0)$

对上述步骤进行总结，标注的流程如 **图15** 所示。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/3b914be0c6274916bc7abe4922d4d0fb75be340172764f7096af5be0c2737c57" width = "700"></center>
<center><br>图15：标注流程示意图 </br></center>


通过这种方式，我们在每个小方块区域都生成了一系列的锚框作为候选区域，并且根据图片上真实物体的位置，标注出了每个候选区域对应的objectness标签、位置需要调整的幅度以及包含的物体所属的类别。位置需要调整的幅度由4个变量描述$(t_x, t_y, t_w, t_h)$，objectness标签需要用一个变量描述$obj$，描述所属类别的变量长度等于类别数C。

对于每个锚框，模型需要预测输出$(t_x, t_y, t_w, t_h, P_{obj}, P_1, P_2,... , P_C)$，其中$P_{obj}$是锚框是否包含物体的概率，$P_1, P_2,... , P_C$则是锚框包含的物体属于每个类别的概率。接下来让我们一起学习如何通过卷积神经网络输出这样的预测值。

#### 标注锚框的具体程序

上面描述了如何对预锚框进行标注，但读者可能仍然对里面的细节不太了解，下面将通过具体的程序完成这一步骤。

```python
# 标注预测框的objectness
def get_objectness_label(img, gt_boxes, gt_labels, iou_threshold = 0.7,
                         anchors = [116, 90, 156, 198, 373, 326],
                         num_classes=7, downsample=32):
    """
    img 是输入的图像数据，形状是[N, C, H, W]
    gt_boxes，真实框，维度是[N, 50, 4]，其中50是真实框数目的上限，当图片中真实框不足50个时，不足部分的坐标全为0
              真实框坐标格式是xywh，这里使用相对值
    gt_labels，真实框所属类别，维度是[N, 50]
    iou_threshold，当预测框与真实框的iou大于iou_threshold时不将其看作是负样本
    anchors，锚框可选的尺寸
    anchor_masks，通过与anchors一起确定本层级的特征图应该选用多大尺寸的锚框
    num_classes，类别数目
    downsample，特征图相对于输入网络的图片尺寸变化的比例
    """

    img_shape = img.shape
    batchsize = img_shape[0]
    num_anchors = len(anchors) // 2
    input_h = img_shape[2]
    input_w = img_shape[3]
    # 将输入图片划分成num_rows x num_cols个小方块区域，每个小方块的边长是 downsample
    # 计算一共有多少行小方块
    num_rows = input_h // downsample
    # 计算一共有多少列小方块
    num_cols = input_w // downsample

    label_objectness = np.zeros([batchsize, num_anchors, num_rows, num_cols])
    label_classification = np.zeros([batchsize, num_anchors, num_classes, num_rows, num_cols])
    label_location = np.zeros([batchsize, num_anchors, 4, num_rows, num_cols])

    scale_location = np.ones([batchsize, num_anchors, num_rows, num_cols])

    # 对batchsize进行循环，依次处理每张图片
    for n in range(batchsize):
        # 对图片上的真实框进行循环，依次找出跟真实框形状最匹配的锚框
        for n_gt in range(len(gt_boxes[n])):
            gt = gt_boxes[n][n_gt]
            gt_cls = gt_labels[n][n_gt]
            gt_center_x = gt[0]
            gt_center_y = gt[1]
            gt_width = gt[2]
            gt_height = gt[3]
            if (gt_height < 1e-3) or (gt_height < 1e-3):
                continue
            i = int(gt_center_y * num_rows)
            j = int(gt_center_x * num_cols)
            ious = []
            for ka in range(num_anchors):
                bbox1 = [0., 0., float(gt_width), float(gt_height)]
                anchor_w = anchors[ka * 2]
                anchor_h = anchors[ka * 2 + 1]
                bbox2 = [0., 0., anchor_w/float(input_w), anchor_h/float(input_h)]
                # 计算iou
                iou = box_iou_xywh(bbox1, bbox2)
                ious.append(iou)
            ious = np.array(ious)
            inds = np.argsort(ious)
            k = inds[-1]
            label_objectness[n, k, i, j] = 1
            c = gt_cls
            label_classification[n, k, c, i, j] = 1.

            # for those prediction bbox with objectness =1, set label of location
            dx_label = gt_center_x * num_cols - j
            dy_label = gt_center_y * num_rows - i
            dw_label = np.log(gt_width * input_w / anchors[k*2])
            dh_label = np.log(gt_height * input_h / anchors[k*2 + 1])
            label_location[n, k, 0, i, j] = dx_label
            label_location[n, k, 1, i, j] = dy_label
            label_location[n, k, 2, i, j] = dw_label
            label_location[n, k, 3, i, j] = dh_label
            # scale_location用来调节不同尺寸的锚框对损失函数的贡献，作为加权系数和位置损失函数相乘
            scale_location[n, k, i, j] = 2.0 - gt_width * gt_height

    # 目前根据每张图片上所有出现过的gt box，都标注出了objectness为正的预测框，剩下的预测框则默认objectness为0
    # 对于objectness为1的预测框，标出了他们所包含的物体类别，以及位置回归的目标
    return label_objectness.astype('float32'), label_location.astype('float32'), label_classification.astype('float32'), \
             scale_location.astype('float32')
```


```python
# 读取数据
reader = multithread_loader('/home/aistudio/work/insects/train', batch_size=2, mode='train')
img, gt_boxes, gt_labels, im_shape = next(reader())
# 计算出锚框对应的标签
label_objectness, label_location, label_classification, scale_location = get_objectness_label(img,
                                                                                              gt_boxes, gt_labels, 
                                                                                              iou_threshold = 0.7,
                                                                                              anchors = [116, 90, 156, 198, 373, 326],
                                                                                              num_classes=7, downsample=32)

```


```python
img.shape, gt_boxes.shape, gt_labels.shape, im_shape.shape
```


    ((2, 3, 320, 320), (2, 50, 4), (2, 50), (2, 2))


```python
label_objectness.shape, label_location.shape, label_classification.shape, scale_location.shape
```


    ((2, 3, 10, 10), (2, 3, 4, 10, 10), (2, 3, 7, 10, 10), (2, 3, 10, 10))



上面的程序实现了对锚框进行标注，对于每个真实框，选出了与它形状最匹配的锚框，将其objectness标注为1，并且将$[d_x^*, d_y^*, t_h^*, t_w^*]$作为正样本位置的标签，真实框包含的物体类别作为锚框的类别。而其余的锚框，objectness将被标注为0，无需标注出位置和类别的标签。

- 注意：这里还遗留一个小问题，前面我们说了对于与真实框IoU较大的那些锚框，需要将其objectness标注为-1，不参与损失函数的计算。我们先将这个问题放一放，等到后面建立损失函数的时候再补上。

------

### 卷积神经网络提取特征

在上一节图像分类的课程中，我们已经学习过了通过卷积神经网络提取图像特征。通过连续使用多层卷积和池化等操作，能得到语义含义更加丰富的特征图。在检测问题中，也使用卷积神经网络逐层提取图像特征，通过最终的输出特征图来表征物体位置和类别等信息。

YOLO-V3算法使用的骨干网络是Darknet53。Darknet53网络的具体结构如 **图16** 所示，在ImageNet图像分类任务上取得了很好的成绩。在检测任务中，将图中C0后面的平均池化、全连接层和Softmax去掉，保留从输入到C0部分的网络结构，作为检测模型的基础网络结构，也称为**骨干网络**。YOLO-V3模型会在骨干网络的基础上，再添加检测相关的网络模块。


<center><img src="https://ai-studio-static-online.cdn.bcebos.com/d5cb1e88d3f44259be1427a90ee454a57738ee8083ad40269f5485988526f30d" width = "400"></center>
<center><br>图16：Darknet53网络结构 </br></center>


下面的程序是Darknet53骨干网络的实现代码，这里将上图中C0、C1、C2所表示的输出数据取出，并查看它们的形状分别是，$C0 [1, 1024, 20, 20]$，$C1 [1, 512, 40, 40]$，$C2 [1, 256, 80, 80]$。

- 名词解释：特征图的步幅(stride)

在提取特征的过程中通常会使用步幅大于1的卷积或者池化，导致后面的特征图尺寸越来越小，特征图的步幅等于输入图片尺寸除以特征图尺寸。例如C0的尺寸是$20\times20$，原图尺寸是$640\times640$，则C0的步幅是$\frac{640}{20}=32$。同理，C1的步幅是16，C2的步幅是8。


```python
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from paddle.fluid.dygraph.nn import Conv2D, BatchNorm
from paddle.fluid.dygraph.base import to_variable

# YOLO-V3骨干网络结构Darknet53的实现代码

class ConvBNLayer(fluid.dygraph.Layer):
    """
    卷积 + 批归一化，BN层之后激活函数默认用leaky_relu
    """
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act="leaky",
                 is_test=True):
        super(ConvBNLayer, self).__init__()

        self.conv = Conv2D(
            num_channels=ch_in,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.02)),
            bias_attr=False,
            act=None)

        self.batch_norm = BatchNorm(
            num_channels=ch_out,
            is_test=is_test,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.02),
                regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(
                initializer=fluid.initializer.Constant(0.0),
                regularizer=L2Decay(0.)))
        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out

class DownSample(fluid.dygraph.Layer):
    """
    下采样，图片尺寸减半，具体实现方式是使用stirde=2的卷积
    """
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=2,
                 padding=1,
                 is_test=True):

        super(DownSample, self).__init__()

        self.conv_bn_layer = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            is_test=is_test)
        self.ch_out = ch_out
    def forward(self, inputs):
        out = self.conv_bn_layer(inputs)
        return out

class BasicBlock(fluid.dygraph.Layer):
    """
    基本残差块的定义，输入x经过两层卷积，然后接第二层卷积的输出和输入x相加
    """
    def __init__(self, ch_in, ch_out, is_test=True):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test
            )
        self.conv2 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out*2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test
            )
    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        out = fluid.layers.elementwise_add(x=inputs, y=conv2, act=None)
        return out

class LayerWarp(fluid.dygraph.Layer):
    """
    添加多层残差块，组成Darknet53网络的一个层级
    """
    def __init__(self, ch_in, ch_out, count, is_test=True):
        super(LayerWarp,self).__init__()

        self.basicblock0 = BasicBlock(ch_in,
            ch_out,
            is_test=is_test)
        self.res_out_list = []
        for i in range(1, count):
            res_out = self.add_sublayer("basic_block_%d" % (i), #使用add_sublayer添加子层
                BasicBlock(ch_out*2,
                    ch_out,
                    is_test=is_test))
            self.res_out_list.append(res_out)

    def forward(self,inputs):
        y = self.basicblock0(inputs)
        for basic_block_i in self.res_out_list:
            y = basic_block_i(y)
        return y

DarkNet_cfg = {53: ([1, 2, 8, 8, 4])}

class DarkNet53_conv_body(fluid.dygraph.Layer):
    def __init__(self,
                 
                 is_test=True):
        super(DarkNet53_conv_body, self).__init__()
        self.stages = DarkNet_cfg[53]
        self.stages = self.stages[0:5]

        # 第一层卷积
        self.conv0 = ConvBNLayer(
            ch_in=3,
            ch_out=32,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test)

        # 下采样，使用stride=2的卷积来实现
        self.downsample0 = DownSample(
            ch_in=32,
            ch_out=32 * 2,
            is_test=is_test)

        # 添加各个层级的实现
        self.darknet53_conv_block_list = []
        self.downsample_list = []
        for i, stage in enumerate(self.stages):
            conv_block = self.add_sublayer(
                "stage_%d" % (i),
                LayerWarp(32*(2**(i+1)),
                32*(2**i),
                stage,
                is_test=is_test))
            self.darknet53_conv_block_list.append(conv_block)
        # 两个层级之间使用DownSample将尺寸减半
        for i in range(len(self.stages) - 1):
            downsample = self.add_sublayer(
                "stage_%d_downsample" % i,
                DownSample(ch_in=32*(2**(i+1)),
                    ch_out=32*(2**(i+2)),
                    is_test=is_test))
            self.downsample_list.append(downsample)

    def forward(self,inputs):
        out = self.conv0(inputs)
        #print("conv1:",out.numpy())
        out = self.downsample0(out)
        #print("dy:",out.numpy())
        blocks = []
        for i, conv_block_i in enumerate(self.darknet53_conv_block_list): #依次将各个层级作用在输入上面
            out = conv_block_i(out)
            blocks.append(out)
            if i < len(self.stages) - 1:
                out = self.downsample_list[i](out)
        return blocks[-1:-4:-1] # 将C0, C1, C2作为返回值
```


```python
# 查看Darknet53网络输出特征图
import numpy as np
with fluid.dygraph.guard():
    backbone = DarkNet53_conv_body(is_test=False)
    x = np.random.randn(1, 3, 640, 640).astype('float32')
    x = to_variable(x)
    C0, C1, C2 = backbone(x)
    print(C0.shape, C1.shape, C2.shape)
```

    [1, 1024, 20, 20] [1, 512, 40, 40] [1, 256, 80, 80]

上面这段示例代码，指定输入数据的形状是$(1, 3, 640, 640)$，则3个层级的输出特征图的形状分别是$C0 (1, 1024, 20, 20)$，$C1 (1, 1024, 40, 40)$和$C2 (1, 1024, 80, 80)$。

------

### 根据输出特征图计算预测框位置和类别

······

------

### 计算预测框是否包含物体的概率



#### 计算预测框位置坐标

#### 计算物体属于每个类别概率

------

### 损失函数

上面从概念上将输出特征图上的像素点与预测框关联起来了，那么要对神经网络进行求解，还必须从数学上将网络输出和预测框关联起来，也就是要建立起损失函数跟网络输出之间的关系。下面讨论如何建立起YOLO-V3的损失函数。

对于每个预测框，YOLO-V3模型会建立三种类型的损失函数：

- 表征**是否包含目标物体**的损失函数，通过pred_objectness和label_objectness计算。

      loss_obj = fluid.layers.sigmoid_cross_entropy_with_logits(pred_objectness, label_objectness)

- 表征**物体位置**的损失函数，通过pred_location和label_location计算。

      pred_location_x = pred_location[:, :, 0, :, :]
      pred_location_y = pred_location[:, :, 1, :, :]
      pred_location_w = pred_location[:, :, 2, :, :]
      pred_location_h = pred_location[:, :, 3, :, :]
      loss_location_x = fluid.layers.sigmoid_cross_entropy_with_logits(pred_location_x, label_location_x)
      loss_location_y = fluid.layers.sigmoid_cross_entropy_with_logits(pred_location_y, label_location_y)
      loss_location_w = fluid.layers.abs(pred_location_w - label_location_w)
      loss_location_h = fluid.layers.abs(pred_location_h - label_location_h)
      loss_location = loss_location_x + loss_location_y + loss_location_w + loss_location_h

- 表征**物体类别**的损失函数，通过pred_classification和label_classification计算。

      loss_obj = fluid.layers.sigmoid_cross_entropy_with_logits(pred_classification, label_classification)

我们已经知道怎么计算这些预测值和标签了，但是遗留了一个小问题，就是没有标注出哪些锚框的objectness为-1。为了完成这一步，我们需要计算出所有预测框跟真实框之间的IoU，然后把那些IoU大于阈值的真实框挑选出来。实现代码如下：

······

------

### 多尺度检测

目前我们计算损失函数是在特征图P0的基础上进行的，它的步幅stride=32。特征图的尺寸比较小，像素点数目比较少，每个像素点的感受野很大，具有非常丰富的高层级语义信息，可能比较容易检测到较大的目标。为了能够检测到尺寸较小的那些目标，需要在尺寸较大的特征图上面建立预测输出。如果我们在C2或者C1这种层级的特征图上直接产生预测输出，可能面临新的问题，它们没有经过充分的特征提取，像素点包含的语义信息不够丰富，有可能难以提取到有效的特征模式。在目标检测中，解决这一问题的方式是，将高层级的特征图尺寸放大之后跟低层级的特征图进行融合，得到的新特征图既能包含丰富的语义信息，又具有较多的像素点，能够描述更加精细的结构。

具体的网络实现方式如 **图19** 所示：

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/b6d3b425644342e48bd0a50ebde90d882fd10717e0e44a53a44e98225bbb6df8" width = "800"></center>
<center><br>图19：生成多层级的输出特征图P0、P1、P2 </br></center>

YOLO-V3在每个区域的中心位置产生3个锚框，在3个层级的特征图上产生锚框的大小分别为P2 [(10×13),(16×30),(33×23)]，P1 [(30×61),(62×45),(59× 119)]，P0[(116 × 90), (156 × 198), (373 × 326]。越往后的特征图上用到的锚框尺寸也越大，能捕捉到大尺寸目标的信息；越往前的特征图上锚框尺寸越小，能捕捉到小尺寸目标的信息。

## Tricks

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

