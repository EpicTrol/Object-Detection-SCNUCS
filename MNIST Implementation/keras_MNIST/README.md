## 概述

- 手写数字识别通常作为第一个深度学习在计算机视觉方面应用的示例，Mnist数据集在这当中也被广泛采用，可用于进行训练及模型性能测试；
- 模型的输入： 32*32的手写字体图片，这些手写字体包含0~9数字，也就是相当于10个类别的图片
- 模型的输出： 分类结果，0~9之间的一个数
- 下面通过多层感知器模型以及卷积神经网络的方式进行实现

## 环境

先安装tensorflow然后安装keras

## 数据集

该数据集中每张图片由28x28个像素点构成，每个像素点用一个灰度值表示。可以将这28x28个像素展开为一个一维的行向量，作为输入，也就是有784x1的向量。

如下是数字1的一个例子,我们的目的是做出一个模型，将这784个数值输入这个模型，然后它的输出是1。
![](https://img-blog.csdnimg.cn/2020010220564516.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lqdzEyMzQ1Ng==,size_16,color_FFFFFF,t_70)

**数据集加载**：def load_data()，其中x_train`是10000x784的数组`，代表10000个样本，每个样本有784个维度，每个维度表示一个像素点的灰度值/255；`y_train`是10000x10的数组，代表着10000个样本的实际输出，这10维度数组中的元素只有1和0，并且只有1个元素是1，其他都是0。元素1所在的索引就代表了实际的数字。比如上面1的索引是5，也就代表该样本的表示的是数字5。

第一次运行时程序会自动下载数据集

## 定义模型

在加载数据后，定义模型

```python
# 定义模型
model = Sequential()
# 定义输入层，输入维度，激活函数等
units = 28*28
model.add(Dense(input_dim=units,units=units,activation='sigmoid'))
......
# 定义好模型后选择不同的损失函数优化器等等compile(xxx)
# 这里我们的损失函数选择交叉熵
model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=100,epochs=20)# 
```

先通过`model = Sequential()`构造一个模型，然后定义输入维度和输出维度，`Dense`代表全连接网络，激活函数用的是常见的`Sigmoid`，此外还有`relu`,`tanh`等等，还可以加入自己自定义的激活函数。