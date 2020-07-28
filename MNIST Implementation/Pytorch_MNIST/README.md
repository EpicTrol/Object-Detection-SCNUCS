# 文件目录结构

大致结构如下：
```
├─Pytorch_MNIST
│  ├─checkpoint
│  │  ├─model-mnist.pth 保存的模型
│  ├─data              MNIST数据集（文件过大未上传)
│  ├─mnist.ipynb	   Jupyter Notebook版文件
│  ├─mnist_cnn.py		  py版文件	
│  ├─Digraph.gv.pdf		 CNN网络结构图
│  ├─README.md		 文件结构说明的markdown文件	
│  ├─运行结果.jpg		 py文件运行截图
```

[数据下载慢的解决方法](https://blog.csdn.net/qq_43280818/article/details/104241326)



`class CNN(nn.Module)`定义的就是我们的神经网络，`Module`类是`nn`模块里提供的一个模型构造类，是所有神经网络模块的基类，我们可以继承它来定义我们想要的模型。下面继承`Module`类构造本节开头提到的多层感知机。这里定义的`CNN`类重载了`Module`类的`__init__`函数和`forward`函数。它们分别用于创建模型参数和定义前向计算。前向计算也即正向传播。以上的`CNN`类中无须定义反向传播函数，系统将通过自动求梯度而自动生成反向传播所需的`backward`函数。

我们可以实例化`CNN`类得到模型变量`model`。下面的代码初始化`model`然后print出来查看一下模型结构