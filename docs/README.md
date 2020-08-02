# Object-Detection-SCNUCS
 Research on Lightweight  Fast Object Detection of SCNUCS

## 简介

本仓库主要包含code和docs等文件夹（外加一些数据存放在data中）。其中code文件夹就是每章相关jupyter notebook代码（基于PyTorch）；docs文件夹就是markdown格式的相关内容学习，可利用[docsify](https://docsify.js.org/#/zh-cn/)将网页文档部署到GitHub Pages上，在本地以网页形式访问文档，具体使用方法如下：

先安装`docsify-cli`工具：（注：npm是node.js，要先安装好）

``` shell
npm i docsify-cli -g
```

然后将本项目clone到本地（如已clone则在根目录打开命令行窗口）：

``` shell
git clone https://github.com/ShusenTang/Dive-into-DL-PyTorch.git
cd Dive-into-DL-PyTorch
```

然后运行一个本地服务器，在命令行输入：

``` shell
docsify serve docs
```

这样就可以很方便的在`http://localhost:3000`实时访问文档网页渲染效果。

## 目录

完善中……

* [简介]()


* [1. 简介]()


* 2\. 预备知识

  * [2.1 基本术语](Basic_tutorial/基本术语.md)
  * [2.2 常见评估指标](Basic_tutorial/常见评估指标.md)
  * [2.3 git](Basic_tutorial/git.md)
  * [2.3 markdown](Basic_tutorial/markdown基本语法.md)

* 3\. 深度学习基础

  * [3.1 神经网络基础](DL_basics/神经网络基础.md)

  * [3.2 ]()
  
    



##**TODO**

+ CNN详细解析

+ 使用keras、tensorflow、pytorch分别实现MNIST手写数字识别

  + 各种分类网络结构

  1. AlexNet，VGG16…
  2. Darknet
  3. ResNet残差网络
  4. DenseNet->CSPNet跨阶段局部网络

+ 各种trick学习

  1. SSP

  2. Batch Normalization

  3. FPN

  4. RPN

  5. anchor锚框

  6. bounding box regression

  7. 多尺度检测

  8. 先验框Anchor Boxes

  9. 常用损失函数（https://www.cnblogs.com/ywheunji/p/13118232.html

  10. Mish激活函数

     ……

+ yolov3解读

  https://blog.csdn.net/leviopku/article/details/82660381

  https://zhuanlan.zhihu.com/p/143747206

  https://blog.csdn.net/litt1e/article/details/88907542

+ 一些其他参考文章

  [理解卷积神经网络的局限](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247493838&idx=1&sn=6280b66c21308c8253a6411ad58f7beb&chksm=ec1c0537db6b8c21d2edd2ad171a4f7e8e11460edb7cb46dfa3ac6851cfbda079e92e79dcae4&scene=158#rd)

  

