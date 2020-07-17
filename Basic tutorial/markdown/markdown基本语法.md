# Markdown基本语法

推荐编辑器->[**Typora**](https://www.typora.io/#windows)！安装时记得勾选创建桌面快捷键，具体一些使用小技巧配置什么的可自行探索

## 一、标题

在想要设置标题的文字前加若干个#来表示，一个#为一级标题，两个#为二级标题，以此类推，最多六个。
注：在#后面要加一个空格才能写文字
示例：

```
# 一级标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
###### 六级标题
```

效果：

# 一级标题

## 二级标题

### 三级标题

#### 四级标题

##### 五级标题

###### 六级标题

## 二、字体

有加粗、斜体、加粗斜体和删除线等
示例：

```
这是 **加粗**  
这是 *斜体*  
这是 ***加粗并斜体***   
~~这是删除线~~
`强调字符`
```

效果：
这是 **加粗**  
这是 *斜体*  
这是 ***加粗并斜体***   
~~这是删除线~~
`强调字符` 

## 三、引用与代码

在引用的文字前加>即可引用；单行代码在代码直接分别用一个反引号`包起来，代码块则分别用三个反引号(```)包起来，且单独占一行
示例：

>引用

```c++
cout<<"hello world!"<<endl;//代码
```

## 四、分割线  

三个或三个以上的-或*都可以,而且效果都是一样的
示例：

```
---
***
```

效果：

----

***

## 五、图片

```
![图片alt](图片地址 ''图片title'')
```

图片alt就是显示在图片下面的文字，相当于对图片内容的解释（不加也行）。
图片title是图片的标题，当鼠标移到图片上时显示的内容。title可加可不加

示例：

```
![blockchain](https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/
u=702257389,1274025419&fm=27&gp=0.jpg "区块链")
```

![blockchain](https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/
u=702257389,1274025419&fm=27&gp=0.jpg "区块链")

当然如果图片是本地的可以使用相对路径，具体图像的路径设置可在typora中设置

## 六、超链接

示例：

```
[百度](www.baidu.com)
```

效果：
[百度](www.baidu.com)

## 七、列表

7.1 无序列表

> 使用 *，+，- （三个都行）表示无序列表

语法：

```markdown
* 无序列表
+ 无序列表
- 无序列表
+ 无序列表项 一
	- 子无序列表 一
	- 子无序列表 二
		* 子无序列表 三
+ 无序列表项 二
+ 无序列表项 三
```

效果：

* 无序列表
+ 无序列表
- 无序列表
+ 无序列表项 一
  - 子无序列表 一
  - 子无序列表 二
    * 子无序列表 三
+ 无序列表项 二
+ 无序列表项 三
7.2 有序列表
>有序列表则使用数字接着一个英文句点

语法：
```
1. 有序列表1
2. 有序列表2
```

效果：

1. 有序列表1
2. 有序列表2

## 八、表格

可以使用`冒号`来定义对齐方式：
绘制表格示例：

```
| 左对齐    |  右对齐 | 居中 |
| :-------- | -------:| :--: |
| Computer  | 5000 元 |  1台 |
| Phone     | 1999 元 |  1部 |
```

| 左对齐   |  右对齐 | 居中 |
| :------- | ------: | :--: |
| Computer | 5000 元 | 1台  |
| Phone    | 1999 元 | 1部  |

## 九、任务列表

> 要创建任务列表，前缀列表项`[ ]`。要将任务标记为完整，请使用`[x]`

语法：

```
- [ ] 跑步
- [ ] 骑车
- [x] 吃饭
- [ ] 睡觉
```

展现效果：

  - [ ] 跑步
  - [ ] 骑车
  - [x] 吃饭
  - [ ] 睡觉

## 十、使用LaTeX表示数学公式

**插入公式**

L aTeX的数学公式有两种：行中公式和独立公式。行中公式放在文中与其它文字混编，独立公式单独成行。

行中公式可以用如下两种方法表示：

```
＼(数学公式＼)　或　$数学公式$
```

独立公式可以用如下两种方法表示：

```
＼[数学公式＼]　或　$$数学公式$$
```

例子：

```
 $$ ＼[J\alpha(x) = \sum{m=0}^\infty \frac{(-1)^m}{m! \Gamma (m + \alpha + 1)} {\left({ \frac{x}{2} }\right)}^{2m + \alpha}＼] $$
```

显示：
$$
\alpha(x) = \sum*{m=0}^\infty \frac{(-1)^m}{m! \Gamma (m + \alpha + 1)} {\left({ \frac{x}{2} }\right)}^{2m + \alpha}\ ] 
$$
**上下标**

`^`表示上标,`_`表示下标。如果上下标的内容多于一个字符，要用`{}`把这些内容括起来当成一个整体。上下标是可以嵌套的，也可以同时使用。

例子：

```
 $$ x^{y^z}=(1+{\rm e}^x)^{-2xy^w} $$
```

显示：
$$ x^{y^z}=(1+{\rm e}^x)^{-2xy^w} $$

**括号和分隔符**

`()`、`[]`和`|`表示自己，`{}`表示`{}`。当要显示大号的括号或分隔符时，要用`\left`和`\right`命令。

例子：

```python
 $$ f(x,y,z) = 3y^2z \left( 3+\frac{7x+5}{1+y^2} \right) $$
```

显示：
$$ f(x,y,z) = 3y^2z \left( 3+\frac{7x+5}{1+y^2} \right) $$

有时候要用`\left.`或`\right.`进行匹配而不显示本身。

例子：

```python
 $$ \left. \frac{{\rm d}u}{{\rm d}x} \right| _{x=0} $$
```

显示：
$$ \left. \frac{{\rm d}u}{{\rm d}x} \right| _{x=0} $$

**分数**

例子：

```python
 $$ \frac{1}{3}　或　1 \over 3 $$
```

显示：
$$ \frac{1}{3} 　或 1 \over 3 $$

**开方**

语法：

```python
 $$ \sqrt{2}　和　\sqrt[n]{3} $$
```

显示：
$$ \sqrt{2} 　和　 \sqrt[n]{3} $$

**省略号**

数学公式中常见的省略号有两种，`\ldots`表示与文本底线对齐的省略号，`\cdots`表示与文本中线对齐的省略号。

语法：

```python
 f(x1,x2,\ldots,xn) = x1^2 + x2^2 + \cdots + xn^2
```

显示：
$$ f(x*1,x*2,\ldots,x*n) = x*1^2 + x*2^2 + \cdots + x*n^2 $$

**矢量**

语法：

```python
 \vec{a} \cdot \vec{b}=0
```

显示：
$$ \vec{a} \cdot \vec{b}=0 $$

**积分**

语法：

```python
 \int_0^1 x^2 {\rm d}x
```

显示：
$$ \int_0^1 x^2 {\rm d}x $$

**极限运算**

语法：

```python
 \lim_{n \rightarrow +\infty} \frac{1}{n(n+1)}
```

显示：
$$ \lim_{n \rightarrow +\infty} \frac{1}{n(n+1)} $$

**累加、累乘运算**

语法：

```python
 \sum{i=0}^n \frac{1}{i^2}　和　\prod{i=0}^n \frac{1}{i^2}
```

显示：
$\sum*{i=0}^n \frac{1}{i^2}$ 　和　$ \prod*{i=0}^n \frac{1}{i^2}$

**如何进行公式应用**

先要在［mathjax］后添加：

```python
  <script type="text/javascript"  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
 </script>

 ＜script type="text/x-mathjax-config"＞ MathJax.Hub.Config({ TeX: {equationNumbers: { autoNumber: ["AMS"], useLabelIds: true}}, "HTML-CSS": {linebreaks: {automatic: true}}, SVG: {linebreaks: {automatic: true}} }); ＜/script＞
```

> 只要按照这个添加代码就能实现本页的效果，包的下载自己找吧，就这样,想在线就得这样

语法：

```python
 ＼begin{equation}\label{equation1}r = rF+ \beta(rM – r_F) + \epsilon＼end{equation}
```

显示：
\begin{equation}\label{equation1}r = rF+ \beta(rM – r_F) + \epsilon\end{equation}

**希腊字母**

大写字母的是其小写latex首字母大写后的形式，如（Δ：\Delta）

发音即是它们各自的latex形式

注意区分Δ（发音为delta，表示增量）与∇（发音为nabla，表示微分，不属于希腊字母，只是一个记号，用来表示微分算子）

|           小写            |    大写    |        LaTeX形式        |
| :-----------------------: | :--------: | :---------------------: |
|         $\alpha$          |     A      |        \alpha A         |
|          $\beta$          |     B      |         \beta B         |
|         $\gamma$          |  $\Gamma$  |      \gamma \Gamma      |
|         $\delta$          |  $\Delta$  |     \delta \ Delta      |
| $\epsilon$  $\varepsilon$ |     E      |    \epsilon \Epsilon    |
|          $\zeta$          |     Z      |         \zeta Z         |
|          $\eta$           |     H      |         \eta H          |
|   $\theta$ $\vartheta$    |  $\Theta$  | \theta \vartheta \Theta |
|          $\iota$          |     I      |         \iota I         |
|         $\kappa$          |     K      |        \kappa K         |
|         $\lambda$         | $\Lambda$  |     \lambda \Lambda     |
|           $\mu$           |     M      |          \mu M          |
|           $\nu$           |     N      |          \nu N          |
|           $\xi$           |   $\Xi$    |         \xi \Xi         |
|        $\omicron$         |     O      |       \omicron O        |
|           $\pi$           |   $\Pi$    |           pi            |
|     $\rho$ $\varrho$      |     P      |     \rho \varrho P      |
|         $\sigma$          |  $\Sigma$  |      \sigma \Sigma      |
|          $\tau$           |     T      |         \tau T          |
|        $\upsilon$         | $\Upsilon$ |    \upsilon \Upsilon    |
|     $\phi$  $\varphi$     |   $\Phi$   |    \phi \varphi \Phi    |
|          $\chi$           |     X      |         \chi X          |
|          $\psi$           |   $\Psi$   |        \psi \Psi        |
|         $\omega$          |  $\Omega$  |      \omega \Omega      |