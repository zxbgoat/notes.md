这部分完成神经风格迁移。大多数算法都是优化代价函数以获得参数值，而这里是优化代价函数获得像素值。

```python
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
```



##### 1.问题描述

神经风格迁移(NST) 会将两幅图像融合，一幅“内容”图像C和一幅“风格图像”S，创造一幅“生成”图像G，生成图像结合了C的内容和S的风格：

<img src="figures/louvre_generated.png" />



##### 2.迁移学习

将其他任务训练好的网络应用到新的任务，即使迁移学习。这里会使用在ImageNet上训练训练好的VGG-19，因此已经学习了多种低层或高层的特征。运行下面的代码加载模型：

```python
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
```

模型以python字典的形式存储，其中变量名为字典的key，tensor变量值为字典的value。要通过网络运行图像，需要旧爱那个图像输入模型，可以使用 `tf.assign`函数：

```python
model["input"].assign(image)
```

之后，若希望获得网络运行在此图像上时特定层的激活值，比如层4_2，可以将TF会话在正确的张量conv4\_2上：

```python
sess.run(model["conv4_2"])
```



##### 3.神经风格迁移

这里会用3步构建NST算法：

- 构建内容代价函数$J_{content}(C,G)$；
- 构建风格代价函数$J_{style}(S, G)$；
- 放在一起获得代价函数$J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$。

**3.1计算内容代价**：在运行的示例中，内容图像$c$是卢浮宫，先加载图像，

```python
content_image = scipy.misc.imread("images/louvre.jpg")
imshow(content_image)
```

卷积网络的浅层趋向于检测边缘或简单纹理这样的低层特征，而深层则检测复杂纹理或物体类别这样的高层特征。这里希望生成图像$G$与$C$有相同的内容，假定已经选择了某层的激活值来表示图像内容。实践中选择中间的一层会获得最优的视觉效果。现在设定$C$为预训练VGG网络的输入，并运行前向传播。另$a^{(C)}$为选定层的激活值，这会是一个$n_H \times n_W \times n_C$张量。然后对图像$G$重复同样的过程：设$G$为输入，运行前向传播。令$a^{(G)}$为对应的浅层的激活值。然后内容代价就定义为：
$$
J_{content}(C,G) =  \frac{1}{4 \times n_H \times n_W \times n_C}\sum _{ \text{all entries}} (a^{(C)} - a^{(G)})^2\tag{1}
$$
这里$n_H,n_W,n_C$分别是选定层的高、宽和通道数：

<img src="figures/NST_LOSS.png" />

实现下面3步完成内容代价的计算：

1. 获得$a^{(G)}$的维数，使用代码：`X.get_shape().as_list()`；
2. 像上图那样展开$a^{(C)}$和$a^{(G)}$；
3. 计算内容代价。

```python
def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(a_C, (m,n_H*n_W,n_C))
    a_G_unrolled = tf.reshape(a_G, (m,n_H*n_W,n_C))
    # compute the cost with tensorflow (≈1 line)
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4*n_H*n_W*n_C)
    
    return J_content
```

**3.2计算风格代价**：先加载风格图像，

```python
style_image = scipy.misc.imread("images/monet_800600.jpg")
imshow(style_image)
```

风格矩阵又被称为“格拉姆”矩阵，一系列向量$(v_1,\dots,v_n)$的格拉姆矩阵$G$是点乘矩阵，其条目为$G_{ij}=v_i^Tv_j$，即$G_{ij}$比较$v_i$和$v_j$的相似性，若相似性很高，则其点乘值$G_{ij}$就很高。早NST中，可以将展开的滤波器矩阵与其转置相乘获得风格矩阵：

<img src="figures/NST_GM.png" />

结果矩阵的维度是$(n_C,n_C)$，其中$n_C$是滤波器个数，$G_{ij}$衡量滤波器$i$和滤波器$j$激活值的相似性。很重要的一点是其对角元素$G_{ii}$也衡量了滤波器$i$的活跃度。比如$i$检测图像中的垂直纹理，则$G_{ii}$检测垂直纹理的普遍性。若$G_{ii}$很大则表示有很多垂直纹理。通过获得不同特征的普遍度($G_{ii}$)和不同特征共现的多少($G_{ij}$)，风格矩阵衡量衣服图像的风格。

```python
def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    GA = tf.matmul(A, tf.transpose(A))
    
    return GA
```

生成风格矩阵后，现在需要最小化风格图像和生成图像之间风格矩阵的差异，当前仅使用单隐层$a^{[l]}$，对应此层的风格代价定义为：
$$
J_{style}^{[l]}(S,G) = \frac{1}{4 \times {n_C}^2 \times (n_H \times n_W)^2} \sum _{i=1}^{n_C}\sum_{j=1}^{n_C}(G^{(S)}_{ij} - G^{(G)}_{ij})^2\tag{2}
$$
其中$G^{(S)}$和$G^{(G)}$分别是风格图像和生成图像的风格矩阵。现在用四步实现计算单层风格代价的函数：

1. 获得隐层$a^{(G)}$的维度；
2. 展开隐层$a^{(S)}$和$a^{(G)}$激活值到2维矩阵；
3. 计算图像$S$和$G$的风格矩阵；
4. 计算风格代价。

```python
#
```

