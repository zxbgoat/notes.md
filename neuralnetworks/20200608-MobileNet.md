MobileNet是一种应用于移动端或嵌入式设备的卷积神经网络，

##### 深度可分离卷积

深度可分离卷积是一种卷积分离的形式，它将标准卷积分解为一个深度卷积（depthwise convolution）和一个被称为逐点卷积的$1\times1$卷积。标准卷积层输入$D_F\times D_F\times M$大小的特征图$\mathbf F$，产生$D_F\times D_F\times N$的特征图$\mathbf G$，卷积层的参数个数为$D_K\times D_K\times M\times N$，卷积输出可以通过下式计算：
$$
\mathbf G_{k,l,n} = \sum_{i,j,m}\mathbf K_{i,j,m,n}\cdot\mathbf F_{k+i-1,l+j-1,m}
$$
其计算代价为$D_K\cdot D_K\cdot M\cdot N\cdot D_F\cdot D_F$。它会同时对特征进行滤波并组合特征形成新表达。深度可分离卷积将这两个步骤分解开来，使用深度卷积来对特征进行滤波，使用逐点卷积来对特征进行组合。深度卷积在输入的每个通道上都使用一个滤波器，其计算公式为：

$$
\hat{\mathbf G}_{k,l,m} = \sum_{i,j}\hat{\mathbf K}_{i,j,m}\cdot\mathbf F_{k+i-1,l+j-1,m}
$$
其计算代价为$D_K\cdot D_K\cdot M\cdot D_F\cdot D_F$。逐点卷积通过

<img src='figures/dwspconv.PNG' width='500px' />

深度可分离卷积就是深度卷积和逐点卷积的组合，其计算代价为$D_K\cdot D_K\cdot M\cdot D_F\cdot D_F + M\cdot N\cdot D_F\cdot D_F$，与标准卷积计算代价的比值为：
$$
\begin{align}
&\frac{D_K\cdot D_K\cdot M\cdot D_F\cdot D_F + M\cdot N\cdot D_F\cdot D_F}{D_K\cdot D_K\cdot M\cdot N\cdot D_F\cdot D_F} \\
= &\frac1N + \frac1{D_K^2}
\end{align}
$$
MobileNet使用$3\times3$的深度可分离卷积，因此计算量比标准卷积有8~9倍的减少，而只损失了很少的精度。

##### 网络结构

除了第一层是标准卷积外，MobileNet基于深度可分离卷积构建而成，其结构如下图所示。

<img src='figures/mobilenetarch.PNG' width='500px' />

除最后一层外，所有层后面都跟有BatchNorm和ReLU，下图对比展示了标准卷积和带有BN和ReLU的可分离卷积。下采样通过跨步(strided)卷积实现。将深度和逐点卷积计为不同的层，则整个模型有28层。