本文将物体表示为在其边界框中心的单个点，其他诸如物体大小、维度、3D扩展、方向、姿势等特性都可以直接从中心位置的图像特征回归得到。这样物体检测就变成一个标准的关键点评估问题，将图像输入到一个全卷积网络产生一个热力图，热力图上的峰点对应物体中心，每个峰点的图像特征预测物体边界框的高度和宽度。模型使用标准的密集监督学习训练，推理就是网络的单个前向传播，无需使用非最大抑制进行后处理。

这个方法非常通用，无需多少改动就能扩展到其他任务。通过在关键点预测附加的输出，我们提供了3D目标检测和多人姿势估计的测验。在3D边界框估计中，我们回归物体的绝对深度、3D边框维数、和物体方向。在人体姿势估计中，则将2D联合位置看成为从中心的偏移并直接在中心点位置回归得到它们。

这个方法的简洁性使其能够告诉运行，利用简单的ResNet-18和上卷积层，能够以142FPS的速度获得0.281的COCO边框AP；利用一个仔细设计的关键点检测网络DLA-34，这个方法以52的FPS获得0.374的COCO AP；当配备当前最好的观检测估计网络Hourglass-104以及多尺度测试时，这个方法以1.4的FPS获得0.451的COCO AP。

#### 初步

令$I \in R^{W \times H \times 3}$表示输入图像，我们的目标是产生一个关键点热力图$\hat Y \in [0, 1]^{\frac{W}{R}\times\frac{H}{R}\times C}$，其中$R$是输出步长，$C$是关键点类别数，在COCO中$C=80$，我们使用诸多文献默认的设置$R=4$，输出步长以因子$R$下采样输出预测。一个预测$\hat Y_{x,y,c}$对应一个检测到的关键点，而$\hat Y_{x,y,c}=0$则表示背景。我们使用了几种全卷积编码-解码网络来从一张图像中预测$\hat Y$，包括堆叠沙漏网络、上卷积残差网络、深层聚合（DLA）等。

在训练关键点预测网络时，对每一个类别为$c$的真实关键点$p\in\mathcal R^2$，我们都计算一个低分辨率等价值$\tilde p = \left\lfloor\frac{p}{R}\right\rfloor$，然后使用高斯核$Y_{xyc}=\exp\left(-\frac{\left(x-\tilde p_x\right)^2+\left(y-\tilde p_y\right)^2}{2\sigma_p^2}\right)$将所有真实关键点铺到热力图$Y\in[0,1]^{\frac{W}{R}\times\frac{H}{R}\times C}$，其中$\sigma_p$时物体大小适应的标准差。如果同意类别的两个高斯重叠，就逐元素取最大值。训练目标是一个使用focal损失的惩罚约减逐像素逻辑回归：
$$
L_k = -\frac1N
\begin{cases}
\left(1-\hat Y_{xyc}\right)^\alpha\log\left(\hat Y_{xyc}\right) & \text{if }Y_{xyc}=1\\
\left(1- Y_{xyc}\right)^\beta\left(\hat Y_{xyc}\right)^\alpha\log\left(1-\hat Y_{xyc}\right) & \text{otherwise}
\end{cases}
$$
其中$\alpha$和$\beta$是focal损失的超参，$N$是图像中关键点个数。除以$N$的正规化是为了正规化所有正focal损失实例到1。在所有实验中我们都设置$\alpha=2, \beta=4$。

为恢复由输出步长引起的离散化损失，我们为每个中心点额外预测一个局部偏移$\hat O \in \mathcal R^{\frac WR \times \frac HR \times 2}$，所有的类别$c$共享同样的偏移预测。偏移用L1损失来训练：
$$
L_{off} = \frac1N\sum_p\left\vert\hat O_p-\left(\frac pR-\tilde p\right)\right\vert
$$
这个监督仅在关键点位置$\tilde p$起作用，所有其他位置都忽略。

#### 物体为点

设$\left(x_1^{(k)}, y_1^{(k)}, x_2^{(k)}, y_2^{(k)}\right)$为类别为$c_k$的物体$k$的边界框，其中心点位于$\left( \frac{x_1^{(k)}+x_2^{(k)}}2,\frac{y_1^{(k)}+y_2^{(k)}}2 \right)$。我们使用关键点估计器$\hat Y$来预测所有中心点，另外为每个物体$k$预测物体大小$s_k=\left(x_2^{(k)}-x_1^{(k)},y_2^{(k)}-y_1^{(k)}\right)$。为限制计算负担，我们为所有物体类别使用单一大小预测$\hat S \in \mathcal R^{\frac WR \times \frac HR \times 2}$，使用在中心点的L1损失：
$$
L_{size} = \frac1N\sum_{k=1}^N\left\vert\hat S_{p_k}-s_k\right\vert
$$
我们并未正规化大小，直接使用了原始像素坐标，但通过一个常数$\lambda_{size}$来放缩损失，整个的训练目标就是：
$$
L_{det} = L_k + \lambda_{size}L_{size} + \lambda_{off}L_{off}
$$
除非特别指定，我们在所有实验中都设$\lambda_{size}=0.1$以及$\lambda_{off}=1$。我们使用单个网络来预测关键点$\hat Y$、偏移$\hat O$和大小$\hat S$，在每个位置共计预测$C+4$个输出，所有输出共享一个普遍的主干网络；对每种形态，主干网络的特征被传递通过一个$3\times3$卷积、ReLU和另一个$1\times1$卷积。

**从点到边框**：在推理时，我们首先为每个类别独立地提取热力图的峰值，检测所有值大于或等于8个直连近邻的响应并保存前100个峰值。令$\hat{\mathcal P}_c$为类别$c$的$n$个检测到中心点$\hat{\mathcal P}=\left\{\left(\hat x_i, \hat y_i\right)\right\}_{i=1}^n$的集合。每个关键点位置由一个整数坐标$(x_i,y_i)$来给定，我们使用关键点值$\hat Y_{x_iy_ic}$为检测信念的度量，并在位置产生一个边界框：$\left(\hat x_i+\delta\hat x_i-\frac{\hat w_i}2,\hat y_i+\delta\hat y_i-\frac{\hat h_i}2,\hat x_i+\delta\hat x_i+\frac{\hat w_i}2,\hat y_i+\delta\hat y_i+\frac{\hat h_i}2\right)$，其中$\left(\delta\hat x_i, \delta\hat y_i\right)=\hat O_{\hat x_i,\hat y_i}$是偏移预测，而$\left(\hat w_i, \hat h_i\right)=\hat S_{\hat x_i,\hat y_i}$则是大小预测。所有输出都直接从关键点估计中获得，无需基于IoU的非最大抑制或其他后处理。峰值关键点提取足够代替NMS的作用并可以使用高效的$3\times3$最大池化操作实现。
