**摘要**：本文引入一种基于区域的孪生多目标跟踪网络，将其称为SiamMOT。

- 它包含一个移动模型来评估两帧之间实例的移动，从而关联检测到的实例；
- 为探究运动建模如何影响追踪能力，本文展示了两种孪生追踪器：
  - 一种对移动隐式建模；
  - 一种对移动显式建模；
- 它在MOT17、TAO-person、Caltech Roadside Pedestrians、HiEve等多个数据集上取得了良好的效果；
- 它十分高效，在720p的视频上单个GPU能达到17fps。



#### 1 简介



#### 2 相关工作



#### 3 SiamMOT

SiamMOT构建于FaterRCNN之上，添加了一个基于区域的孪生追踪器来对实例层次的移动进行建模，如图1所示。SiamMOT：

- 取两帧$\mathbf I^t,\mathbf I^{t+\delta}$，以及在时间$t$的检测实例集合$\mathbf R^t=\left\{ R_1^t,\cdots,R_i^t,\cdots \right\}$作为输入；
- 检测网络输出检测到的实例集合$\mathbf R^{t+\delta}$。

类似SORT，SiamMOT：

- 包含一个移动模型，对每个从时间$t$到$t+\delta$检测到的实例，都会通过将时间$t$的边界框$R^t_i$传播到$t+\delta$时的$\tilde R^{t+\delta}_i$来对其进行追踪；
- 并且执行一个将追踪器输出的$\tilde R_I^{t+\delta}$与在$t+\delta$时的检测$R_i^{t+\delta}$关联起来的空间匹配过程，来将检测实例从$t$连接到$t+1$。

##### 3.1 使用孪生追踪器进行移动建模

在SiamMOT中，给定时间$t$时刻的检测实例$i$，孪生追踪器在帧$\mathbf I^{t+\delta}$上，其在$\mathbf I^t$中位置周围的上下文窗口中，搜寻此实例：
$$
\left( v_i^{t+\delta}, \tilde R^{t+\delta}_i \right) = \mathcal T\left( \mathbf f_{R^i}^t, \mathbf f_{S_i}^{t+\delta};\Theta \right)
$$


其中：

- $\mathcal T$是可学习的孪生追踪器，参数为$\Theta$；
- $\mathbf f_{R^i}^t$是从帧$\mathbf I^t$的$R^t_i$区域提取的特征图；
- $\mathbf f_{S_i}^{t+\delta}$则是从帧$\mathbf I^{t+\delta}$的搜索区域$S_i^{t+\delta}$提取的特征图；
- $v_i^{t+\delta}$是实例$i$在$S_i^t+\delta$中可见的置信度。
