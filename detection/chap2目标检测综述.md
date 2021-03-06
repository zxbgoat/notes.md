一个典型的基于深度学习的检测器如，一般由主干网络和检测前端组成。主干网络一般是一个深度神经网络，用于提取输入图像的特征，获得一个关于图像的$S\times S$大小的特征图，特征图的每个位置都是关于原来那一部分图像内容的高维特征向量；而检测前端则是一个较小的神经网络，它会在特征映射的每个位置上，使用预先定义的锚框（Anchor）框出一部分的图像内容，判断这些图像内容是否包含有目标、如果包含目标的话目标的种类和边框又是什么；在前一步中，同一个位置会预测目标类别和边框，这些预测一般都是冗余的，所以还要用非最大抑制这样的方法来筛选出最终的结果。由此可见目标检测是一个多阶段、多组成的过程，这其中每个阶段、每个组成的不同选择和改善，就衍生出了现在形形色色的检测器。下面我们就从主干网络设计、是否使用Proposal、是否使用Anchor等多个方面来探讨和总结现代检测器。



##### 是否从头训练网络

当前大多数的检测器都是基于ImageNet上预训练的主干网络、再使用目标检测的数据集调优训练而得；这样做一方面是因为已有很多公开发表的最优性能模型因此十分方便重用，另一方面所需实例层次的标注训练数据相比分类任务也少得多，因此取得了不错的效果。但也有很多学者对这种做法提出了质疑。

文献[1]指出，在目标检测中使用预训练网络有3个严重的缺陷：一是结构设计空间受限，二是会有学习偏差，三是领域不匹配；为解决这些问题，作者设计了一个专注于目标检测的神经网络DSOD，并使用MICROSOFT COCO和PASCAL VOC数据集从头训练，取得了超越前面使用预训练网络的效果；此外作者还总结了一些如何训练好一个从头开始目标检测器的经验：一是不使用提议，因为只有无提议的检测器才能在没有预训练模型的情况下快速收敛；二是由逐层密集连接提供的深度监督至关重要；三是使用3个$3\times3$后跟$2\times2$最大池化这样的简单茎干块；四是使用密集预测结构。

在这个研究的基础上，文献[2]进一步指出使用分类网络作为检测的主干还有两个严重的问题：一是像FPN这样最新的检测器与ImageNet分类网络相比包含了额外的阶段来检测多种大小的目标，二是由于很大的下采样因子传统的主干网络会产生更高的感受野，这就无法定位大目标或者识别小目标；为此作者设计了DetNet，在主干网络引入了更多的阶段，使得网络在保持高分辨率的同时保持了较大的感受野，因此在定位大目标边界和找到小物体方面更加强大。

文献[3]提出分析了前面从头训练网络的方法，指出当前被忽视的一个点是批正规化。的ScratchDet。

而文献[4]则更进一步只要使用适于优化方法的正规化和训练充分长的时间，用随机初始化的标准模型在也能COCO数据集上获得了与ImageNet预训练相匹配的结果，而且这种情况即便是在仅使用10%的数据集、使用更深或更宽的数据集、对多任务和标准也成立。通过试验，作者发现：ImageNet预训练只是加速了收敛，尤其是在训练早期，但从随机初始化训练在训练一段时间后能够赶上；ImageNet并没有自动提供更好的正则化；ImageNeg预训练在目标任务/量度对空间定位预测更加敏感时并无益处。



##### 网络架构搜索



#####  解决多尺度问题



##### 是否使用Proposal

依据是否在检测过程中提出一些最有可能包含目标的互选区域，又可以将检测器分为单阶段检测器和两阶段检测器。单阶段检测器也即没有proposal的检测器，将检测问题看成由图片直接到目标边界框的回归问题。



##### 如何处理样本不均衡问题

在训练检测器时，



##### 是否使用Anchor

前期的检测器大都使用Anchor。

文献[5]将

文献[7]则将一个目标表示为单个边框中心点，其



[1] DSOD: Learning Deeply Supervised Object Detectors from Scratch.