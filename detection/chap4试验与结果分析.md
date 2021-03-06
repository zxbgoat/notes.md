#### 准备数据集

当前目标检测领域使用的主流数据集有2个，一个是Pascal VOC，另一个是Microsoft COCO。其中VOC是一个相对较小的数据集，其最新的VOC2012训练集共有20个类别11530张图像，包含了27450个标注的ROI；COCO相比则更大，最新的COCO2017训练集包含有80个类别共123287张图像。但是回归到本文的问题，并不能直接使用这些数据集，因为每张图像会包含多个边框，如图4-1所示就是COCO数据集中的一张图像，因此在元学习中设计$N$路$K$次训练时，就无法很好地控制输入给网络的训练目标的数目。

为此我们统计了一张COCO数据集中一张图像包含的目标个数，得到了图4-2的结果。从中可以看出只有一个目标的图像是最多的，有将近1万张，因此我们将所有只包含一个目标的图像提取出来，进一步统计在这些图像中每个目标都有多少张训练样本，其结果如图4-3所示。我们将所有样本数超过25的类别的所有样本挑选出来，获得了一个COCO数据集的子集，我们将其命名为MiniCoco。后面所有的训练都以此数据集为基础。



#### 主干网络设计

现在主流的检测器大都可以分为主干网络和检测前端两个部分。主干网络主要是将输入图像提取出特征映射，以便于后面进行进一步的操作；检测前端则在特征映射的每个位置上以一定的锚框为基础进行类别预测和边框回归。主干网络一般都使用在ImageNet分类数据集上训练好的网络的基础上进行调优。

为了设计一个既体量足够轻巧、又性能足够强大、同时检测出的特征映射还要便于后续处理的主干网络，我们在论文MAML所使用网络的基础上，逐步增加网络的大小以提高其学习能力、同时使用学习进行测试其是否太过复杂，最终得到的主干网络为：

从上面的曲线可以看到，当网络到达第。



#### 检测前端设计

根据。



#### Anchor的选择

为了更加方便地作出预测，现在主流的检测器都会使用一些预先设定好的锚框，让检测器根据这些锚框中的内容来实现类别和边框的预测。一般有两种设置锚框的方式。一种是类似Faster RCNN，按照不同尺度和不同长宽比做排列组合来进行设置，比如选择128、256、512和三种大小，以及1:1、1:2、2:1三种长宽比，这样总共就会有9种锚框。另一种就是YOLO中使用的对检测数据集中的边框聚类来获得锚框，聚类的距离定义为锚框与边框之间的IOU。

我在实验中对这两种方法都进行了尝试。在尝试第一种方法时，鉴于我们网络的输入图像大小为160，提取特征映射时的步长时32，因此选择了32、64、128三种大小和1:1、1:2、2:1三种不同比例；在使用聚类方法时，按照与比边框的IOU为距离将MiniCoco数据集中的边框聚为4类。



#### 正负样本不均衡

在检测时，检测前端会在特征映射的每一个位置上基于每个锚框做出类别预测和边框回归，这样就会有将近$256\times5\times5=6400$个结果，而其中只有少量的结果是正确的，即预测的类别正确、预测的边框与实际边框的IOU大于一定阈值的结果，才归为证样本的类别，其余都是负样本，也就是说正负样本的比例达到了$1:1000$，这非常不利于神经网络的学习。

已经有一些方法来解决这种类型的问题，一种是对比例较少的类别进行过采样或对比例较多的类别进行欠采样，比如Faster RCNN中就通过RPN网络按照1:3的正负样本比例提交给网络进行学习；还有一种是设计特殊的损失函数来调整不同比例的类别对损失的贡献度，RetinaNet中使用的Focal Loss就是这种方法的典型代表。

在试验中，我们尝试了多种方法来解决这个问题：



#### 模型评估

在机器学习的分类问题中有正确率 Precision和召回率Recall两种评价指标，分别用于指示查准率（在检测出的正样本中真实正样本的比例）和查全率（检测出的正样本占所有证样本的比例）。它们是模型性能两种不同维度的度量，而且不可能同时降低。通过调节分类器的阈值可以调节这两个指标。

在目标检测中一个常用的评估检测器性能的指标时mean Average Precision，即mAP。它定义为每个类别的Average Precision。

