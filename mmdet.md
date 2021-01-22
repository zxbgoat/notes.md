#### 1 上手

##### 1.1 准备数据集



#### 2 配置系统



#### 3 基准与模型库



#### 4 调优模型

将一个模型在一个数据集上进行调优需要进行两步：

1. 依照下一节增加对新数据集的支持；
2. 像这一章描述地一样调整配置。

以Cityscapes数据集为例的调优过程为例，用户需要调整配置的五个部分。

##### 4.1 继承基础配置

为减轻编写整个配置的负担，减少问题，MMDetection第2.0版支持从多个已有配置中继承配置：

- 要调优一个MaskRCNN模型，新配置需要继承`_base_/models/mask_rcnn_r50_fpn.py`来构建模型的基础结构；
- 要使用Cityscapes数据集，新配置需要继承`_base_/datasets/cityscapes_instance.py`；
- 对于训练过程这样的运行时设置，新配置需要继承`_base_/default_runtime.py`；

这些配置在`config`目录下，用户也可以选择编写全部的内容而非使用继承。

```python
_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/cityscapes_instance.py', '../_base_/default_runtime.py'
]
```

##### 4.2 修改头

之后新配置需要根据数据集的类别数目修改头。若只改变`roi_head`中的`num_classes`，则除最后的预测头的预训练模型的权值都会被重用。

```python
model = dict(
    pretrained=None,
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=8,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=8,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))))
```

##### 4.3 修改数据集

用户也需要准备数据集并编写关于数据集的配置。MMDetection 2.0已经支持VOC、Wider FACE、COCO和Cityscapes数据集。

##### 4.4 修改训练计划

调优超参与默认计划不同，通常它的学习率更小、训练epochs更少。

```python
# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[7])
total_epochs = 8  # actual epoch = 8 * 8 = 64
log_config = dict(interval=100)
```

##### 4.4 使用预训练模型

要使用预训练模型，新配置需要在`load_from`添加预训练模型的链接，用户最好在训练前将其下载下来。

```python
load_from = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth'
```



#### 5 增加数据集

##### 5.1 重组数据集为已有格式

最简单的就是将数据集转换为已有的数据集格式（COCO或PASCAL VOC），COCO的标注json文件有3个必须的关键字：

- `images`：包含一个图像列表，以及`file_name`、`height`、`width`和`id`这样的关键字；
- `annotations`：包含实例标注列表；
- `categories`：包含类别名称列表以及它们的id。

##### 5.2 重组数据集到中间格式





#### 6 定制数据流程



#### 7 增加新模块

