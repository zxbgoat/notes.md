```python
# load packages
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
%matplotlib inline
```

##### 1.问题描述

用YOLO模型构建一个车辆检测系统。现已收集很多照相机拍摄的路面场景，并将车框起标记了出来，如下图：

<img src="figures/box_label.png" />

若需要识别80类物体，可以用数字1到80，或80维one-hot向量表示类别标记$c$。

##### 2.YOLO

YOLO算法中在图像上"only look once"表示仅需一次经过网络的前向传播来做预测。在非最大抑制后，输出识别的物体以及对应的边界框。

**2.1模型细节**：模型输入形状为$(m,608,608,3)$的一批图像，输出为边界框以及识别的类，每个边界框由$(p_c,b_x,b_y,b_h,b_w,c)$这6个数字表示，若将$c$扩展为80维向量，则边界框用85个数字表示。这里会使用5个锚边框，因此可以认为YOLO结构如下：
$$
\text{IMAGE}(m,608,608,3) \to \text{DEEP CNN} \to \text{ENCODING}(m,19,19,5,85)
$$
更详细的细节见下图：

<img src="figures/architecture.png" />

若物体的中点落入某个网格单元，则此单元负责监测物体。因为使用了5个锚边框，因此$19\times19$每个单元编码5个边框的信息。锚边框仅由其宽度和高度定义，为简洁性，这里会将$(19,19,5,85)$的最后两维铺平，因此CNN的输出形状为$(19,19,425)$。

<img src="figures/flatten.png" />

现在，对每个单元的每个边框盒子会计算下面的元素并提取为边框含有某一物体的概率：

<img src="figures/probability_extraction.png" />

下面是一种可视化YOLO在一幅图像做预测的方法：

- 对每个网格单元，找到最大的概率得分的类；
- 基于此单元最可能含有的类着相应的颜色。

这就得到下面这样的图片：

<img src="figures/proba_map.png" />

另一种可视化YOLO输出的方法是描绘出其输出的边界框，这样会得到下面的图片：

<img src="figures/anchor_map.png" />

上面的图片仅描绘出了模型赋予高概率的边框，单依然太多了。需要过滤算法的输出到小的多的检测到的物体，为此需要使用非最大抑制。为此，执行下面的步骤：

- 去除得分低的边框；
- 当多个边框重叠时，仅选择一个。

**2.2用类分数阈值过滤**：算法需要去除任何类分数小于阈值的框盒。模型总计会给出$19\times19\times5$的框盒，每个框盒由85个数字描述，可以将这些张量重新安排成这些变量：

- `box_confidence`：形为$(19\times19,5,1)$的包含每个单元5个锚框盒的$p_c$（存在物体的信念概率）；
- `boxes`：包含每个单元锚框盒$(b_x,b_y,b_h,b_w)$的$(19\times19,5,4)$张量；
- `box_class_probs`：包含检测概率$(c_1,c_2,\dots,c_{80})$的$(19\times19,5,80)$的张量。

要实现

```python
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    """
    
    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs
    
    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    filtering_mask = box_class_scores >= threshold
    
    # Step 4: Apply the mask to scores, boxes and classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes
```

**2.3非最大抑制**：即便在用阈值过滤后，依然会有很多重合的框盒。第二种选择正确框盒的过滤方法是非最大抑制(NMS)，其效果如下图所示：

<img src="figures/non-max-suppression.png" />

非最大抑制一个很重要的函数是“交上并”，或IoU：

<img src="figures/iou.png" />

这里，使用两角坐标定义框盒（左上和右下）：$(x_1,y_1,x_2,y_2)$；这样，框盒的面积就是$(y_2-y_1)\times(x_2-x_2)$；另外还需要找到两个框盒交的坐标$(x_{i1},y_{i1},x_{i2},y_{i2})$，其中：

- $x_{i1} = \max(x_{a1}, x_{b1}),\ y_{i1}=\max(y_{a1}, y_{b1})$；
- $x_{i2} = \min(x_{a2}, x_{b2}),\ y_{i2}=\min(y_{a2}, y_{b2})$。

另外，要确保并区域的高和宽都是整的，否则并区域为0。使用$\max(\text{height}, 0)$和$\max(\text{width}, 0)$。

```python
def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2-xi1) * max(0, yi2-yi1)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    union_area = box1_area + box2_area - inter_area
    
    # compute the IoU
    iou = inter_area / union_area
    
    return iou
```

现在已经可以实现非最大抑制了。其关键步骤是：

1. 选择有最高分的框盒；
2. 计算所有与它相交的框盒，当重叠部分超过一定阈值时，将相交的框盒删除；
3. 回到步骤1迭代直到没有比当前所选更分数更低的框盒。

 TF中有内置的`tf.image.non_max_supression()`，因此实际使用时并无需自己实现。

```python
def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    """
    
    # tensor to be used in tf.image.non_max_suppression()
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    # initialize variable max_boxes_tensor
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression(boxes, scores, 10, 0.5)
    
    # Use K.gather() to select only nms_indices from scores, boxes and classes
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes
```

**2.4打包过滤**：现在实现将深度CNN的输出用刚才实现过滤的函数。

```python
def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    
    # Retrieve outputs of the YOLO model (≈1 line)
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions 
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
    
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
    
    return scores, boxes, classes
```

YOLO总结：

- 输入图片是(608, 608, 3)；
- 输入图片经过CNN，产生维度为(19, 19, 5, 85)的输出；
- 在过滤最后两维后，输出形状是(19, 19, 425)：
  - 每个图像上的$19\times19$网格会给出425个数字；
  - 每个网格有5个锚框盒，每个框盒对应85个数字；
  - 每个框盒的85个数字中，前5个是$(p_c,p_x,p_h,p_w,p_w)$，80是需要探测的类的数目；
- 经过下面步骤后，仅选择少量框盒：
  - 阈值过滤：将分数小于一个阈值的框盒丢弃；
  - 非最大抑制：计算iou以避免选择重叠的框盒；
- 这样得到最终的YOLO输出。



##### 3.在预训练的模型上测试图像

这里会使用预训练的模型测试汽车检测数据集，先创建会话开始图：

```python
sess = K.get_session()
```

**3.1定义类、锚框盒盒图像形状**：80个类盒5个锚框盒的信息分别已收集到文件`coco_classes.txt`和`yolo_anchors.txt`中，加载这些文件，另外输入$720\times1090$图像：

```
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)    
```

**3.2加载预训练模型**：并查看其描述，此模型会将(m,608,608,3)形状的输入预处理为(m,19,19,5,85)张量。

```python
yolo_model = load_model("model_data/yolo.h5")
yolo_model.summary()
```

**3.3将模型输出转化为可用的边界框张量**：

```python
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
```

**3.4过滤框盒**：

```python
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
```

**3.5在图像上运行图**：

现在已经创建了一个图，可以总结为：

1. `yolo_model.input`输入给`yolo_model`，模型计算输出`yolo_model.output`；
2. `yolo_model.output`经由`yolo_head`处理，得到`yolo_outputs`；
3. `yolo_outputs`经由过滤函数`yolo_eval`，输出预测：scores, boxes, classes。

现在实现运行这个图的`predict()`函数。注意使用批正规化的模型（比如YOLO），需要在`feed_dict`中传递占位符：`K.learning_phase(): 0`

```python
def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
    
    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    ### START CODE HERE ### (≈ 1 line)
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input:image_data, K.learning_phase():0})
    ### END CODE HERE ###

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes
```

需要记住的是：

- YOLO是很先进的目标检测算法；
- It runs an input image through a CNN which outputs a 19x19x5x85 dimensional volume.
- The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
- You filter through all the boxes using non-max suppression. Specifically:
  - Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes
  - Intersection over Union (IoU) thresholding to eliminate overlapping boxes
- Because training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset as well as lot of computation, we used previously trained model parameters in this exercise. If you wish, you can also try fine-tuning the YOLO model with your own dataset, though this would be a fairly non-trivial exercise.
