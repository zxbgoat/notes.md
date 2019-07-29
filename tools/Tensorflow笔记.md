##### 1. 基础

Tensorflow有以下特点：

- 使用图(Graph)来表示任务，图中的节点被称为op(operation)，接受0或多个tensor为输入，产生0或多个tensor为输出，它描述了计算的过程；
- 使用tensor表示数据，它是一个类型化多维数组，例如表示一组图像集时则维度分别是`[batch, height, width, channels]`；
- 在被称为会话(Session)的上下文中执行图，图必须在会话里启动，会话将图的op分发到设备上，同时提供执行op的方法；
- 通过变量(Variable)维护状态；
- 使用feed和fetch可对任意操作赋值或提取数据。



##### 2. 计算图

Tf程序通常分两个阶段：先是构建阶段用一个图来表示和训练神经网络，后是执行阶段反复执行图中的op。

1. 构建图的第一步是创建源op，通常不需要任何输入，如常量(Constant)；tf库中有一个默认图，op构造器可以为其增加节点。下面的程序：

   ```python
   m1 = tf.constant([[3., 3.]])
   m2 = tf.constant([[2.], [2.]])
   prod = tf.matmul(m1, m2)
   ```

   默认图此时有3个节点，两个`constant()`op和一个`matmul()`op；要真正执行计算，还需要在会话中启动这个图。

2. 启动图的第一步是创建一个`Session`对象，若无任何创建参数，会话构造器将启动默认图。

   ```python
   sess = tf.Session()
   res = sess.run(prod)
   sess.close
   ```

   当然也可以用`with`代码块完成；tf会将图转化成分布式操作以充分利用可计算资源；一般无需显式指定，若检测到GPU，会尽可能利用检测到的第一块GPU执行操作。可以用`with ...tf.device`语句快制定具体设备，如：

   ```python
   with tf.Session() as sess:
       with tf.device("/gpu:1"):	# the second GPU
           ...
   ```

   为便于交互操作，可使用`InteractiveSession`代替`Session`，使用`Tensor.eval()`和`Operation.eval()`代替`Session.run()`。

   ```python
   sess = tf.InteractiveSession()
   x = tf.Variable([1.0, 2.0])
   a = tf.constant([3.0, 3.0])
   # 使用初始化器initializer op的run()方法初始化'x'
   x.initializer.run()
   add = tf.add(x, a)
   ```

   

##### 3. Tensor

一个tensor包含一个静态类型的rank和一个shape。`Variable`维护图执行过程中的状态信息；另外，启动图后，变量必须经过初始化op来初始化：

```python
state = tf.Variable(0, name="counter")
one = tf.constamt(1)
new_val = tf.add(state, one)
update = tf.assign(state, new_val)
#首先必须增加一个“初始化” op到图中
init_op = tf.initializer_all_variables()
with tf.Session() as sess:
    #运行“初始化”op
    sess.run(init_op)
    print sess.run(state)
    for _ in range(5):
        sess.run(update)
        print sess.run(state)
```

代码中的`assign()`操作如`add()`一样是图所描绘表达式的一部分，在调用`run()`执行表达式之前并不会真正执行赋值操作。通常将一个统计模型中的参数表示为一组变量，如将神经网络的权重存储在变量中。



##### 4. 输入输出

为取回操作op的输出内容，可以在使用`Session`对象的`run()`调用执行图时，传入一些tensor，这些tensor能取回结果，可以取回多个结果：

```python
i1 = tf.constant(3.0)
i2 = tf.constant(2.0)
i3 = tf.constant(5.0)
tmd = tf.add(i2, i3)
mul = tf.mul(i1, tmd)
with tf.Session() as sess:
    res = sess.run([mul, tmd])
```

TF中提供了feed机制，可以临时代替图中任意操作中的tensor，可以对图中任何操作提交补丁，直接插入一个tensor。feed使用一个tensor值临时替换一个操作的输出结果，可以提供feed数据为`run()`调用的参数，feed只在调用它的方法内有效，方法结束，feed就会消失，最常见的是将某些特殊操作指定为"feed"操作，标记的方法是使用`tf.placeholder()`为这些操作创造占位符。

```python
i1 = tf.placeholder(tf.types.float32)
i2 = tf.placeholder(tf.types.float32)
o = tf.mul(i1, i2)
with tf.Session() as sess:
    print sess.run([output], feed_dict={i1:[7.], i2:[2.]})
```



TF提供了多种API，最底层API—tf Core—提供了完整的编程控制，推介需要对模型有很好控制者（如机器学习研究者）使用；高层次的API基于tf Core，通常易于学习和使用，像tf.estimator能有效管理数据集、评估算子、训练和推断。



##### TF Core教程

图通过**占位符(placeholder)**接受外部输入，它表示稍后提供值：

```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
```

上面三行类似一个定义基于两个输入运算的函数或lambda，使用feed_dict参数来给出具体的值到占位符中：

```python
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
```

**Variable**能添加可训练的参数到图中，以类型和初始值构造：

```python
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b
```

常量值在调用`tf.constant()`就已经初始化，其值不可改变；而变量值在调用`tf.Variable()`时并未初始化，要初始化需显式地调用下面的操作：

```Python
init = tf.global_variables_initializer()
sess.run(init)
```

`init`是TF子图初始化所有全局变量的句柄，直到调用`sess.run`，变量才真正初始化。

TF提供了很多优化器(optimizer)，最简单的便是**gradient descent**；在仅给定模型描述的情况下，TF能调用函数`tf.gradients`自动求导：

```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```



##### tf.estimator

`tf.estimator`是一个简化ML机制的高层TF库，包含：1.运行训练循环；2.循序评估循环；3.管理数据集。它还定义了很多常用模型：

```python
# Declare list of features. We only have one numeric feature.
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
# An estimator is the front end to invoke training (fitting) and evaluation
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# Invoke 1000 training steps by invoking the method and passing the training data set.
estimator.train(input_fn=input_fn, steps=1000)
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
```

`tf.estimator`支持自定义模型，同时依然能获得数据集、训练、输入等高层次的抽象。要自定义模型，需使用`tf.estimator.Estimator`；而`tf.estimator.LinearRegressor`是前者的子类。这里只要向`Estimator`提供如何评估预测、训练步骤和损失的函数：

```python
def model_fn(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # EstimatorSpec connects subgraphs built to the appropriate functionality.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.estimator.Estimator(model_fn=model_fn)
# next are the same as above
...
```



##### 识别MNIST

导入数据，打开会话：

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
```

这里`mnist`是一个存储训练、验证、测试集的轻量级类，也提供了迭代小批次数据的函数。TF基于一个高效的C++后端进行运算，到后端的连接称为会话。为提高运算效率，tf将大量计算置于python之外；此外，也让用户描述正个在python外运行的交互操作的图。因此，python代码的作用就是构建外部计算图，并确定运行计算图的哪一部分。下面构建计算图输入和目标输出类的节点：

```python
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
```

参数中的`None`表示输入的第一维，对应于批大小，可以是任意值。现在定义模型的权值`W`和偏置`b`：

```python
W = tf.Variable(tf.zeros(784, 10))
b = tf.Variable(tf.zeros([10]))
```

在`session`中使用前`Variable`必须先用此`session`初始化，这一步将指定的初始值分配到每个`Variable`中：

```python
sess.run(tf.global_variable_initializer())
```

实现模型：

```python
y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
```

注意`tf.nn.softmax_cross_entropy_with_logits`内部实现了softmax。因TF知道整个计算图，因此就能用自动推导来得到损失对应各个变量的梯度。下面训练模型：

```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

这一步实际是添加新的操作到计算图中，这些操作包括计算梯度、计算参数更新步骤、应用更新步骤到参数。返回的操作`train_step`在运行时会应用梯度下降更新参数，而训练模型就是反复运行`train_step`：

```python
for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x:batch[0],y_:batch[1]})
```

下面是评估模型：

```python
correct_prediction  = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test_labels})
```



TensorFlow中的中心数据单元是张量(tensor)，是形状为一列任意数字维数的原始值集合。张量的秩(rank)是维数的数量。

标准的导入TF的语句为：

```python
import tensorflow as tf
```

##### 计算图

可以将TF核心程序看成是离散的两个会话组成：

1. 建立计算图；
2. 运行计算图

计算图是一系列组织进一个图节点的TF操作。每个节点输入零个或多个张量并产生一个张量作为输出。有一种节点是常量(`constant`)，像其它TF的常量一样，它没有输入，输出内部存储的值。

要真正地评估节点，必须在一个会话(`session`)中运行计算图。会话将TF运行时的控制和状态都封装了起来。可以结合`Tensor`节点建立更复杂的计算，

```python
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
sess = tf.Session()
print "sess.run(node3): ", sess.run(node3)
```

图可以被参数化为接受外部的输入，利用`placeholder`，`placeholder`表示将来输入一个值的承诺。

```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
```

上面的代码有些类似函数或`lambda`，在这之中定义两个输出参数（a和b）以及在它们之上的操作，可以用`feed_dict`参数具体指定提供给`placeholder`具体值来使用多个输入评估图。

```python
print sess.run(adder_node, {a:3, b:4.5})
print sess.run(adder_node, {a:[1,3], b:[2,4]})
```

机器学习中通常需要模型接受任意输入，为使模型可训练，需要能用同样的输入修改模型以得到不同的输出。`Variable`可添加可训练的参数到图中，由类型和初始值构造（缺失则默认为0）：

```python
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
```

当调用`tf.constant`时常数(constants)就被初始化并永远不能改变；相反，变量(`variables`)在调用`tf.Variable`时并未被初始化。要初始化TF程序中所有的`variables`，需要像下面一样显式地调用特殊的操作：

```python
init = tf.global_variables_initializer()
sess.run(init)
```

需注意到`init`是TF初始所有全局`variables`子图的句柄，直到调用`sess.run`时`variables`才会初始化。因为`x`是`placeholder`，可以用多个`x`的值来同时评估模型`linear_model`：

```python
print sess.run(linear_model, {x:[1,2,3,4]})
```

要在训练数据上评估模型，需要`y placeholder`来提供想要的值和一个损失函数。损失函数衡量当前模型离提供的数据有多远。

```python
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})
```

变量(`variable`)会被初始化为提供给`tf.Variable`的值，但可以用像`tf.assign`这样的操作来改变。

```python
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})
```

##### tf.train API

TF提供了优化器(`optimizers`)逐渐改变每个变量(`variable`)来最小化损失函数。最简单的优化器(`optimizer`)是梯度下降(`gradient dwescent`)。它依据每个变量的导数值来修改相应的变量，TF可以仅依据模型描述使用`tf.gradients`来自动产生导数。为简单起见，优化器会完成这项工作：

```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)
for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
print sess.run([W, b])
```

完整的程序为：

```python
import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print "W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss)
```

计算图表示为：

<img src="figures/Fig.png" width="750px" text-align="middle" />

##### tf.contrib.learn

`tf.contrib.learn`是高层简化机器学习的TF库，包含：

- 运行训练循环
- 运行评估循环
- 管理数据集
- 管理输入

`tf.contrib.learn`定义了许多常用模型。在使用了之后线性回归程序变得十分简洁：

```python
import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Declare list of features. We only have one real-valued feature. There are many
# other types of columns that are more complicated and useful.
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use `numpy_input_fn`. We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4,
                                              num_epochs=1000)

# We can invoke 1000 training steps by invoking the `fit` method and passing the
# training data set.
estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did. In a real example, we would want
# to use a separate validation and testing data set to avoid overfitting.
print(estimator.evaluate(input_fn=input_fn))
```

`tf.contrib.learn`并未限制在预定义的模型中。假设并不是在TF中建立一个定制模型，仍然可以使用`tf.contrib.learn`	中高度抽象的数据集、输入和训练。

为定义一个用`tf.contrib.learn`工作的定制模型，需要使用`tf.contrib.learn.Estimator`。`tf.contrib.learn.LinearRegressor`实际上是`tf.contrib.learn.Estimator`的一个子类。这里仅提供`Estimator`一个`model_fn`函数告诉`tf.contrib.learn`它如何评估预测、训练步骤和损失：

```python
import numpy as np
import tensorflow as tf
# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
# define our data set
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
print estimator.evaluate(input_fn=input_fn, steps=10)
```



下面两行代码会自动下载并读入mnist数据，

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```

TF依赖于高效的C++后端来进行计算。到后端的连接被称为会话(`session`)，TF程序通常的用法是先创建图然后在会话中投放。

这里使用简便的`InteractiveSession`类，它使得TF对用户如何构建代码更具弹性。它允许交错的建立计算图并用之运行图的操作。这在交互的上下文比如IPython中特别方便。

```python
import tensorflow as tf
sess = tf.InteractiveSession()
```

#### 建立Softmax回归模型

这部分会建立mnist数据的一个线形层的softmax回归。

##### 占位符(`Placeholders`)

```python
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
```

输入图像`x`由二维张量的浮点数组成，形状(`shape`)是`[None, 784]`，其中的`None`表示第一维，对应于批(`batch`)大小，可以是任意值。`placeholder`的`shape`	参数是可选的。

##### 变量(`Variables`)

变量是存在于TF计算图中的值，能被计算使用和修改，在机器学习应用中，通常使用模型参数为`Variables`：

```python
W = tf.Variable(tf.zeros[784, 10])
b = tf.Variables(tf.zeros([10]))
```

在使用之前，`Variable`必须先在会话(`session`)中初始化（这里初始化为0）。

```python
sess.run(tf.global_variables_initializer())
```

##### 预测的类与损失函数

损失(loss)指示了在一个样本上模型预测的坏的程度，在训练时尝试最小化之。这里的损失函数使用目标和应用到模型预测上的softmax激活函数之间的交叉熵(cross-entropy)。

```python
y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels-y_, logits=y))
```

注意`tf.nn.softmax_cross_entropy_with_logits`内部在模型非正规化预测上应用softmax并在所有类上求和；而`tf.reduce_mean`在这些和上求平均。

##### 训练模型

因TF知道整个计算图，它可以使用自动推导来找到损失每个变量对应的梯度。TF有很多内建的优化算法，这里使用最速梯度下降，步长为0.5，来在交叉熵上下降。

```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

TF实际上做的是在计算图上增加了一个运算，这个运算包含了计算梯度、计算参数更新步、应用更新步到参数。训练模型就能通过不断地运行`train_step`来完成：

```python
for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x:batch[0], y_:batch[1]})
```

在每个训练迭代中加载100个训练样本，然后运行`train_step`操作，使用训练样本的`feed_dict`来代替占位符(`placeholder`)张量(`tensor`)`x`和`y_`。

##### 评估模型

`tf.argmax`是一个非常有用的函数，它给出张量在某个维度最高量的索引；使用`tf.equal`核对预测是否符合事实。

```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```

返回布尔指南列表，转换为浮点值后求平均：

```python
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

最后在测试数据上评估精度，最后的正确率大约为92%：

```python
print accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})
```

#### 建立多层卷积神经网络

这一部分介绍适当复杂的内容：一个nmist数据的小型的卷积神经网络。

##### 参数初始化

创造模型会用到许多的权值(`weights`)和偏移(`bias`)。初始化参数时通常用很小的噪声量来打破对称，以此避免0梯度。这里会用到`ReLU`神经元(`neurons`)，使用略微正性的初始偏移来初始化权值是好的实践方法，这样可以避免“死神经元(dead neurons)”。先创建两个便利的函数来完成这些工作：

```python
def weight_variables(shape):
    initial = tf.truncated_normal(hape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
```

##### 卷积(`convolution`)与池化(`pooling`)

TF也提供了卷积和池化操作的许多灵活性。这里的卷积使用一步长(`stride`)和0填充(`padding`)，这样输出与输入的大小相同。池化是在$2 \times 2$块上简单的最大池化，这里将这些操作抽取到函数中：

```python
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                         strides=[1,2,2,1], padding='SAME')
```

##### 第一个卷积层

第一层由卷积以及之后的最大池化组成。卷积对每个$5 \times 5$的片(`patch`)会计算32个特征，其权值张量的形状为`[5,5,1,32]`，前两个维数是片的大小，下一个是输入通道(`channel`)数，最后一个是输出通道数，每个输出通道都有一个偏移向量。

```python
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
```

要应用层，先将`x`重塑(`reshape`)为4维张量，第二和第三维对应图像的宽度和高度，最后的维数对应颜色通道的数目：

```python
x_image = tf.reshape([-1, 28, 28, 1])
```

然后用权值张量对`x_image`进行卷积，增加偏置，应用`ReLU`函数，最后最大池化。`max_pool_2x2`方法会将图像尺寸减到$14\times 14$：

```python
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
```

##### 第二个卷积层

为建立深度网络，将几个这类的网络层堆积起来。第二层会在每个$5\times 5$片上有64个特征：

```python
W_conv2 = wieght_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

```

##### 紧密连接层

现在图像尺寸被缩减到了$7\times 7$，增加一个有1024个神经元的全连接层来允许处理整个图像。将池化层的张量重塑为一批向量，并用一个权值矩阵相乘，添加偏置，应用`ReLU`：

```python
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

```

##### Dropout

为减少过拟合，在读出(`readout`)层之前应用dropout。创建神经元的输出在dropout期间保持不变的概率的占位符，这使得我们可以在训练期间将dropout打开，在测试期间关闭。TF的`tf.nn.dropout`操作(`op`)除了遮蔽，还自动处理缩放神经元，因此无需额外的缩放dropout就能工作：

```python
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```

##### 读出层

最后添加一个像上面的softmax回归层：

```python
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
```

##### 训练和评估模型

训练和评估模型的代码与上面单层softmax网络很类似，但有一些不同：

- 将最速梯度下降优化器代替为更复杂的ADAM优化器；
- 会在`feed_dict`中加入额外的`keep_prob`参数来控制dropout速率；
- 在训练过程中每100次迭代增加日志。

代码如下：

```python
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```



