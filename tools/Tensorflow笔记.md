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

   ​

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

