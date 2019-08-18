paddle整体架构包含5个模块：最开始为组网模块，使用PythonAPI来编写深度学习模型；用户组好模型之后，交给模型表达与优化模块，它会将Python组好的模型统一转化为一个SSA的图，然后基于统一的中间表达库对图进行各种优化；优化之后的模型会交给训练模块，它会将优化后的模型进行解释执行；训练后的模型可以进一步优化，为预测和部署使用；优化好的模型就可以部署在服务器端和移动端。

##### 基本概念

组网是一层层嵌套的概念，底层包括了了`layers`，它表示一个独立的计算逻辑，通常包含一个或多个operator；`layer`的输入和输出是`Variable`，通常表示一个张量，也可以是其他类型，类似编程语言中的变量；`layer`和`variable`组合起来形成program，是一个相对完整的模型的执行逻辑，从用户角度而言，它是顺序和完整执行的；program通过executor执行，executor接受一个program，将program中的所有layer进行解释执行，同时它也可以接受feed提供的数据输入，或通过`fetch`来获取执行的中间结果。下面是一个简单的例子：

```python
from paddle import fluid
x = fluid.layers.fill_constant(shape=[1], dtype='int64', value=5)
y = fluid.layers.fill_constant(shape=[1], dtype='int64', value=1)
z = x + y # 后台建立一个program，将组建的layer和variable写入
exe = fluid.Executor(fluid.CPUPlace()) # 创建一个在CPU上执行的executor
exe.run(fluid.default_main_program(), fetch_list=[z]) # 调用run方法，将后台program传人executor
```

使用判断的例子：

```python
from paddle import fluid
a = fluid.layers.fill_constant(shape=[2, 1], dtype='int64', value=5)
b = fluid.layers.fill_constant(shape=[2, 1], dtype='int64', value=6)
ifcond = fluid.layers.less_than(x=a, y=b)
ie = fluid.layers.IfElse(ifcond)
with ie.true_block():
    c = ie.input(a)
    c += 1
    ie.output(c)
exe.run(fluid.default_main_program(), fetch_list=[c])
```

使用循环的例子：

