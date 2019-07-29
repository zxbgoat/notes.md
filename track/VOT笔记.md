##### 术语

- *Workspace*：一个包含一堆测试结果和工作脚本的目录，必须与包含toolkit的目录不同；
- *Tracker*：一个包含一个追踪算法的可执行程序或脚本；
- *Sequence*：一个包含手工标记的目标真实位置的图像序列；
- *Annotation*：目标视觉位置的手工或自动描述；
- *Trajectory*：一个描述目标在序列中运动的标记序列，工具包以txt格式保存；
- *Tracker run*：追踪器在整个序列或子集上的单词执行；
- *Trial*：若追踪器追踪时失败，在首次失败的点重新初始化；
- *Repetition*：为适当解决算法自身潜在的随机性，每个序列执行多次；
- *Experiment*：在特定情况追踪器在一个序列集合的评估；特定实验可能改变原始序列来模拟一些特定情况（如如向噪声、初始化错误）；
- *Evaluation*：在一个追踪器上执行一些列实验；
- *Region overlap*：两个区域间的重叠距离。



##### Workspace模块

这个包含用于初始化、加载和使用一个workspace的函数，workspace是一个区别于工具包目录的专用目录，存储实验专用的结果。目录包含几个特定子目录来存储序列数据、每个追踪器原始结果、报告文档和缓存数据。



#### 设置VOT工作空间

1. 从[VOT toolkit GitHub repository](https://github.com/votchallenge/vot-toolkit)下载源码；

2. 现在假定已在`vot-toolkit`有工具包，推介至少将这个目录添加到Matlab/Octave默认路径；
3. 创建一个空目录用于执行实验，不妨称其为`vot-workspace`；
4. 运行`toolkit_path`来确保所有要求的路径都在matlab路径；
5. 转到`vot-workspace`目录，运行Matlab/Octave并执行`workspace_create`命令；
6. 选择实验栈（比如`vot2018`表示VOT2018基准测试栈）；
7. 输入追踪器的唯一标识符（比如`NCC`表示`tracker/examples`提供的正规交叉关联追踪器）；
8. 选择追踪器使用的解释器（若追踪器编译为可执行文件的话可以是无）；

脚本会自动初始化工具包环境，生成工作空间配置文件`vot-workspace/configuration.m`和追踪器的配置模板（比如`vot-workspace/tracker_NCC.m`），注意此时追踪器还未配置。



#### 将追踪器集成进VOT工具包

假定电脑中已有工具包，并已正确设置了工作空间。这之后设置脚本会为追踪器配置生成一个需要手动编辑的模板文件。打开追踪器配置文件`vot-worspace/tracker_{{name}}.m`，其中`{{name}}`是追踪器的唯一标识符：

- 删除行`error('Tracker not configured!')`；
- 作为可选项，将`tracker_label = [];`设置为`tracker_label = '{{name}}';`，其中`{{name}}`是非唯一易读的追踪器名；
- 设置`tracker_command`变量为追踪器可执行文件的绝对路径；
- 若使用解释器来执行追踪器（比如Matlab或Python）确保`tracker_interpreter`变量设为正确值；
- 设置`tracker_linkpath`若追踪器需要一些不在标准库路径的库；

同样，确保`./vot-workspace/run_experiments.m`的`tracker = tracker_load('{{tracker}}')`行的`{{tracker}}`设置为追踪器的唯一标识符（比如`tracker = tracker_load('NCC')`）。这在相同工作空间中运行多个追踪器时十分重要。

##### Python追踪器

对python追踪器，工具包和追踪器之间的通信通过`tracker/examples/python`目录中的`vot.py`解决，如下：

```python
#!/usr/bin/python
import vot
import sys
import time

handle = vot.VOT("rectangle")
selection = handle.region()
# Process the first frame
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    handle.report(selection, confidence)
    time.sleep(0.01)
```

当指定`tracker_command`变量时，注意打包的脚本并非正在被执行的那个，而仅是作为python解释执行文件参数的函数。工具包提供了一个通过定位解释器执行程序和指定要运行脚本以及那之前应该被包含的目录，来为Python追踪器生成有效`tracker_command`字串的函数：

```matlab
tracker_command = generate_python_command('<TODO: tracker script>', {'<TODO: path to script>'});
```

Python的打包类覆盖在Trax库之上来与工具包通信，可在[reference implementation repository](https://github.com/votchallenge/trax)获得Trax协议的Python实现，默认初始化时会下载，检查`native/trax`是否存在。

##### 测试集成

不推介不用一个简单人物测试集成就立刻运行整个评估，工具包提供了`run_test`函数提供一个地在多种序列上测试追踪器的交互环境。

##### 集成原则

为使追踪器评估公正，列举了几条需要注意的规则：

- *随机过程*；
- *图像流*；
- *追踪器参数*；
- *资源接口*；
- *追踪信念*。