自定义层通常被称为接口（plugin），由应用创建并实例化，通过扩展`IPluginCreator`类和一个接口基础类来实现。其中：

- `IPluginCreator`是自定义层的创建器类，通过它能够获得接口名称、版本和接口域参数，也提供了在构建阶段和推理时反序列化时创建接口对象的方法；

- 用户必须从某个接口基类衍生得到，对应于不同类型/格式输入输出或动态形状网络的支持，这些基类拥有丰富的表达能力。下表归纳了基类，按照表达能力升序排列：

  |                       | 引入的TensorRT版本 | 混合输入/输出格式/类型 | 动态形状 |
  | --------------------- | ------------------ | ---------------------- | -------- |
  | `IPluginV2Ext`        | 5.1                | 受限                   | 否       |
  | `IPluginV2IOExt`      | 6.0.1              | 通用                   | 否       |
  | `IPluginV2DynamicExt` | 6.0.1              | 通用                   | 是       |

所有这些基类都包含版本支持，并帮助自定义层除了支持`NCHW`和单精度外，还支持其他数据格式。

>注意：
>
>- 无论使用哪种基类，都应该为接口提供一个FP32的实现，以使接口能在任何网络中适当地发挥作用；
>- 目前仍然支持6.01版本以前的`IPluginV2`和`IPluginV2Ext`，但建议迁移到`IPluginV2IOExt`和`IPluginV2DynamicExt`来使用所有新的接口功能。



#### 3.1 接口注册

通过调用`REGISTER_TENSORRT_PLUGIN(pluginCreator)`，TensorRT也提供了注册接口的功能，这个函数会静态地将`Plugin Creator`注册到`Plugin Registry`。在运行期间，使用外部（extern）函数`getPluginRegistry`就能查询`Plugin Registry`。`Plugin Registry`存储一个指向所有已注册`Plugin Creator`的指针，可以使用根据接口名和版本它来查询特定的`Plugin Creator`。TensorRT库包含了能加载到应用的接口，列表详见[GitHub:TensorRT plugins](https://github.com/NVIDIA/TensorRT/tree/master/plugin#tensorrt-plugins)。

>注意：
>
>- 要使用已注册的TensorRT接口，必须加载`libnvinfer_plugin.so`动态库，并且所有接口都要注册过。这可以通过调用`initLibNvInferPlugins(void *logger, const char* libNameSpace)()`实现。
>- 若用户拥有自己的接口库，可以包含一个类似的入口点来在唯一命名空间注册处（registry）注册所有接口，这样就防止了使用多个接口库时的命名冲突。

若希望了解关于这些接口的更多信息，可参考[NvInferPlugin.h](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/_nv_infer_plugin_8h.html)文件。使用接口创造器，可以调用`IPluginCreator::createPlugin()`函数，它会返回`IPluginV2`类型的接口对象。这个对象可以使用`addPluginV2()`函数添加到TensorRT网络，它会创建并添加一个层到网络，并将这一层与给定的接口绑定起来。这个方法也会返回一个指向这一层的`IPluginV2Layer`类型指针，通过它可以获得这一层，使用`getPlugin()`函数获得这个接口本身。例如，要以`pluginName`为名称、以`pluginVersion`为版本添加一个接口层到网络，可以参考下列步骤：

```cpp
// 使用外部函数 getPluginRegistry 来获得全局TensorRT接口注册处
auto creator = getPluginRegistry()->getPluginCreator(pluginName, pluginVersion);
const PluginFieldCollection* pluginFC = creator->getFieldNames();
// 为接口层分配域参数（比如 layerFields）
PluginFieldCollection *pluginData = parseAndFillFields(pluginFC, layerFields);
// 使用 layerName 和接口元数据创建接口对象
IPluginV2 *pluginObj = creator->createPlugin(layerName, pluginData);
// 使用网络API将接口添加到TensorRT网络
auto layer = network.addPluginV2(&inputs[0], int(intputs.size()), pluginObj);
... (build rest of the network and serialize engine);
pluginObj->destroy();  // 销毁接口对象
... (destroy network, engine, builder);
... (free allocated pluginData);
```

>注意：
>
>- 在传递到`createPlugin`之前，`pluginData`应该在堆上分配`PluginFields`入口；
>- `createPlugin`方法会在堆上创建一个新接口对象，并返回一个指向它的指针，用户需要想上面那样确保销毁`pluginObj`，以防内存泄漏。

在序列化期间，TensorRT引擎会为所有的`IPluginV2`类型接口在内部保存接口类型、版本、命名空间（如果存在）；在反序列化期间，TensorRT引擎会查询这个信息来从接口注册处找到接口创建器。这就使得TensorRT引擎能在内部调用`IPluginCreator::deserializePlugin()`方法。在反序列化期间创建的接口对象会被TensorRT引擎在内部通过调用`IPluginV2::destroy()`方法来销毁。

在前面的版本中，用户需要实现`nvinfer1::IPluginFactory`类来在反序列化期间调用`createPlugin`方法，这在使用TensorRT注册、使用`addPluginV2`添加的接口中不再需要。



#### 3.2 示例：使用C++添加自定义层

要使用C++添加一个自定义层，将其继承上面所述的基类中的一个，这里不需要动态输入，因此使用`IPluginV2IOExt`。对基于Caffe的网络，若使用TensorRT Caffe解析器，用户还需要继承来自`nvcaffeparser1::IPluginFactoryExt`（对`IPluginExt`类型的接口）和`nvinfer1::IPluginFactory`的类。下面的代码添加了一个名为`FooPlugin`的接口：

```cpp
class FooPlugin: public IPluginV2IOExt
{
  ... override all pure virtual methods of IPluginV@IOExt;
  ... Do not override the TRT_DEPRECATED methods.
};

class MyPluginFactory: public nvinfer1::IPluginFactory
{
  ... implement all factory methods;
};
```

若使用的是通过TensorRT接口注册处注册的`IPluginV2`类型接口，则无需实现`nvinfer1::IPluginFactory`类。

