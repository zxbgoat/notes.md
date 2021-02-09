#### 2.1 实例化TensorRT类

要执行推理，使用`IExecutionContext`接口；要创建`IExecutionContext`类的对象，首先创建`ICudaEngine`（引擎）类的对象。有两种方式来创建引擎。

- 通过用户模型的网络定义；
- 通过硬盘上序列化的引擎。

创建一个全局`ILogger`类对象，它是多种TensorRT API方法的参数，下面是一个创建logger的示例：

```cpp
class Logger: public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // supress info-level messages
        if(severity != Severity::kINFO) std::cout msg << std::endl;
    }
}
```

##### 2.1.1 通过用户模型的网络定义

使用TensorRT API的独立函数`createInferBuilder(gLogger)`来创建`IBuilder`类对象；使用`IBuilder::createNetworkV2`来创建`INetworkDefinition`对象；使用`INetworkDefinition`作为输入来创建下面几种解析器：

- ONNX：`auto parser = nvonnxparser::createParser(*network, gLogger)`；
- Caffe：  `auto parser = nvcaffeparser1::createParser()`；
- UFF：    `auto parser = nvuffparser::createUffParser()`；

调用`IParser::parse()`方法来读入模型文件并构成（populate）TensorRT网络；调用`IBuilder::buildEngineWithConfig()`方法来创建`ICudaEngine()`类对象；引擎能被序列化或转储到文件；创建并使用一个执行上下（execution context）文来进行推理。

##### 2.1.2 通过硬盘上序列化的引擎

若序列化的引擎已经被保存到文件中，则可以跳过上面的大多数步骤。可以使用TensorRT API的独立函数`createInferRuntime(gLogger)`来创建`IRuntime`类的对象，并调用`IRuntime::deserializeCudaEngine()`方法来创建引擎。

无论是直接从网络构建，还是从文件反序列化引擎，剩余的推理过程都是一致的。构造器（builder）和运行时（runtime）会随着与创建线程关联的GPU上下文被创建，尽管上下文不存在时会有一个默认的被创建，但依然建议介在创建一个运行时或构造器对象前创建并配置CUDA上下文。

#### 2.2 创建网络定义

执行推理的第一步是创建是从模型创建一个TensorRT网络，有两种方法来实现：

- 最简单的是使用TensorRT解析器库来导入模型，它支持ONNX、Caffe和UFF三种模型；
- 另一个是直接使用TensorRT API定义模型，这需要执行一些API调用来定义：
  - 模型的每一层，
  - 导入训练好模型参数的机制。

无论使用哪种方法，都需要显式地告诉TensorRT哪些张量是推理的输出，未被标记为输出的张量会被认为是暂时变量，可能会被构造器优化掉。对于输出张量的数量并无限制，将张量标记为输出可以避免在其上的一些优化。

输入和输出张量必须（使用`ITensor::setName()`）给定名称，在推理时需要向引擎提供一些输入和输出缓冲区的指针；为确定引擎按照怎样的顺序期待（expect）这些指针，可以使用张量名称来查询。

TensorRT网络定义的一个重要方面是它包含模型参数的指针，这些指针会被构造器复制到优化后的引擎；若网络通过解析器创建，则这个解析器拥有这些权值占据的内存，因此解析器对象在构造器运行之前都不能被删除。

##### 2.2.1 从头创建一个网络定义

可以使用网络定义API直接将网络定义到TensorRT，这种场景假设每一层权值都已经在主机内存中准备好来在网络创建时传送给TensorRT。下面的例子会创建一个包含输入、卷积、池化、全连接、激活和SoftMax的简单网络。

1. 创建构造器和网络：

   ```cpp
   // create the builder
   IBuilder* builder = createInferBuilder(gLogger);
   // create the network
   const auto explicitBatch = 1U << static_cast<uint32_t>(
       NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
   INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
   ```

2. 添加输入层到网络，网络可以有多个输入：

   ```cpp
   // add input layer with input dimensions, including dynamic batch
   auto data = network->addInput(INPUT_BLOB_NAME, dt,
                                 Dims3{-1, 1, INPUT_H, INPUT_W});
   ```

3. 添加卷积层，

   ```cpp
   // add the convolution layer with hidden layer input nodes, strides, 
   // and weights for filter and bias
   auto conv1 = network->addConvolution(*data->getOutput(0), 20, DimsHW{5, 5},
                                        weightMap["conv1filter"],
                                        weightMap["conv1bias"]);
   conv1->setStride(DimsHW{1, 1});
   ```

   >注意：传递给TensorRT层的权值在主机内存中。

4. 添加池化层：

   ```cpp
   // add the pooling layer
   auto pool1 = network->addPooling(*conv1->getOutput(0), 
                                    PoolingType::kMax, DimsHW{2, 2});
   pool1->setStride(DimsHW(2, 2));
   ```

5. 添加全连接和激活层：

   ```cpp
   // add the full-connected and acttivation layers
   auto ip1 = network->addFullyConnected(*pool1->getOutput(0), 500,
                                         weightMap["ip1filter"],
                                         weightMap['ip1bias']);
   auto relu1 = network->addActivation(*ip1->getOutput(0),
                                       ActivationType::kRELU);
   ```

6. 添加SoftMax层来计算最终的概率：

   ```cpp
   // add softmax layer
   auto prob = network->addSoftMax(*relu1->getOutput(0));
   // set as output
   prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
   ```

7. 标记输出：

   ```cpp
   // mark the output
   network->markOutput(*prob->getOutput(0));
   ```

##### 2.2.2 使用解析器导入模型

构造器必须在网络之前创建，因为它表现为网络的工厂。不同的解析器有不同的创建网络输出的机制。

要导入模型，需要执行下面的几个步骤：

1. 创建TensorRT构造器；
2. 为特定的格式创建TensorRT网络和解析器；
3. 使用解析器来解析导入的模型，并构成网络。

##### 2.2.3 导入Caffe模型

下面的步骤展示如何倒入Caffe模型：

1. 创建构造器和网络：

   ```cpp
   // create the builder and the network
   IBuilder* builder = createInferBuilder(gLogger);
   INetworkDefinition* network = builder->createNetworkV2(0U);
   ```

2. 创建Caffe解析器：

   ```cpp
   // create the caffe parser
   ICaffeParser* parser = createCaffeParser();
   ```

3. 解析导入的模型：

   ```cpp
   // parse the imported model
   const IBlobNameToTensor* blobNameToTensor = parser->parse("deploy_file", "modelFile", 
                                                             *network, DataType::kFLOAT);
   ```

   这会从caffe模型构成TensorRT网络，最后的参数通知解析器生成权值为32位浮点的网络，使用`DataType::kHALF`会生成16为浮点的模型。

   此外，解析器还会返回一个将Caffe blob名称映射为TensorRT张量的字典。不同于Caffe，TensorRT网络定义中没有就地操作的概念，当Caffe模型使用就地操作时，字典中返回的TensorRT张量会对应到最后写到的blob。比如一个卷积写到一个blob后跟随一个就地的ReLU，这个blob的名称会映射到ReLU输出的张量。

4. 指定网络的输出：

   ```cpp
   // specify the outputs of the network
   for(auto& s: outputs)
       network->markOutput(*blobNameToTensor->find(s.c_str()));
   ```

##### 2.2.4 使用UFF解析器API导入TensorFlow模型

下面的例子展示如何导入TensorFlow模型。

>注意：对于新工程，推介使用TF-TRT集成作为转换TensorFlow网络的方法，以使用TensorRT推理。

导入TensorFlow模型，需要先将TensorFlow模型转化为中间的UFF格式。

1. 创建网络和构造器：

   ```cpp
   // create the builder
   IBuilder* builder = createInferBuilder(gLogger);
   // create the network
   INetworkDefinition* network = builder->createNetworkV2(0U);
   ```

2. 创建UFF解析器：

   ```cpp
   // create the uff parser
   IUFFParser* parser = createUffParser();
   ```

3. 声明网络的输入和输出：

   ```cpp
   // declare the network inputs to uff parser
   parser->registerInput("Input_0", DimsHW{1, 28, 28}, UffInputOrder::kNCHW);
   // declare the network onputs to uff parser
   parser->registerOutput("Binary_3");
   ```

4. 解析导入模型以构成网络：

   ```cpp
   // parse the imported model to populate the network
   parser->parse(ufFile, *network, nvinfer1::DataType::kFLOAT);
   ```

##### 2.2.5 导入ONNX模型

下面的示例展示如何导入ONNX模型。

> 注意：通常更新版的ONNX解析器会向后兼容到opset 7，当变化不是可向后兼容时会出现一些异常，这种情况下，将更早的ONNX模型文集爱你转化为更后一些支持的版本。
>
> 也可能有一些导出工具生成的用户模型主持比TensorRT的ONNX解析器所支持的`opset`更后，这种情况下，验证GitHub上发布的最新TensorRT是否支持所要求的版本；支持版本由`onnx_trt_backend.cpp`中的`BACKEND_OPSET_VERSION`变量定义。下载并构建最新版的ONNX TensorRT解析器。
>
> 另外在TensorRT 7.0中，ONNX解析器仅支持全方位模式，这意味着模型必须用`explicitBatch`标记设置来创建。
>
> UFF不支持`explicitBatch`或动态形状的输入。

1. 创建构造器和网络：

   ```cpp
   // create the builder
   IBuilder* builder = createInferBuilder(gLogger);
   // create the network
   const auto explicitBatch = 1U << static_cast<uint32_t>(
       NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
   INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
   ```

2. 创建ONNX解析器：

   ```cpp
   // create the onnx parser
   nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
   ```

3. 解析网络：

   ```cpp
   // parse the model
   parser->parseFromFile(onnx_filename, ILogger::Severity::kWARNING);
   for(int i = 0; i < parser.getNbErrors(); ++i)
       std::cout << parser->getError(i)->desc << std::endl;
   ```

#### 2.3 创建引擎

下一步是调用TensorRT构造器来创建优化的运行时，构造器中的一个函数就是为最快的实现搜索CUDA核心的目录，因此使用与优化后的引擎运行相同的GPU来构建是有必要的。

`IBuilderConfig`有许多特性，可以用它来设置在哪个GPU上需要运行的精度这样的事情，以及自动调整当探明哪个最快时TensorRT应该对每个核心计时多少次这样的参数（更多的迭代会导致更长的运行时，但对噪音更少的敏感性）。也可以对构造器进行查询来找出硬件本地支持哪种类型的简化精度。

一个特别主要的特性是最大工作空间大小：层算法通常要求暂时的工作空间，这个参数限制网络中任何层能使用的最大尺寸。若未提供足够的初始设置，可能导致TensorRT对给定的层无法找到实现。

1. 使用构造对象创建引擎

   ```cpp
   // build the engine using builder object
   IBuilderConfig* config = builder->createBuilderConfig();
   config->setMaxWorkspaceSize(1 << 20);
   ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
   ```

2. 免除（dispense）网络、构造器和解析器：

   ```cpp
   // 2. dispense with the network, builder, parser if using one
   parser->destroy();
   network->destroy();
   config->destroy();
   builder->destroy();
   ```

##### 2.3.1 构造器层计时缓存

构建引擎时非常耗时的，因为构造器需要对每个层的候选核心计时，为减少构造器时间，TensorRT会在构造器期间设置一个层计时缓存来保存层的描述信息。

若有其他层有相同的输入/输出张量配置和层参数，则TensorRT构造器会对重复的层跳过描述并再使用缓存的结果。层计时缓存默认是打开的，可以通过设置构造器标记来关闭：

```cpp
config->setFlag(BuilderFlag::kDISABLE_TIMING_CACHE);
```

#### 2.4 序列化模型

所谓序列化，就是将引擎变形为一种格式来在后面的推理中存储并使用。

> 注意：序列化的引擎在不同的平台和TensorRT版本中不是可用的，引擎是特定于它们所构建的GPU的模型的（以及平台和TensorRT版本）。

1. 运行构造器作为一个优先线下步骤，然后序列化：

   ```cpp
   // run the builder as a prior offline step and then serialize
   IHostMemory *serializeModel = engine->serialize();
   // store model to disk
   // <...>
   serializeModel->destroy();
   ```

2. 创建一个运行时来解序列化：

   ```cpp
   // create a runtime object to deserialize
   IRuntime* runtime = createInferRuntime(gLogger);
   ICudaEngine* engine = runtime->deserializeCudaEngine(modelData, modelSize, nullptr);
   ```

最后的参数是用于使用定制层应用的插件层工厂。

#### 2.5 执行推理

下面的步骤展示在获得引擎后如何执行推理。

1. 创建一些空间来存储中间激活值，因为引擎持有网络定义和训练后的参数，额外的空间时必须的，这些都掌握在一个执行上下文中：

   ```cpp
   // 1. create some space to store intermediate activation values
   IExecutionContext *context = enqine->createExecutionContext();
   ```

   一个引擎可以有多个执行上下文，这使得一个参数集合能被多个重叠的推理任务所使用，例如可以使用一个引擎在并行CUDA流中处理图像，而每个流一个上下文。每个上下文都会在同一个GPU上被创建为与引擎。

2. 使用输入和输出blob名称来获得对应的输入和输出索引：

   ```cpp
   // 2. get the corresponding input and output index
   int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
   int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
   ```

3. 使用这些索引设置一个缓冲区数组指向GPU上的输入和输出缓冲区：

   ```cpp
   // 3. set up a buffer array pointing to the input and output buffer on the GPU
   void* buffers[2];
   buffers[inputIndex] = inputBuffer;
   buffers[outputIndex] = outputBuffer;
   ```

4. TensorRT执行通常是异步的，因此将核心排队到一个CUDA流中：

   ```cpp
   // enqueue the kernels on a CUDA stream
   context->enqueueV2(buffers, stream, nullptr);
   ```

   若数据尚未就位，通常会在核心前和后将异步的`memcpy()`函数排队来从GPU移动数据。`enqueueV2()`最后的参数是一个可选的CUDA事件，当输入缓冲区被消费后它会发送消息，内存就被安全地使用。

   要确定核心（以及可能的`memcpy()`）何时结束，使用标准的CUDA同步机制，如时间、或在流上等待。

#### 2.6 内存管理

TensorRT提供两种机制来使应用在设备内存上有更多的控制。默认情况下，在创建`IExecutionContext`时，持续的设备内存会被分配来存储激活数据。调用`createExecutionContextWithoutDeviceMemory`来避免这种分配，这样以后应用就必须调用`IExecutionContext::setDeviceMemory()`提供所要求的内存来运行网络。内存块的大小由`ICudaEngine::getDeviceMemorySize()`返回。

另外，应用可以在构建和运行时期间，通过实现`IGpuAllocator`接口，来提供定制的分配器以供使用。这在应用希望控制所有GPU内存并代替TensorRT分配时非常有用。一旦接口实现以后，在`IBuilder`或`IRuntime()`接口上调用：

```cpp
setGpuAllocator(&allocator);
```

这样所有的设备内存都会通过这个接口分配和释放。

#### 2.7 改装（refit）引擎

TensorRT可以用新的权值在无需重新构建的情况下改装一个引擎，但这个引擎必须被构建为“可改装的（refittable）”，因为引擎优化的方式，当改变了一些权值后，必须提供一些其他的权值，这个接口能告诉你需要什么样的额外权值。下面是这个过程：

1. 在构建前需要设置引擎为可改装的：

   ```cpp
   // ...
   config->setFlag(BuilderFlag::kREFIT);
   builder->builderEngineWithConfig(network, config);
   ```

2. 创建一个改装器（refitter）对象：

   ```cpp
   ICudaEngine* engine = ...;
   IRefitter* refitter = createInferRefitter(*engine, gLogger);
   ```

3. 更新需要更新的权值，例如更新一个名为`MyLayer`卷积层核心的权值：

   ```cpp
   Weights newWeights = ...;
   refitter->setWeights("MyLayer", WeoightsRole::kKERNEL, newWeights);
   ```

   新权值的总数必须与用于构建引擎的原始权值相同。若程序出错，比如一个层名称出错、参数角色（role）出错、或权值总数不同，`setWeights`方法会返回`false`。

4. 找到需要提供的其他权值，这通常需要调用`IRefitter::getMissing`两次，第一次获得提供权值对象的数量，第二次则获得它们的层和角色：

   ```cpp
   const int n = refitter->getMissing(0, nullptr, nullptr);
   std::vector<const char*> layerNames(n);
   std::vector<WeightsRole> weightsRoles(n);
   refitter->getMissing(n, layerNames.data(), weightsRoles.data());
   ```

5. 之后可以以任意的顺序，提供缺失的权值：

   ```cpp
   for(int i = 0; i < n; ++i)
   {
     refitter->setWeights(layerNames[i], weightsRoles[i], Weights{...});
   }
   ```

   仅提供缺失的权值不会产生更多参数的需求，而提供任意的额外参数则会出发更多参数的需求。

6. 用所有提供的权值来更新引擎：

   ```cpp
   bool success = refitter->refitCudaEngine();
   assert(success);
   ```

   若`success`是`false`，检查诊断日志，可能与权值依然缺失相关。

7. 销毁改装器：

   ```cpp
   refitter->destroy();
   ```

更新后的引擎就表现得犹如它是从用新权值更新后的网络构建的。要查看一个引擎中所有可更新的权值，想第3步中`getMissing`使用地那样，使用`refitter->getAll()`。

#### 2.8 算法选择

TensorRT提供了一种机制来控制网络中不同层的算法选择，它的默认行为是选择能够全局最小化引擎执行时间的算法。

**`IAlgorithmsSelector`**

通过实现`IAlgorithmSelector`接口，应用可以提供一个定制的算法选择器来在构建引擎时使用。一旦实现接口后，调用

```cpp
config.setAlgorithmSelector(&selector);
```

其中`config`是被传递给`IBuilder::createBuilderConfig`的来构建引擎的`createBuilderConfig`，而`selector`则是从`IAlgorithmSelector`派生类的实例。

**`IAlgorithmSelector::selectAlgorithms`**

这个方法能够使应用引导算法的选择，需要向它提供层的算法上下文和适用于这个上下文的`IAlgorithm`选项列表，可以使用这个方法的重载来指示TensorRT应该考虑哪些选项，基于任意的启发方法，或者返回所有选项如果TensorRT应该考虑所有选择。

从`selectAlgorithm`返回的选项限制了一些层所允许的算法范围，构造器会使用允许的选项作全局优化。若没有选项返回，TensorRT会回退到其默认行为，可以复原`BuilderFlag::kSELECT_TYPES`来在重载返回空列表时避免这种回退并获得一个错误。若重载只返回一个选项，则它保证会被使用。

**`IAlgorithmSelector::reportAlgorithms`**

重载`reportAlgorithms`方法能用来记录TensorRT为每一层所做的最后选项。对给定的优化情况（profile），TensorRT在所有`selectAlgorithms`调用后调用`reportAlgorithms`。要在后面的构建中重现前面的构建，使`selectAlgorithms`方法返回`reportAlgorithms`方法在前面构建中报告的相同方法。

>注意：
>
>- 算法选择中“层”的概念与`INetworkDefinition`中的`ILayer`不同，依据融合优化前者的“层”等价于多个`ILayer`的聚集；
>- 在`selectAlgorithms`中选择最快的算法可能无法获得整个网络上的最佳性能。TensorRT最小化整个网络的计时，作为减少重定格式负载的交换可能会脱离局部贪心选项；
>- `reportAlgorithms`方法并不提供`IAlgorithm`的计时和工作空间要求，可以使用`selectAlgorithms`方法来查询这些信息；
>- `IAlgorithmContext`和`IAlgorithm`序列，以及每个`IAlgorithm`的计时，在每次构建中可能不同。

##### 2.8.1 构造器中的确定性和再现性

TensorRT的默认行为是选择引擎执行时间全局最小的层实现，但而对一个算法的执行时间实验几乎从不相等，如果两个算法有相同的计时，同样算法的结果也可能不是每次都更快，因此结果就是每次构造所选择的方法可能不同，即便网络和构造配置不变。

但是算法选择API可以用来确定性地构建TensorRT引擎，`IAlgorithmSelector::selectAlgorithms`方法可以为层从选项列表中选择算法，通过总是返回相同的选项，就能为层强制相同的选项。

`IAlgorithmSelector`也可以再现相同的实现，`IAlgorithmSelector::reportAlgorithms`可用于缓存算法选择，`selectAlgorithms`方法就能用来选择缓存中记录的算法。若为每个层名、实现、策略和输入/输出格式的组合返回相同的算法选项，就能获得相同的引擎。

类`sampleAlgorithmSelector`展示了如何使用算法选择器来在构造器中获得确定性和再现性。