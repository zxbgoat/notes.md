#### 2.1 创建网络定义

执行推理的第一步是创建是从模型创建一个TensorRT网络，有两种方法来实现：

- 最简单的是使用TensorRT解析器库来导入模型，它支持ONNX、Caffe和UFF三种模型；
- 另一个是直接使用TensorRT API定义模型，这需要执行一些API调用来定义模型的每一层，以及导入训练好模型参数的机制。

无论使用哪种方法，都需要显式地告诉TensorRT哪些张量是推理的输出，未被标记为输出的张量会被认为是暂时变量，可能会被构造器优化掉。对于输出张量的数量并无限制，将张量标记为输出可以避免在其上的一些优化。

输入和输出张量必须（使用`ITensor::setName()`）给定名称，在推理时需要向引擎提供一些输入和输出缓冲区的指针；为确定引擎按照怎样的顺序期待（expect）这些指针，可以使用张量名称来查询。

TensorRT网络定义的一个重要方面是它包含模型参数的指针，这些指针会被构造器复制到优化后的引擎；若网络通过解析器创建，则这个解析器拥有这些权值占据的内存，因此解析器对象在构造器运行之前都不能被删除。

##### 2.2.1 从头创建网络定义

可以使用网络定义API直接将网络定义到TensorRT，这种场景假设每一层权值都已经在主机内存中准备好来在网络创建时传送给TensorRT。下面的例子会创建一个包含输入、卷积、池化、全连接、激活和SoftMax的简单网络：

1. 创建构造器和网络定义：

   ```cpp
   // create the builder
   IBuilder* builder = createInferBuilder(gLogger);
   // create the network
   const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
   INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
   ```

2. 添加输入层到网络，网络可以有多个输入：

   ```cpp
   // add input layer with input dimensions, including dynamic batch
   auto data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{-1, 1, INPUT_H, INPUT_W});
   ```

3. 添加卷积层：

   ```cpp
   // add the convolution layer with hidden layer input nodes, strides, and weights for filter and bias
   auto conv1 = network->addConvolution(*data->getOutput(0), 20, DimsHW{5, 5},
                                        weightMap["conv1filter"],
                                        weightMap["conv1bias"]);
   conv1->setStride(DimsHW{1, 1});
   ```

4. 添加卷积层：

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

构造器必须在网络之前创建，因为它的作用是网络工厂，不同的解析器标识网络输出的机制各不相同。当前TensorRT支持三种类型的解析，分别是Caffe模型、TensorFlow的UFF模型和ONNX模型。要导入模型，需要执行下面的几个步骤：

1. 创建构造器：

   ```cpp
   // create the builder and the network
   IBuilder* builder = createInferBuilder(gLogger);
   ```

2. 为特定的格式创建TensorRT网络和解析器：

   - **onnx模型**

     ```cpp
     const auto explicitBatch = 1U << static_cast<uint32_t>(
         NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
     INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
     auto parser = nvonnxparser::createParser(*network, gLogger);
     ```

   - **UFF模型**

     ```cpp
     // create the network
     INetworkDefinition* network = builder->createNetworkV2(0U);
     // create the uff parser
     auto parser = createUffParser();
     ```

   - **Caffe模型**

     ```cpp
     INetworkDefinition* network = builder->createNetworkV2(0U);
     // create the caffe parser
     auto parser = createCaffeParser();
     ```

3. 使用解析器来解析导入的模型，并分配到网络。



#### 2.2 创建引擎

`IBuilderConfig`有许多特性，可以用它来设置网络需要运行的精度、在探明最快参数（比如每个核心应该计时的次数）后自动进行调整（迭代越多，运行时间越长，但对噪音也更鲁棒）。也可以使用构造器来查询来硬件本地支持的简化精度类型。其中比较重要的一个特性是最大工作空间大小：层算法通常需要临时工作空间，这个参数限制了网络每一层能使用的最大空间，若未提供足够的初始值，可能会导致TensorRT无法找到给定层的实现。

##### 2.2.1 引擎的构建与销毁

下一步是调用构造器来创建一个经过优化的运行时（runtime），构造器的一个功能就是搜索CUDA核心（kernel）目录来找到最快的实现，因此需要使用与引擎将要运行的相同GPU来构建。构建引擎的过程如下：

1. 使用构造对象创建引擎

   ```cpp
   // build the engine using builder object
   IBuilderConfig* config = builder->createBuilderConfig();
   config->setMaxWorkspaceSize(1 << 20);
   ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
   ```

2. 销毁（dispense）网络、构造器和解析器：

   ```cpp
   // 2. dispense with the network, builder, parser if using one
   parser->destroy();
   network->destroy();
   config->destroy();
   builder->destroy();
   ```

##### 2.2.2 构造器层计时缓存

构建引擎非常耗时，因为构造器需要对每个层的候选核心计时，为减少构造器时间，TensorRT会在构造器期间设置一个层计时缓存来保存层的描述信息。若其他层有相同的输入/输出张量和参数，则TensorRT构造器会直接使用缓存的结果。层计时缓存默认是打开的，可以通过设置构造器标记来关闭：

```cpp
config->setFlag(BuilderFlag::kDISABLE_TIMING_CACHE);
```



#### 2.3 执行推理

在获得引擎后，可以通过下面的步骤来执行推理：

1. 因为引擎持有网络的定义参数，因此需要创建一些空间来保存中间激活值，这些都掌握在一个执行上下文中：

   ```cpp
   // 1. create some space to store intermediate activation values
   IExecutionContext *context = enqine->createExecutionContext();
   ```

   一个引擎可以有多个执行上下文，可以使一个参数集合能被多个重叠的推理任务使用。比如可以使用一个引擎在并行的CUDA流中处理图像，而每个流一个上下文，每个上下文都会被创建在与引擎相同的GPU上。

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

   要确定核心（以及可能的`memcpy()`）何时结束，使用标准的CUDA同步机制，如时间、或在流上等待。对于隐batch网络，参考[enqueue()](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_execution_context.html#a84436f784eb3f0ea9089de2678d77954)获得更多信息；显batch网络则参考[enqueueV2()](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_execution_context.html#ac7a5737264c2b7860baef0096d961f5a)。

