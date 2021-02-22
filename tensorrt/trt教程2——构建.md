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

   



#### 2.2 创建引擎

