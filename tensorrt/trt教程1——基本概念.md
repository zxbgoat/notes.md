#### 1.1 安装

可以选择

```bash
os="ubuntu1804"
tag="cuda10.2-trt7.2.1.6-ga-20201006"
sudo dpkg -i nv-tensorrt-repo-${os}-{tag}_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-${tag}/7fa2af80.pub
sudo apt-get update
sudo apt-get install tensorrt
```



#### 1.2 实例化

TensorRT使用`IExecutionContext`接口来进行推理，而创建`IExecutionContext`类对象，则需要创建`ICudaEngine`（引擎）类对象，有两种方式来创建引擎：

- 一种是通过用户模型的网络定义；
- 另一种是通过硬盘上存储的经过序列化的引擎。

在创建TensorRT对象前，一般会先创建一个全局的`ILogger`类对象，它是多种方法的参数：

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

