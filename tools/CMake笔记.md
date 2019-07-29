在linux下使用CMake生成Makefile并编译的流程如下：

1. 编写CMake配置文件CMakeLists.txt；
2. 执行`cmake path`或`ccmake path`生成Makefile文件，其中`path`为CMakeLists.txt所在目录；
3. 使用`make`命令进行编译。


**步骤1 基本起始点**

最基础的工程是从源码的可执行构建。对于简单工程而言两行CMakeLists.txt就已足够，如下：

```cmake
cmake_minimum_required (VERSION 2.6)
project (Tutorial)
add_executable(Tutorial tutorial.cxx)
```

CMake支持大写、小写或混合