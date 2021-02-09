gRPC既可以将protocol buffers用作界面定义语言（Interface Definition Language，IDL），又可以作为起底部的消息交换格式。

##### 概览

在gRPC中，客户应用可以直接调用位于不同机器服务应用上的方法，就像它是一个本地对象一样，这样就能很方便地创建分布式应用和服务。像在许多RPC系统中一样，gRPC基于这样的思想：定义一个服务 ，指明能够被远程调用的方法，以及它们的参数和返回类型。在服务端，服务实现这个接口并运行一个gRPC服务来处理客户端调用；在客户端，客户有一个存根（stub，在一些语言中就叫客户）提供像服务一样的方法。

<img src='landing-2.svg' />

gRPC客户和服务可以在许多环境中运行及相互交流——从谷歌内部的服务器到用户的桌面电脑——并且可以用任何gRPC支持的语言编写，而最新的谷歌API会有gRPC版的接口。

##### 用protocol buffers运行

gRPC默认使用protocol buffers，谷歌成熟的用于序列化结构化数据的开源机制，下面快速介绍它如何工作：

1. 首先是定义希望在proto文件中序列化的数据的结构，这是一个以`.proto`作为扩展名的普通文本文件。Protocol Buffers数据结构化为消息（message），每个消息都是一个信息的小型记录（record），信息包含一序列被称为域名-值对，如下例所示：

   ```protobuf
   message Person
   {
       string name = 1;
       int32 id = 2;
       bool has_ponycopter = 3;
   }
   ```

2. 在指定数据结构后，就可以使用protocol buffer编译器`protoc`从proto定义生成所偏好语言的数据通道（access）类，这些类为每个域都提供了简单的存取方法，像`name()`和`set_name()`，以及序列化/解析整个结构到/自原始字节的方法。因此，如果选择C++语言，在上面的例子中运行编译器会生成一个被称为`Person`的类，可以使用这个类来分发（populate）、序列化以及获取`Person` protocol buffers的消息。在普通proto文件中定义gRPC服务，使用指定为protocol buffer消息的RPC方法参数和返回类型。

   ```cpp
   // The greeter service definition.
   service Greeter
   {
     // Sends a greeting
     rpc SayHello (HelloRequest) returns (HelloReply) {}
   }
   
   // The request message containing the user's name.
   message HelloRequest
   {
     string name = 1;
   }
   
   // The response message containing the greetings
   message HelloReply
   {
     string message = 1;
   }
   ```

gRPC通过一个特殊的gRPC接口来用`protoc`从proto文件生成代码：生成的gRPC客户和服务代码，以及用于分发、序列化和获取消息类型的普通protocol buffer代码。

