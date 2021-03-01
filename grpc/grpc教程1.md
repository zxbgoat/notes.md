这个示例是一个简单的路由（route）映射应用，客户端获取路由的相关特征（feature）信息，创建它们路由的概述（summary），并与服务端以及其他客户端交换路由信息。通过grpc，用户可以在`.proto`文件中一次定义自己的服务，然后一任意grpc支持的语言生成客户端和服务器，反过来这些客户端和服务器可以运行在各种各样的设备中。

##### 定义服务

第一步是使用`protocol buffer`定义grpc服务（`service`）、请求（`request`）方法和响应类型（`type`）。要定义一个服务，需要在自己的`.proto`文件中指定一个命名`service`：

```protobuf
service RouteGuide
{}
```

然后在服务定义中定义`rpc`方法，指定它们的请求和响应类型。grpc可以定义四种服务方法，所有这些都应用在`RouteGuide`服务中：

- 一个简单的rpc，其中客户端使用存根（stub）向服务器发送一个请求，等待返回的响应：

  ```protobuf
  // Obtains the feature at a given position.
  rpc GetFeature(Point) returns (Feature) {}
  ```

- 一个服务端的流式RPC，其中客户端向服务器发送一个请求，并获得一个流来读取一系列的信息，客户端读取返回的流直到没有信息为止。如下所示，用户可以在响应类型关键字前加一个`stream`关键字来指定一个服务端的流式方法。

  ```protobuf
  // Obtains the Features available within the given Rectangle.  Results are
  // streamed rather than returned at once (e.g. in a response message with a
  // repeated field), as the rectangle may cover a large area and contain a
  // huge number of features.
  rpc ListFeatures(Rectangle) returns (stream Feature) {}
  ```

- 一个客户端流式RPC，客户端也使用提供的流写一系列消息到它们的服务器，客户端写完消息后就等待服务器读取消息并返回其响应：

  ```protobuf
  // Accepts a stream of Points on a route being traversed, returning a
  // RouteSummary when traversal is completed.
  rpc RecordRoute(stream Point) returns (RouteSummary) {}
  ```

- 一个双向流式RPC，两端都是用一个读写流发送一系列消息，两个流互相独立，因此客户端和服务器可以任意顺序读或写，例如服务器可以等到接收完所有消息再写它的响应，也可以交替读一个消息然后写一个消息，或其他方式的结合。每个流的消息顺序都会保留。

  ```protobuf
  // Accepts a stream of RouteNotes sent while a route is being traversed,
  // while receiving other RouteNotes (e.g. from other users).
  rpc RouteChat(stream RouteNote) returns (stream RouteNote) {}
  ```

用户的`.proto`文件还应包含