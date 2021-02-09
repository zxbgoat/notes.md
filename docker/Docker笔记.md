#### 一 介绍

Docker是一个新的容器化的技术，它轻巧，且易移植，号称“build once, configure once and run anywhere”，其主要特性包括：速度飞快以及优雅的隔离框架；CPU/内存的低消耗；快速开/关机；跨云计算基础构架。

##### 组件与元素

Docker有三个组件和三个基本元素。三个组件分别是：

- `Docker Client` 是用户界面，它支持用户与`Docker Daemon`之间通信。
- `Docker Daemon`运行于主机上，处理服务请求。
- `Docker Index`是中央registry，支持拥有公有与私有访问权限的Docker容器镜像的备份。

三个基本要素分别是：

- `Docker Containers`负责应用程序的运行，包括操作系统、用户添加的文件以及元数据。
- `Docker Images`是一个只读模板，用来运行Docker容器。
- `DockerFile`是文件指令集，用来说明如何自动创建Docker镜像。

<img src="figures/arc.png" width="600px" text-align="middle" />

先谈谈Docker的支柱，它使用以下操作系统的功能来提高容器技术效率：

- `Namespaces` 充当隔离的第一级。确保一个容器中运行一个进程而且不能看到或影响容器外的其它进程。
- `Control Groups`是LXC的重要组成部分，具有资源核算与限制的关键功能。
- `UnionFS`（文件系统）作为容器的构建块。为了支持Docker轻量级以及速度快的特性，它创建了用户层。

##### 组合起来

运行任何应用程序，都需要有两个基本步骤：

1. 构建一个镜像。
2. 运行容器。

这些步骤都是从`Docker Client`的命令开始的。`Docker Client`使用的是Docker二进制文件。在基础层面上，`Docker Client`会告诉`Docker Daemon`需要创建的镜像以及需要在容器内运行的命令。当Daemon接收到创建镜像的信号后，会进行如下操作：

**第1步：构建镜像**

`Docker Image`是一个构建容器的只读模板，它包含了容器启动所需的所有信息，包括运行程序和配置数据。
每个镜像都源于一个基本的镜像，然后根据Dockerfile中的指令创建模板。对于每个指令，在镜像上创建一个新的层面。

一旦镜像创建完成，就可以将它们推送到中央registry：`Docker Index`，以供他人使用。然而，`Docker Index`为镜像提供了两个级别的访问权限：公有访问和私有访问。你可以将镜像存储在私有仓库，Docker官网有私有仓库的套餐可以供你选择。总之，公有仓库是可搜索和可重复使用的，而私有仓库只能给那些拥有访问权限的成员使用。`Docker Client`可用于`Docker Index`内的镜像搜索。

**第2步：运行容器**

运行容器源于我们在第一步中创建的镜像。当容器被启动后，一个读写层会被添加到镜像的顶层。当分配到合适的网络和IP地址后，需要的应用程序就可以在容器中运行了。



#### 二 命令

首先，通过下面的命令来检查Docker的安装是否正确：

```bash
docker info 
```

