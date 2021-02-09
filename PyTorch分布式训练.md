就PyTroch1.6而言，`torch.distributed`中的特征可以分为三个主要部分：

- Distributed Data-Parallel Training：DDP是一个被广泛应用的单程序多数据训练范式，通过DDP，模型被复制到每个进程上，而每个模型副本会输入数据样本的不同集合，DDP关注梯度通信来同步模型副本，并通过梯度计算来将其重叠从而加速计算。
- RPC-Based Distributed：RPC用于支持无法适配数据并行训练的通用训练结构，比如分布式管道并行、参数服务器范式、以及将DDP与其他训练范式结合。它有助于管理远程对象的生命周期，并将自动梯度扩展到机器边界之外。
- Collectivbe Communication ：c10d库支持使用组在进程间发送张量，既提供了collective communication接口（如`all_reduce`和`all_gather`），也支持P2P communication接口（如`send`和`isend`）。