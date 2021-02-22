##### 设置docker

```bash
curl https://get.docker.com | sh \
  && sudo systemctl start docker \
  && sudo systemctl enable docker
```



##### 设置NVIDIA容器工具集

设置`stable`仓库和GPG密钥：

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

安装`nvidia-docker2`包：

```bash
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

重启docker守护进程以设置默认运行时：

```bash
sudo systemctl restart docker
```

使用一个基础CUDA容器测试安装是否成功：

```bash
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

若安装成功，会有下面这样的输出：

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| N/A   34C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```



##### 设置CLion调试环境

```bash
sudo docker run -itd \
                --cap-add sys_ptrace \
                --runtime=nvidia \
                -p 2222:22 \
                --name dbg \
                nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 \
                /bin/bash
```

