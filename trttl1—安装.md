可以选择

```bash
os="ubuntu1804"
tag="cuda10.2-trt7.2.1.6-ga-20201006"
sudo dpkg -i nv-tensorrt-repo-${os}-{tag}_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-${tag}/7fa2af80.pub
sudo apt-get update
sudo apt-get install tensorrt
```

若使用python3，运行

```bash
sudo apt-get install python3-libnvinfer-dev
```

会安装`python3-libnvinfer`。

