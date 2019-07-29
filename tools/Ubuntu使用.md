1. **验证可用cuda的GPU**：

   ```bash
   lspci | grep -i invidia
   ```

   若未见任何设置，则在命令行输入`update-pciids`（通常在`/sbin	`中）然或再执行前面的命令。若图形卡来自Nvidia并且在[列表](http://developer.nvidia.com/cuda-gpus)中，则GPU是CUDA可用的。

   ```python
   01:00.0 3D controller: NVIDIA Corporation GM107M [GeForce GTX 960M] (rev a2)
   ```

   **验证Linux版本受支持**：

   ```bash
   uname -m && cat /etc/*release
   ```

   应该看到类似这样的内容：

   ```bash
   x86_64
   DISTRIB_ID=Ubuntu
   DISTRIB_RELEASE=16.04
   ...
   ```

   **验证安装了GCC**：

   ```bash
   gcc --version
   ```

   未报错即可。

   **验证系统有正确的Kernel Headers和开发包**：CUDA驱动安装或重构时，要求所运行kernel版本的kernel头和开发包。Runfile文件安装时不会执行包验证，Deb驱动文件安装时会尝试安装这些工具若尚未安装，但会安装最新的版本，可能与系统所运行kernel版本不符，因此最好手工确保安装正确的版本。查看系统kernel版本：

   ```bash
   uname -r
   ```

   安装相关工具：

   ```bash
   sudo apt-get install linux-headers-$(uname -r)$
   ```

   

2. **安装方法选择**

   有针对特定发行包（RPM和Deb包）和发行办无关包（runfile包）两种安装方法，**若可以的话推介前者**。CUDA工具集，包含CUDA驱动，以及创造、构建和运行CUDA应用的工具、库、头文件，以及cuda示例代码及其他资源。针对tensorflow当前版本(1.4)要求，在[这里](http://developer.nvidia.com/cuda-downloads)下载8.0版工具包。若需验证的话，可用md5sum（这一步可选）：

   ```
   md5sum <file>
   ```

   再与[这里](http://developer.nvidia.com/cuda-downloads/checksums)相应的序列对比即可。

   **解决冲突的安装方法**：安装cuda之前，应该卸载先前任何可能冲突的安装。先前未安装cuda、或者安装方法保持一致的系统（因此推介一直使用Deb文件安装）。查看下表确定是否需要卸载：

   <img src="1.png" />

   若需卸载runfile安装的工具集，运行：

   ```bash
   sudo /usr/local/cuda-X.Y/bin/uninstall_cuda_X.Y.pl
   ```

   若需卸载runfile按照的去驱动，运行：

   ```bash
   sudo /usr/bin/nvidia-uninstall
   ```

   若需卸载Deb文件的安装：运行：

   ```bash
   sudo apt-get --purge remove <package_name>
   ```

   

3. 包管理器安装

   **安装元数据仓库**：

   ```python
   sudo dpkg -i cuda-repo-<distro>_<version>_<architecture>.deb
   ```

   **安装cuda的GPG公钥**——若使用的是**本地**repo：

   ```bash
   sudo apt-key add /var/cuda/-repo-<version>/7fa2af80.pub
   ```

   **安装cuda的GPG公钥**——若使用的是**网络**repo：

   ```bash
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/<distro>/<architecture>/7fa2af80.pub
   ```

   **更新apt仓库缓存**：

   ```bash
   sudo apt-get uodate
   ```

   **安装cuda**：

   ```bash
   sudo apt-get install cuda
   ```

   **查看所安装的包**：

   ```python
   cat /var/lib/apt/lists/*cuda*Packages | grep "Package:"
   ```

   

4. 安装后的操作

   **设置`PATH`**，将下面的语句写入bashrc文件中：

   ```bash
   export PATH=/usr/local/cuda-<version>/bin${PATH:+:${PATH}}
   ```

   若是使用runfile安装，还需**改变系统环境变量**：

   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   ```

   修改以后注意运行`source .bashrc`使修改生效。

   **安装可写的示例**：为修改、编译和运行示例，示例必须以写权限安装，运行：

   ```bash
   cuda-install-samples-<version>.sh <dir>
   ```

   **查看驱动版本**，可运行：

   ```bash
   cat /proc/driver/nvidia/version
   ```

   **编译验证**：cuda工具集的版本可通过`nvcc -V`查看，转到刚才安装的示例目录，运行`make`。产生的二进制文件在其`bin`目录下。转到此目录，运行：

   ```bash
   ./deviceQuery
   ```

   若最后的结果为PASS，则成功安装cuda。

   

5. **安装cuDNN**

   针对tensorflow要求，[下载](https://developer.nvidia.com/rdp/cudnn-download)6.0版工具包。有两种选择，一是使用针对特定平台的Deb文件安装，另一个是平台无关的库。若选择**deb文件**，则执行：

   安装运行时库

   ```bash
   sudo dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb
   ```

   安装开发者库

   ```bash
   sudo dpkg -i libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb
   ```

   安装代码示例和用户指南

   ```bash
   sudo dpkg -i libcudnn6-doc_6.0.21-1+cuda8.0_amd64.deb
   ```

   而若选择**库文件**，则解压下载的文件，执行：

   ```bash
   cd <installpath>/lib
   export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
   ```

   将`<installpath>`添加到构建和链接过程。在编译行添加`-I<installpath>/include`、在链接行添加`-L<installpath>/lib -lcudnn`。

   

6. **安装tensorflow**

   ```bash
   sudo pip install tensorflow-gpu
   ```

   验证安装，运行：

   ```python
   import tensorflow as tf
   hello = tf.constant('Hello, TensorFlow!')
   sess = tf.Session()
   print sess.run(hello)
   ```

   


##### 1.删除一些无用软件

包括亚马逊链接以及其他一些无需的软件

```bash
sudo apt-get remove unity-webapps-common # 卸载Amazon
sudo apt-get remove libreoffice-common   # 卸载LibreOffice
```



##### 2.安装谷歌拼音



##### 3.修改pip源

在home目录创建.pip目录，在其中创建pip.conf文件

```bash
cd ~
mkdir .pip
cd .pip
vim pip.conf
```

之后编辑pip.conf的内容：

```configure
# pip安装需使用https加密，在此需添加trusted-host
[global]
trusted-host = mirrirs.aliyun.com
index-url = https://mirrors.aliyun.com/pypi/simple
```

或者也可以手动指定安装源：

```bash
pip -i https://pypi.douban.com
```



##### 4.设置github账户

验证已有的SSH密钥

```bash
ls -al ~/.ssh
```

若无此文件的存在，则需创建新SSH密钥

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
Generating public/private rsa key pair.
Enter a file in which to save the key (/home/you/.ssh/id_rsa): [Press enter]
Enter passphrase (empty for no passphrase): [Type a passphrase]
Enter same passphrase again: [Type passphrase again]
```

将SSH密钥添加到ssh-代理中

```bash
eval "$(ssh-agent -s)"
Agent pid 59566
ssh-add ~/.ssh/id_rsa
```

将新SSH密钥添加到Github账户

```bash
gedit ~/.ssh/id_rsa.pub
```

复制SSH密钥，在自己的github账户依次点击"setting" -> "SSH and GPG keys" -> "New SSH key"，在"title"域填上一些描述，在"key"域粘贴复制的密钥即可。



##### 附录

pipy国内的镜像有：

> https://pypi.douban.com/ 豆瓣
>
> https://pypi.hustunique.com/ 华中理工大学
>
> http://pypi.sdutlinux.org/ 山东理工大学
>
> https://pypi.mirrors.ustc.edu.cn/ 中国科学技术大学
>
> https://mirrors.aliyun.com/pypi/simple/ 阿里云
>
> https://pypi.tuna.tsinghua.edu.cn/simple/ 清华大学