#### 概念

##### Conda目录结构

**ROOT_DIR**：Anaconda安装的目录；

**/pkgs**：也表示为*PKG_DIR*，包含压缩包、准备好在conda环境中被链接的，每个包属于一对应其名的子目录；

**/envs**：其他创建的conda环境的系统位置；

**/bin, /include, /lib, /share**组成了Anaconda的默认环境，其他conda环境通常包含同样的子目录为默认环境。

##### Conda环境

一个包含特定conda包集合的目录；可以通过分享`enviroment.yaml`文件来分享环境。

##### conda包

一个包含系统层库、python或其他模块可执行程序和其他成分的压缩tarball文件；可以从远程通道下载，conda命令搜索默认通道集合，包可以自动从https://repo.continuum.io/pkgs/下载或更新；可以修改默认下载的通道，也可以维护一个私有或内部通道；通过`conda install [packagename]`安装conda包；一个conda包包含一个指向在`info/`下包含元数据的压缩文件和直接安装进`install`前缀的文件集的链接；安装时除`/info`目录下，其他文件被提取到`install`前缀。



#### 开始使用conda

##### 管理conda

验证conda：`conda --version`；

更新conda：`conda update conda`。

##### 管理环境

当开始使用conda时，已经有了名为`base`的默认环境。可以创建单独的环境使程序隔离：

1. 创建新环境并安装其包：`conda create --name [enviromentname] [packagename]`；
2. 使用或激活环境：`source activate [enviromentname]`；
3. 产看环境列表：`conda info --envs`，标*的为当前激活的环境；
4. 当当前的环境变为默认：`source deactivate`。

##### 管理python

当需要使用不同的python时，可以在创建环境时制定python版本：

```bash
onda create --name snakes python=3.5
```

##### 管理包

1. 寻找已安装的包，先激活相搜索的环境；
2. 查证未安装的包是否可得：`conda search [packagename]`；
3. 安装到当前环境：`conda install [packagename]`；
4. 查看新安装的包是否已在环境中：`conda list`。



#### 安装

##### 在已有python的系统安装anaconda。

正常安装，让安装器添加python的conda安装到PATH环境变量，无需设置PYTHONPATH环境变量。

##### 卸载miniconda

1. 删除整个miniconda目录：`rm -rf ~/miniconda`；
2. 将PATH中的相关内容编辑掉；
3. 删除下面的隐藏文件：`rm -rf ~/.condarc ~/.conda ~/.continum`

##### 卸载anaconda

1. 安装Anaconda-Clean包：`conda install anaconda-clean`；

2. 同样的窗口下运行：`anaconda-clean --yes`；

   Anaconda-Clean会在`.anaconda_backup`创建所有可能被删除文件的备份，也会使AnacondaProject目录中的数据文件不受影响。

3. 删除Anaconda目录：`rm -rf ~/anaconda3`；

4. 将`.bashrc`中的`export PATH="/Users/jsmith/anaconda3/bin:$PATH"`行用实际的路径代替。