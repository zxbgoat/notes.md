##### 目录跳转

**cd**：`cd -`的作用是可以回到前一个目录，`-`在此处等同于`$OLDPWD`变量，`$OLDPWD`变量就是bash所记录的前一个目录，用`cd -`和`cd $OLDPWD`都可以在最近所操作的两个目录之间进行切换。

**pushd**：切换到作为参数的目录，并将原目录和当前目录压入到一个虚拟的堆栈中；若无参数，则会回到前一个目录，并交换堆栈中最近两个目录；用无参`pushd`在最近两个目录间切换时，当前目录总是位于堆栈顶端；用 `pushd +n`可在多个目录之间切换，`n`为一数字，表示切换到堆栈中第n个目录，并把此目录以堆栈循环方式推到顶部。堆栈从第0个开始数起；用`-n`参数可以只影响堆栈而不切换目录。

**popd**：弹出堆栈中最近的目录；无参数时会把堆栈顶端的目录从堆栈中删除，并切换到新的顶端目录；`popd +n`表示把堆栈中第n个目录从堆栈中删除；位于堆栈顶部的目录是当前目录，不能被pop出去。

**dirs**：列出当前堆栈中保存的目录列表；`-p`参数以每行一个目录的形式显示堆栈中目录列表； `-v`参数在目录前加上编号，此时无`-p`也可以每行一个目录的形式显示；最近压入堆栈的目录位于顶端；0表示当前目录，1是下个要切换的栈顶目录；`-c`参数清空目录堆栈。



##### 用apt管理软件

`apt-cache search package` 搜索包

`apt-cache show package` 获取包的相关信息

`sudo apt-get install --reinstall package` 重新安装包

`sudo apt-get -f/—fix-missing install` 修复安装

`sudo apt-get remove --purge package` 删除包，包括配置文件等

`sudo apt-get autoremove --purge` 删除包及其依赖的软件包+配置文件等

`sudo apt-get update` 更新源

`sudo apt-get upgrade` 更新已安装的包

`sudo apt-get dist-upgrade` 升级系统

`apt-cache depends package` 了解使用依赖

`apt-cache rdepends package` 了解某个具体的依赖

`sudo apt-get build-dep package` 安装相关编译环境

`apt-get source package` 下载该包的源码

`sudo apt-get clean` 清理下载文件的存档

`sudo apt-get autoclean` 只清理过时的包

`sudo apt-get check` 检查是否有损坏的依赖



##### 用dpkg管理软件

`-i` 安装一个debian包裹文件，如手动下载的文件；

`-c` 列出包裹文件的内容；

`-I` 提取包裹信息；

`-r` 移除已安装包裹；

`-P` 完全清楚已安装包裹，包括所有配置文件；

`-L` 列出安装的所有文件清单；

`-s` 显示已安装包裹信息；



##### 用scp远程传输文件

下载文件：`scp username@192.168.0.101:/remote/location/filename /local/location/ `

上传文件：`scp /local/location/filename username@192.168.0.101:/remote/location`

下载目录：`scp -r username@192.168.0.101:/remote/location/dirname /local/location`

上传目录：`scp -r local/location/dirname username@192.168.0.101:/remote/location`



##### 安装ftp服务

首先下载安装vsftpd：`sudo apt-get install vsftpd`

然后修改/etc/vsftpd.conf文件，修改如下几行：

```bash
anonymous_enable=YES    #设置匿名可登录
local_enable=YES        #本地用户允许登录
write_enable=YES        #用户是否有写的权限
anon_upload_enable=YES   #允许匿名用户上传
anon_mkdir_write_enable=YES   #允许匿名用户创建目录文件
```

保存文件，重启vsftpd服务器：`sudo service vsftpd restart`。

匿名用户默认访问的是`/srv/ftp`文件夹，在其中新建两个文件夹，一个是`upload`用来上传，权限设置为可读可写，一个是`download`用来下载，权限设置为可读不可写：

```bash
sudo chmod -R 777 /srv/ftp/upload
sudo chmod -R 755 /srv/ftp/download
```



##### 远程登录Jupyter笔记

首先需要配置jupyter notebook，登录远程服务器后，使用如下命令生成配置文件：

```bash
jupyter notebook --generate-config
```

然后打开文件修改其内容：

```bash
vim ~/.jupyter/jupyter_notebook_config.py
```

重要修改两处：

```python
c.NotebookApp.ip='*' #不限制ip访问
c.NotebookApp.password = u'hash_value'
c.NotebookApp.open_browser = False #禁用服务器端浏览器
c.NotebookApp.port =8888 # 使用默认8888不用改，自行指定一个端口修改
```

上面的`hash_value`由用户给定密码生成，可以使用ipython中的命令获得：

```python
from notebook.auth import passwd
passwd()
"""
这里会要求用户输入密码并确认，生成的hash值要填写到上面
"""
```

之后，在远程服务器上启动`jupyter notebook`，然后在本地机器上访问`remote_ip:8888`（默认端口为8888，也可以在配置文件中修改），输入密码即可远程访问jupyter笔记。

类似Vim编辑器，Jupyter Notebook有两种键盘输入模式：

- 编辑模式，在单元中键入代码或文本，这时的单元框线是绿色的。通过`Esc`可以切换至命令模式。
- 命令模式，键盘输入运行命令，这时的单元框线是蓝色的。通过`Enter`可以切换至编辑模式。

常用的快捷键有：

- `h`：弹出快捷键列表
- `m`：将该Cell转换为Markdown输入状态
- `y`：将该Cell转换为代码输入状态
- `Shift+Enter`：执行该Cell并将焦点置于下一个Cell（如果没有，将新建）
- `Ctrl+Enter`：执行该Cell



##### 包位置

在ubuntu中使用apt-get安装包时，会先从 `/etc/apt/sourcs.list`读取图个一个列表，从而获知从何处去寻找包，包括从本地文件系统或使用http或ftp在网上寻找。可在`/etc/apt/sources.list.d`目录中添加更多的来源。`apt-get`及类似工具使用本地数据库来确定安装的包，可针对可用级别来检查已安装的级别。为此，从`/etc/apt/sources.list`中列出的来源获取可用级别，并存储在本地系统上。可以使用命令 `apt-get update` 将本地数据库信息与`/etc/apt/sources.list`指定来源同步。在安装或更新任何包之前、以及修改`/etc/apt/sources.list`或向`/etc/apt/sources.list.d`添加文件后，都应该执行这样的操作。

