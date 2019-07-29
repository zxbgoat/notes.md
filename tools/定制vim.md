#### 1 源码构建vim

首先，安装依赖库：

```bash
sudo apt install libncurses5-dev libgnome2-dev libgnomeui-dev \
    libgtk2.0-dev libatk1.0-dev libbonoboui2-dev \
    libcairo2-dev libx11-dev libxpm-dev libxt-dev python-dev \
    python3-dev ruby-dev lua5.1 liblua5.1-dev libperl-dev git
```

若已有vim，则卸载

```bash
sudo apt remove vim vim-runtime gvim
```

然后获取源码、编译安装：

```bash
cd ~/Workspace
git clone git@github.com:vim/vim.git
cd vim
./configure --with-features=huge \
            --enable-multibyte \
            --enable-rubyinterp=yes \
            --enable-pythoninterp=yes \
            --with-python-config-dir=/usr/lib/python2.7/config \
            --enable-python3interp=yes \
            --with-python3-config-dir=/usr/lib/python3.5/config \
            --enable-perlinterp=yes \
            --enable-luainterp=yes \
            --enable-gui=gtk2 \
            --enable-cscope \
            --prefix=/usr/local
make VIMRUNTIMEDIR=/usr/local/share/vim/vim80
```

其中，选项`--enable-pythoninterp`表示支持python编写的插件，后面关于ruby、perl、lua的选项类似。`--enable-gui=gtk2`表示生成GNOME2风格的gvim，`--enable-cscope`支持cscope，`--with-python-config-dir=/usr/lib/python2.7/config/`指定python路径。注意，在ubuntu16.04中，python2和python3不能同时选择，因此上面的语句中选择一个即可。

若希望将来能方便地删除vim，使用`checkinstall`，除了安装vim，checkinstall还会生成一个vim\*.deb文件，若要删除，直接`dpkg -r vim*.deb`即可。先安装`checkinstall`

```bash
sudo apt install checkinstall
```

然后执行

```bash
sudo checkinstall
```

否则直接使用`make`来安装：

```bash
sudo make install
```

使用`update-alternative`设置vim为默认编辑器：

```bash
sudo update-alternatives --install /usr/bin/editor editor /usr/local/bin/vim 1
sudo update-alternatives --set editor /usr/local/bin/vim
sudo update-alternatives --install /usr/bin/vi vi /usr/local/bin/vim 1
sudo update-alternatives --set vi /usr/local/bin/vim
```

最后验证`vim --version`的信息；在vim中执行`:echo has('python')`可检测支持python与否，输出1为支持。



#### 2 .vimrc文件

写更为复杂的Vimscript时 ~/.vimrc是控制vim行为的配置文件，不论窗口、字体，还是操作、快捷键、插件均可通过该文件配置。

vim自带很多快捷键，加上各类插件的快捷键，大量快捷键出现在单层空间中难免引起冲突，为缓解该问题，引入了前缀键 `<leader>`，这样键r可以配置成`r`、`<leader>r`、`<leader><leader>r`等多个快捷键。设置一个自己方便的前缀键即可，这里使用分号键，就在右手小指处：

```vimscript
" 定义快捷键的前缀，即<Leader>
let mapleader=";"
```

开启文件类型侦测、允许基于不同语言加载不同插件：

```vimscript
" 开启文件类型侦测
filetype on
" 根据侦测到的不同类型加载对应的插件
filetype plugin on
```

快捷键设定原则：不同快捷键尽量不要有同序的相同字符。如`<leader>e`、`<leader>eb`执行不同操作，执行前者时键入`<leader>e`后vim会等待片刻以确认，这降低了响应速度。下面设置一些快捷键：

```vimscript
" 定义快捷键到行首和行尾
nmap LB 0
nmap LE $
" 设置快捷键将选中文本块复制至系统剪贴板
vnoremap <Leader>y "+y
" 设置快捷键将系统剪贴板内容粘贴至 vim
nmap <Leader>p "+p
" 定义快捷键关闭当前分割窗口
nmap <Leader>q :q<CR>
" 定义快捷键保存当前窗口内容
nmap <Leader>w :w<CR>
" 定义快捷键保存所有窗口内容并退出 vim
nmap <Leader>WQ :wa<CR>:q<CR>
" 不做任何保存，直接退出 vim
nmap <Leader>Q :qa!<CR>
" 依次遍历子窗口
nnoremap nw <C-W><C-W>
" 跳转至右方的窗口
nnoremap <Leader>lw <C-W>l
" 跳转至左方的窗口
nnoremap <Leader>hw <C-W>h
" 跳转至上方的子窗口
nnoremap <Leader>kw <C-W>k
" 跳转至下方的子窗口
nnoremap <Leader>jw <C-W>j
" 定义快捷键在结对符之间跳转
nmap <Leader>M %
" 让配置变更立即生效
autocmd BufWritePost $MYVIMRC source $MYVIMRC
```

搜索、命令补全等设置：

```vimscript
" 开启实时搜索功能
set incsearch
" 搜索时大小写不敏感
set ignorecase
" 关闭兼容模式
set nocompatible
" vim 自身命令行模式智能补全
set wildmenu
```



#### 3 插件管理

`.vim/`目录存放所有插件，插件主要用vimscript编写，也支持perl、python、lua、ruby等脚本语言，但需在源码编译时增加`--enable-pythoninterp`等对应选项。插件有\*.vim和\*.vba两类。前者是plg.vim脚本与plg.txt帮助的打包，解包后分别拷到`~/.vim/plugin/`和`~/.vim/doc/`。重启后写更为复杂的Vimscript时插件即生效，执行`:helptags ~/.vim/doc/`帮助生效，通过`:h plg`查看插件帮助信息。\*.vba格式插件安装相对便捷，只需执行shell命令：

```bash
vim plugin.vba
:so %
:q
```

但这些都相对繁琐也不便于插件的卸载、升级，现在主要用vundle管理插件。vundle会接管`.vim`目录，因此先清空该目录，再通过命令安装vundle：

```bash
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
```

然后在.vimrc顶部增加配置信息：

```vimscript
" vundle 环境设置
filetype off
set rtp+=~/.vim/bundle/Vundle.vim
" vundle 管理的插件列表必须位于 vundle#begin() 和 vundle#end() 之间
call vundle#begin()
Plugin 'VundleVim/Vundle.vim'
Plugin 'altercation/vim-colors-solarized'
Plugin 'tomasr/molokai'
Plugin 'vim-scripts/phd'
Plugin 'Lokaltog/vim-powerline'
Plugin 'octol/vim-cpp-enhanced-highlight'
Plugin 'nathanaelkane/vim-indent-guides'
Plugin 'derekwyatt/vim-fswitch'
Plugin 'kshenoy/vim-signature'
Plugin 'vim-scripts/BOOKMARKS--Mark-and-Highlight-Full-Lines'
Plugin 'majutsushi/tagbar'
Plugin 'vim-scripts/indexer.tar.gz'
Plugin 'vim-scripts/DfrankUtil'
Plugin 'vim-scripts/vimprj'
Plugin 'dyng/ctrlsf.vim'
Plugin 'terryma/vim-multiple-cursors'
Plugin 'scrooloose/nerdcommenter'
Plugin 'vim-scripts/DrawIt'
Plugin 'SirVer/ultisnips'
Plugin 'Valloric/YouCompleteMe'
Plugin 'derekwyatt/vim-protodef'
Plugin 'scrooloose/nerdtree'
Plugin 'fholgado/minibufexpl.vim'
Plugin 'gcmt/wildfire.vim'
Plugin 'sjl/gundo.vim'
Plugin 'Lokaltog/vim-easymotion'
Plugin 'suan/vim-instant-markdown'
Plugin 'lilydjwg/fcitx.vim'
" 插件列表结束
call vundle#end()
filetype plugin indent on
```

其中每项对应一个插件，后面增加新插件时追加至该列表即可。vundle支持源码托管在github上的插件，例如在在.vimrc中配置信息为dyng/ctrlsf的插件，vundle会构造出https://github.com/dyng/ctrlsf.vim.git下载地址，然后借助git工具下载安装。进入vim执行下面命令即可安装插件：

```vimscript
:PluginInstall
```

要卸载插件，先在.vimrc中注释或者删除对应插件配置信息，然后在vim中执行：

```vimscript
:PluginClean
```

插件更新频率较高，差不多每隔一个月即应查看插件是否有新版本，批量更新只需执行：

```vimscript
:PluginUpdates
```



#### 4 外观设置

##### 主题风格

vim内置了10多种配色方案字符模式下，调整配置信息重启vim查看写更为复杂的Vimscript时效果（csExplorer 插件可不用重启即可查看效果）。不满意可以查看[python](https://metalelf0.github.io/VimColorSchemeTest-Ruby/python.html)版或[ruby](https://metalelf0.github.io/VimColorSchemeTest-Ruby/ruby.html)选择。推介素雅solarized、多彩molokai和复古phd三种。

```vimscript
" 配色方案
set background=dark
colorscheme solarized
"colorscheme molokai
"colorscheme phd
```

而不同主题又有暗/亮色系，通过`set background=light`定义。

##### 营造专注氛围

下面的设置可以营造专注的氛围：

```vimscript
" 禁止光标闪烁
set gcr=a:block-blinkon0
" 禁止显示滚动条
set guioptions-=l
set guioptions-=L
set guioptions-=r
set guioptions-=R
" 禁止显示菜单和工具条
set guioptions-=m
set guioptions-=T
```

vim需借助第三方工具 wmctrl实现全屏，安装wmctrl：

```bash
sudo apt-get install wmctrl
```

再在 .vimrc 中增加如下信息：

```vimscript
" 将外部命令 wmctrl 控制窗口最大化的命令行参数封装成一个 vim 的函数
fun! ToggleFullscreen()
	call system("wmctrl -ir " . v:windowid . " -b toggle,fullscreen")
endf
" 全屏开/关快捷键
map <silent> <F11> :call ToggleFullscreen()<CR>
" 启动 vim 时自动全屏
autocmd VimEnter * call ToggleFullscreen()
```

上面是一段简单的 vimscript 脚本，外部命令 wmctrl 及其命令行参数控制将指定窗口 windowid（即，vim）全屏，绑定快捷键 F11 实现全屏/窗口模式切换，最后配置启动时自动全屏。

##### 添加辅助信息

设置一些有用信息：

```vimscript
" 总是显示状态栏
set laststatus=2
" 显示光标当前位置
set ruler
" 开启行号显示
set number
" 高亮显示当前行/列
set cursorline
set cursorcolumn
" 高亮显示搜索结果
set hlsearch
" 禁止折行
set nowrap
```

##### 其他美化

设置gvim字体，由于字体名存在空格，需要用转义符“\”进行转义；最后的11.5指定字体大小：

```vimscript
" 设置 gvim 显示字体
set guifont=YaHei\ Consolas\ Hybrid\ 11.5
```

前面介绍的主题风格对状态栏不起作用，需要借助插件Powerline美化状态栏，在 .vimrc 中设定状态栏主题风格：

```vimscript
" 设置状态栏主题风格
let g:Powerline_colorscheme='solarized256'
```



#### 5 代码加强

##### 语法高亮：

```vimscript
" 开启语法高亮功能
syntax enable
" 允许用指定语法高亮配色方案替换默认方案
syntax on
```

vim对C++语法高亮支持不够好，插件vim-cpp-enhanced-highlight可对其进行增强，通过`.vim/bundle/vim-cpp-enhanced-highlight/after/syntax/cpp.vim`控制高亮关键字及规则，所以若发现某个STL容器类型未高亮，将该类型追加进 cpp.vim 即可。如`initializer_list`默认并不会高亮，添加

```vimscript
syntax keyword cppSTLtype initializer_list
```

##### 代码缩进

设置缩进的tab：

```vimscript
" 自适应不同语言的智能缩进
filetype indent on
" 将制表符扩展为空格
set expandtab
" 设置编辑时制表符占用空格数
set tabstop=4
" 设置格式化时制表符占用空格数
set shiftwidth=4
" 让 vim 把连续数量的空格视为一个制表符
set softtabstop=4
```

注意expandtab、tabstop 与 shiftwidth、softtabstop、retab：

- expandtab，把制表符转换为多个空格，具体空格数量参考 tabstop 和 shiftwidth 变量；
- tabstop 指定插入模式下输入一个制表符的空格数，linux 内核编码规范建议是 8；shiftwidth指定缩进格式化时制表符的空格数。缩进格式化指的是通过vim命令自动对源码进行缩进处理缩进格式化，需要先选中指定行，键入`=`对该行进行智能缩进或按需键入多次`<`或`>`手工缩进；
- softtabstop处理连续多个空格，expandtab已把制表符转换为空格，当删除制表符时需连续删除多个空格，该设置就是告诉vim把连续数量的空格视为一个制表符。通常应将其与tabstop、shiftwidth、softtabstop 三个变量设置为相同值；
- 可以手工执行 vim 的 retab 命令，让 vim 按上述规则重新处理制表符与空格关系。

插件Indent Guides能将相同缩进的代码可视化，安装好该插件后，增加如下配置信息：

```vimscript
" 随 vim 自启动
let g:indent_guides_enable_on_vim_startup=1
" 从第二层开始可视化显示缩进
let g:indent_guides_start_level=2
" 色块宽度
let g:indent_guides_guide_size=1
" 快捷键 i 开/关缩进可视化
:nmap <silent> <Leader>i <Plug>IndentGuidesToggle
```

Indent Guides通过识别制表符来绘制缩进连接线，换行-空格-退格可消除断节。

##### 代码折叠

vim支持多种折叠：手动建立（manual）、基于缩进进行（indent）、基于语法进行（syntax）、未更改文本构成（diff）等等，按需选用。操作为：za，打开或关闭当前折叠；zM，关闭所有折叠；zR，打开所有折叠。增加如下配置信息：

```vimscript
" 基于缩进或语法进行代码折叠
"set foldmethod=diff
"set foldmethod=indent
set foldmethod=syntax
" 启动 vim 时关闭折叠代码
set nofoldenable
```

##### 接口与实现的快速切换

C++常有在接口文件（MyClass.h）和实现文件（MyClass.cpp）中来回切换的操作。在接口文件时，[vim-fswitch](https://github.com/derekwyatt/vim-fswitch)插件能自动找到对应实现文件，键入快捷键后便会在新buffer中打开对应实现文件。安装后增加配置信息：

```vimscript
" *.cpp 和 *.h 间切换
nmap <silent> <Leader>sw :FSHere<cr>
```

##### 代码收藏

分析源码时常在不同代码间来回跳转，可以“收藏”分散在不同处的代码行以便查看时能快速跳转，这时可以使用vim的书签（mark）功能，在要收藏的行键入`mm`即可添加书签，插件[vim-signature](https://github.com/kshenoy/vim-signature)在书签行前添加字符的形式可视化，但要求vim具备signs特性，检查：

```
:echo has('signs')
```

vim有独立书签和分类书签两类；独立书签名字只能由字母(a-zA-Z)组成，长度不超过2个字符，并且同个文件不同书签名不恩那个含有相同字符；分类书签名字只能由可打印字符(!@#$%^&*())组成，长度只能为1，同个文件中设成同名书签的行在逻辑上即归为同类书签。两种书签完全分布在各自不同的空间中，它俩的任何操作都是互不相同的，都有各自的使用场景。vim-signature 快捷键如下：

```vimscript
let g:SignatureMap = {
        \ 'Leader'             :  "m",
        \ 'PlaceNextMark'      :  "m,",
        \ 'ToggleMarkAtLine'   :  "m.",
        \ 'PurgeMarksAtLine'   :  "m-",
        \ 'DeleteMark'         :  "dm",
        \ 'PurgeMarks'         :  "mda",
        \ 'PurgeMarkers'       :  "m<BS>",
        \ 'GotoNextLineAlpha'  :  "']",
        \ 'GotoPrevLineAlpha'  :  "'[",
        \ 'GotoNextSpotAlpha'  :  "`]",
        \ 'GotoPrevSpotAlpha'  :  "`[",
        \ 'GotoNextLineByPos'  :  "]'",
        \ 'GotoPrevLineByPos'  :  "['",
        \ 'GotoNextSpotByPos'  :  "mn",
        \ 'GotoPrevSpotByPos'  :  "mp",
        \ 'GotoNextMarker'     :  "[+",
        \ 'GotoPrevMarker'     :  "[-",
        \ 'GotoNextMarkerAny'  :  "]=",
        \ 'GotoPrevMarkerAny'  :  "[=",
        \ 'ListLocalMarks'     :  "ms",
        \ 'ListLocalMarkers'   :  "m?"
        \ }
```

常用的操作也就如下几类：

- 书签设定：
  - mx，设定/取消当前行名为x的标签；
  - m,，自动帮选定下一个可用独立书签名；
  - mda，删除当前文件中所有独立书签。
- 书签罗列：
  - m?，罗列出当前文件中所有书签，选中后回车可直接跳转；
- 书签跳转：
  - mn，按行号前后顺序，跳转至下个独立书签；
  - mp，按行号前后顺序，跳转至前个独立书签。
  - 书签跳转方式还有很多，比如基于书签名字母顺序、分类书签同类、分类书签不同类间等等。

若觉得只有行首符号来表示不够醒目，插件[BOOKMARKS--Mark-and-Highlight-Full-Lines](https://github.com/vim-scripts/BOOKMARKS--Mark-and-Highlight-Full-Lines)可以让书签行高亮。



#### 6 标签系统

要实现标识符列表、定义跳转、声明提示、实时诊断、代码补全等系列功能，需要vim能很好地理解代码（不论是vim自身还是借助插件或第三方工具），有两种主流方式帮vim理解代码：标签系统和语义系统，两者优劣简单来说，标签系统配置简单，语义系统效果精准，后者是趋势，若同个功能插件两种实现都有，优选语义。

代码中的类、结构、类成员、函数、对象、宏等统称为标识符；每个标识符的定义、在文件中的行位置、所在文件的路径等信息就是标签（tag）；[Exuberant Ctags](http://ctags.sourceforge.net/)，简称ctags是一款经典的用于生成代码标签信息的工具，目前已支持41种语言，查看支持语言和某种语言支持的全量运行：

```vimscript
ctags --list-languages
ctags --list-kinds=python
```

后者输出为：

```bash
c  classes
f  functions
m  class members
v  variables
i  imports [off]
```

其中标为off的类型默认不会生成标签，需显式加上。在cpp主函数目录所在文件中运行：

```bash
ctags -R --c++-kinds=+c+f+m+v+i --fields=+liaS --extra=+q --language-force=c++
```

就会生成一个名为tag的标签文件。其中!开头的几行是ctags生成的软件信息可以忽略，下面每个标签项至少有如下字段（命令行参数不同标签项的字段数不同）：标识符名、标识符所在的文件名（也是该文件的相对路径）、标识符所在行的内容、标识符类型（如，l 表示局部对象），另外，若是函数，则有函数签名字段，若是成员函数，则有访问属型字段等等。

##### 基于标签的标识符列表

在阅读代码时，经常需要分析指定函数的实现细节，希望把当前代码文件中提取出的所有标识符放在一个侧边子窗口中，并且能能按语法规则将标识符进行归类，[tagbar](https://github.com/majutsushi/tagbar)是一款基于标签的标识符列表插件，它自动周期性调用 ctags 获取标签信息（仅保留在内存）。安装完 tagbar 后，在 .vimrc 中增加如下信息：

```vimscript
" 设置 tagbar 子窗口的位置出现在主编辑区的左边 
let tagbar_left=1 
" 设置显示／隐藏标签列表子窗口的快捷键。速记：identifier list by tag
nnoremap <Leader>ilt :TagbarToggle<CR> 
" 设置标签子窗口的宽度 
let tagbar_width=32 
" tagbar 子窗口中不显示冗余帮助信息 
let g:tagbar_compact=1
" 设置 ctags 对哪些代码标识符生成标签
let g:tagbar_type_cpp = {
    \ 'kinds' : [
         \ 'c:classes:0:1',
         \ 'd:macros:0:1',
         \ 'e:enumerators:0:0', 
         \ 'f:functions:0:1',
         \ 'g:enumeration:0:1',
         \ 'l:local:0:1',
         \ 'm:members:0:1',
         \ 'n:namespaces:0:1',
         \ 'p:functions_prototypes:0:1',从
         \ 's:structs:0:1',
         \ 't:typedefs:0:1',
         \ 'u:unions:0:1',
         \ 'v:global:0:1',
         \ 'x:external:0:1'
     \ ],
     \ 'sro'        : '::',
     \ 'kind2scope' : {
         \ 'g' : 'enum',
         \ 'n' : 'namespace',
         \ 'c' : 'class',
         \ 's' : 'struct',
         \ 'u' : 'union'
     \ },
     \ 'scope2kind' : {
         \ 'enum'      : 'g',
         \ 'namespace' : 'n',
         \ 'class'     : 'c',
         \ 'struct'    : 's',
         \ 'union'     : 'u'
     \ }
\ }
```

ctags默认不提取局部对象、函数声明、外部对象等类型的标签，必须让tagbar告诉ctags改变默认参数，这便是变量tagbar_type_cpp的作用，因此配置信息中这些标签显式加进其kinds域中。具体格式为

```vimscript
{short}:{long}[:{fold}[:{stl}]]
```

用于描述不同类型的标识符。其中short将作为ctags中--c++-kinds命令行选项的参数，类似：

```vimscript
--c++-kinds=+p+l+x+c+d+e+f+g+m+n+s+t+u+v
```

long作为简要描述展示在vim的tagbar子窗口；fold表示这种类型标识符是否折叠显示；stl指定是否在vim状态栏显示附加信息。重启 vim 后，打开一个 C/C++ 源码文件，键入`;ilt`即可在左侧tagbar窗口看到标签列表。

tagbar显示有几个特点：

- 按作用域归类不同标签，按名字空间、类归类，内部有声明、定义； 
- 显示标签类型，名字空间、类、函数等等；
- 显示完整函数原型；
- 图形化显示公有成员（+）、私有成员（-）、保护成员（#）。

在标识符列表中选中对应标识符后回车即可跳至源码中对应位置；在源码中停顿，tagbar将高亮对应标识符；每次保存文件或者切换文件tagbar自动调用ctags更新标签数据库；tagbar有两种排序方式：按标签名字母先后顺序和按标签在源码中出现先后顺序，在前面.vimrc中选用了后者，键入 s 切换不同不同排序方式。

##### 基于标签的跳转

在分析某个开源项目源码，在 main.cpp 中遇到调用函数 func()，想要查看它如何实现，像真正函数调用一样：光标选中调用处的 func() -> 键入某个快捷键自动转换到 func() 实现处 -> 键入某个键又回到 func() 调用处，这就是所谓的定义跳转。基本上，vim 世界存在两类导航：基于标签的跳转和基于语义的跳转。

通过ctags命令可生成的标签文件，可用于实现声明/定义跳转，需先让vim知晓标签文件的路径。使用vim打开要分析的cpp文件，执行引入标签文件tags命令：

```vimscript
:set tags+=/path/to/tags
```

这样vim就能根据标签跳转至标签定义处，这时便可以体验初级的声明/定义跳转功能。把光标移到某个名称上，键入快捷键`g]`，vim将罗列出名称的所有标签候选列表，按需选择键入编号即可跳转进入。

若不想输入数字来选择，但能接受按键遍历所有，另一按键逆序遍历。命令`:tnext`和`:tprevious`分别先后和向前遍历匹配标签，可定义两个快捷键：

```vimscript
" 正向遍历同名标签
nmap <Leader>tn :tnext<CR>
" 反向遍历同名标签
nmap <Leader>tp :tprevious<CR>
```

但vim中有个标签栈机制，`:tnext`和`:tprevious`只能遍历已经压入标签栈内的标签，因此在遍历前需通过快捷键`ctrl-]`将光标所在单词匹配的所有标签压入标签栈中，再键入`tn` 或`tp` 才能遍历。

要返回先前位置需先键入`ctrl-t`返回上次光标停留行，`ctrl-o`返回上个标签，再次进入则键入`ctrl-i`。

插件[indexer](https://github.com/vim-scripts/indexer.tar.gz)周期性针对这个工程自动生成标签文件，并通知 vim 引人该标签文件。indexer依赖[DfrankUtil](https://github.com/vim-scripts/DfrankUtil)和[vimprj](https://github.com/vim-scripts/vimprj)两个插件，需一并安装。在 .vimrc 中增加：

```VimL
" 设置插件 indexer 调用 ctags 的参数
" 默认 --c++-kinds=+p+l，重新设置为 --c++-kinds=+p+l+x+c+d+e+f+g+m+n+s+t+u+v
" 默认 --fields=+iaS 不满足 YCM 要求，需改为 --fields=+iaSl
let g:indexer_ctagsCommandLineOptions="--c++-kinds=+p+l+x+c+d+e+f+g+m+n+s+t+u+v --fields=+iaSl --extra=+q"
```

另外，indexer 还有自己的配置文件可设定各个工程的根目录路径，其位于 ~/.indexer_files，内容可为：

```viml
--------------- ~/.indexer_files ---------------  
[foo] 
/data/workplace/foo/src/
[bar] 
/data/workplace/bar/src/
```

上例设定两个工程的根目录，方括号内为工程名，路径为工程代码目录。此时打开以上目录任何代码文件indexer便对整个目录创建标签文件；若代码文件有更新，则在文件保存时indexer将自动更新标签文件，生成的标签文件以工程名位于 ~/.indexer_files_tags/，并自动引入进 vim 中。



##### 语义系统

ctags这类标签系统能一定程度助力vim理解代码，但随着语言的不断迭代已经显得有心无力，因此最理想的是让编译器在语义上帮助vim理解代码。GCC和clang两大主流 C/C++ 编译器，而作为语义系统的支撑工具则选择后者，除对新标准支持及时、错误诊断信息清晰外，在高内聚、低耦合方面也很好，各类插件可调用 libclang 获取非常完整的代码分析结果。先安装clang及相应的标准库：

````bash
cd ~/Workspace
# Checkout LLVM
svn co http://llvm.org/svn/llvm-project/llvm/trunk llvm
# Checkout Clang
cd llvm/tools
svn co http://llvm.org/svn/llvm-project/cfe/trunk clang
# Checkout extra Clang Tools
cd clang/tools
svn co http://llvm.org/svn/llvm-project/clang-tools-extra/trunk extra
# Checkout LLD linker
cd ../..
svn co http://llvm.org/svn/llvm-project/lld/trunk lld
# Checkout Polly Loop Optimizer
svn co http://llvm.org/svn/llvm-project/polly/trunk polly
# Checkout Compiler-RT
cd ../projects
svn co http://llvm.org/svn/llvm-project/compiler-rt/trunk compiler-rt
# Checkout libcxx and libcxxabi
svn co http://llvm.org/svn/llvm-project/libcxx/trunk libcxx
svn co http://llvm.org/svn/llvm-project/libcxxabi/trunk libcxxabi
cd ../..
````

然后编译和安装源码：

```bash
mkdir build
cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ../llvm
make
sudo make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

完成后验证：`clang --version`。之后的编译即可通过：

```bash
clang++ -std=c++11 -stdlib=libc++ test.cpp
```



##### 基于语义的跳转

通过YCM插件和C++编译器clang将获得无与伦比的代码导航和代码补全体验。目前只给出快捷键设置，看完“基于语义的智能补全”后返回此处重新查阅。增加如下快捷键到.vimrc中：

```vimscript
nnoremap <leader>jc :YcmCompleter GoToDeclaration<CR>
" 只能是 #include 或已打开的文件
nnoremap <leader>jd :YcmCompleter GoToDefinition<CR>
```



##### 内容查找

内容查找，你第一反应会想到 grep 和 ack 两个工具，没错，它俩强大的正则处理能力无需质疑，如果有插件能在 vim 中集成两个工具之一，那么任何查找任务均可轻松搞定，为此，出现了 grep.vim（<https://github.com/yegappan/grep> ）和 ack.vim（<https://github.com/mileszs/ack.vim> ）两个插件，通过它们，你可以在 vim 中自在地使用高度整合的 grep 或 ack 两个外部命令，就像 vim 的内部命令一样：查找时，把光标定位到待查找关键字上后，通过快捷键立即查找该关键字，查询结果通过列表形式将关键字所在行罗列出来，选择后就能跳转到对应位置。很好，这全部都是我想要的，但是，不是我想要的全部。

在分析源码时，同个关键字会在不同文件的不同位置多次出现，grep.vim 和 ack.vim 只能“将关键字所在行罗列出来”，如果关键字出现的那几行完全相同，那么，我单凭这个列表是无法确定哪行是我需要的，比如，我查找关键字 cnt，代码中，cnt 在 4 行出现过、64 行、128 行、1024 行都出现过，且每行内容均相同。除了罗列关键字所在行之外，我还需要看到所在行的上下几行，这样，有了上下文，我就可以最终决定哪一行是我需要的了。ctrlsf.vim（<https://github.com/dyng/ctrlsf.vim> ）为此而生。

ctrlsf.vim 后端调用 ack，所以你得提前自行安装，版本不得低于 v2.0。ctrlsf.vim 支持 ack 所有选项，要查找某个关键字（如，yangyang），你可以想让光标定位在该关键字上面，然后命令模式下键入`:CtrlSF`将自动提取光标所在关键字进行查找，你也可以指定 ack 的选项

```vimscript
:CtrlSF -i -C 1 [pattern] /my/path/
```

为方便操作设定了快捷键：

```vimscript
" 使用 ctrlsf.vim 插件在工程内全局查找光标所在关键字，设置快捷键。快捷键速记法：search in project
nnoremap <Leader>sp :CtrlSF<CR>
```

避免手工键入命令的麻烦。查找结果将以子窗口在左侧呈现，不仅罗列出所有匹配项，而且给出匹配项的上下文。如果从上下文中你还觉得信息量不够，没事，可以键入 p 键，将在右侧子窗口中给出该匹配项的完整代码，而不再仅有前后几行。不想跳至任何匹配项，可以直接键入 q 退出 ctrlsf.vim；如果有钟意的匹配项，光标定位该项后回车，立即跳至新 buffer 中对应位置。

##### 快捷替换

前面介绍的 ctrlsf 已经把匹配的字符串汇总在侧边子窗口中显示了，同时，它还允许我们直接在该子窗口中进行编辑操作，在这种环境下，如果我们能快捷选中所有匹配字符串，那么就可以先批量删除再在原位插入新的字符串。快捷选中 ctrlsf 子窗口中的多个匹配项，关键还是这些匹配项分散在不同行的不同位置，这就需要多光标编辑功能，vim-multiple-cursors 插件（<https://github.com/terryma/vim-multiple-cursors> ）为此而生。装好 vim-multiple-cursors 后，你随便编辑个文档，随便输入多个相同的字符串，先在可视化模式下选中其中一个，接着键入 ctrl-n，你会发现第二个该字符串也被选中了，持续键入 ctrl-n，你可以选中所有相同的字符串，把这个功能与 ctrlsf 结合。

##### 精确替换

vim 有强大的内容替换命令：

```
:[range]s/{pattern}/{string}/[flags]
```

在进行内容替换操作时，我关注几个因素：如何指定替换文件范围、是否整词匹配、是否逐一确认后再替换。

如何指定替换文件范围？

- 如果在当前文件内替换，[range] 不用指定，默认就在当前文件内；

- 如果在当前选中区域，[range] 也不用指定，在你键入替换命令时，vim 自动将生成如下命令：

  ```vimscript
  :'<,'>s/{pattern}/{string}/[flags]
  ```

- 你也可以指定行范围，如，第三行到第五行：

  ```vimscript
  :3,5s/{pattern}/{string}/[flags]
  ```

- 如果对打开文件进行替换，你需要先通过 :bufdo 命令显式告知 vim 范围，再执行替换；

- 如果对工程内所有文件进行替换，先 :args **/*.cpp \**/*.h 告知 vim 范围，再执行替换；

是否整词匹配？{pattern} 用于指定匹配模式。如果需要整词匹配，则该字段应由 < 和 > 修饰待替换字符串（如，<iFoo>）；无须整词匹配则不用修饰，直接给定该字符串即可；

是否逐一确认后再替换？[flags] 可用于指定是否需要确认。若无须确认，该字段设定为 ge 即可；有时不见得所有匹配的字符串都需替换，若在每次替换前进行确认，该字段设定为 gec 即可。

是否整词匹配和是否确认两个条件叠加就有 4 种组合：非整词且不确认、非整词且确认、整词且不确认、整词且确认，每次手工输入这些命令真是麻烦；我把这些组合封装到一个函数中，如下 Replace() 所示：

```vimscript
" 替换函数。参数说明：
" confirm：是否替换前逐一确认
" wholeword：是否整词匹配
" replace：被替换字符串
function! Replace(confirm, wholeword, replace)
    wa
    let flag = ''
    if a:confirm
        let flag .= 'gec'
    else
        let flag .= 'ge'
    endif
    let search = ''
    if a:wholeword
        let search .= '\<' . escape(expand('<cword>'), '/\.*$^~[') . '\>'
    else
        let search .= expand('<cword>')
    endif
    let replace = escape(a:replace, '/\&~')
    execute 'argdo %s/' . search . '/' . replace . '/' . flag . '| update'
endfunction
```

为最大程度减少手工输入，Replace() 还能自动提取待替换字符串（只要把光标移至待替换字符串上），同时，替换完成后自动为你保存更改的文件。现在要做的就是赋予 confirm、wholeword 不同实参实现 4 种组合，再绑定 4 个快捷键即可。如下：

```vimscript
" 不确认、非整词
nnoremap <Leader>R :call Replace(0, 0, input('Replace '.expand('<cword>').' with: '))<CR>
" 不确认、整词
nnoremap <Leader>rw :call Replace(0, 1, input('Replace '.expand('<cword>').' with: '))<CR>
" 确认、非整词
nnoremap <Leader>rc :call Replace(1, 0, input('Replace '.expand('<cword>').' with: '))<CR>
" 确认、整词
nnoremap <Leader>rcw :call Replace(1, 1, input('Replace '.expand('<cword>').' with: '))<CR>
nnoremap <Leader>rwc :call Replace(1, 1, input('Replace '.expand('<cword>').' with: '))<CR>
```



##### 快速注释

插件[NERD Comment](https://github.com/scrooloose/nerdcommenter)会根据文档扩展名来自适应注释代码，\*.cpp采用//注释风格，*.x采用/××/风格；而选中部分并非整行时，将用/**/只注释部分。常用操作：

- \<leader\>cc，注释当前选中文本，如果选中的是整行则在每行首添加 //，如果选中一行的部分内容则在选中部分前后添加分别 /*、*/；
- \<leader\>cu，取消选中文本块的注释。

若需要 ASCII art 风格的注释，可用[DrawIt](https://github.com/vim-scripts/DrawIt)，可以用方向键快速绘制出。常用操作有两个，:Distart，开始绘制，可用方向键绘制线条，空格键绘制或擦除字符；:Distop，停止绘制。

##### 模板补全

插件[UltiSnips](https://github.com/SirVer/ultisnips)能自动完成代码片断模板的输入，并且光标停留在需要编辑的位置，比如键入`if`自动补充为

```cpp
if (/* condition */) {
    TODO
}
```

且选中`/* condition */`部分。在模板补全时键入模板名（如if）后键入补全快捷键（默认 \<tab\>），UltiSnips会根据模板名在模板文件中搜索匹配的“模板名-模板”，找到模板后在光标位置展开。UltiSnips有一套自己的代码模板语法规则，类似：

```vimscript
snippet if "if statement" i
if (${1:/* condition */}) { 
    ${2:TODO} 
} 
endsnippet
```

其中`snippet`和`endsnippet`表示模板的开始和结束；`if`是模板名；`"if statement"` 是模板描述，可将多个模板名称定义成一样（如`if(){}`和`if(){}else{}`都定义为if），在模板描述中区分（如分别对应 "if statement" 和 "if-else statement"），这样在YCM的补全列表中可根据模板描述区分选项不同模板；`i`是模板控制参数，用于控制模板补全行为，具体参见“快速编辑结对符”一节；`${1}`、`${2}`是\<tab\> 跳转的先后顺序。

在[vim-snippets](https://github.com/honza/vim-snippets)有各类语言的代码模板，也可自定义模板，只要在.vimrc中设定模板所在目录名。比如自定义模板文件路径为~/.vim/bundle/ultisnips/mysnippets/cpp.snippets，则对应设置`let g:UltiSnipsSnippetDirectories=["mysnippets"]`，注意目录需要是vim运行时目录~/.vim/bundle/的子目录，且名称切勿取为UltiSnips内部保留关键字snippets。完整cpp.snippets内容见附录1。UltiSnips默认模板补全快捷键是`<tab>`，与YCM快捷键冲突，因此必须在.vimrc重新设定：

```vimscript
" UltiSnips 的 tab 键与 YCM 冲突，重新设定
let g:UltiSnipsExpandTrigger="<leader><tab>"
let g:UltiSnipsJumpForwardTrigger="<leader><tab>"
let g:UltiSnipsJumpBackwardTrigger="<leader><s-tab>"
```

##### 基于标签的智能补全

前面介绍过标签，其每项含有标签名、作用域等信息，当键入某几个字符时，基于标签的补全插件就在标签文件中搜索匹配的标签项并罗列出来，这与代码导航类似，前者用于输入、后者用于跳转。基于标签补全，后端ctags先生成标签文件，前端采用插件new-omni-completion（内置）进行识别。这种方式操作简单、效果不错，一般来说两步搞定。

1. 生成标签文件。在工程目录的根目录执行ctags，该目录下会多出个 tags 文件；

2. 引入标签文件。在 vim 中引入标签文件，在 vim 中执行命令。

   ```vimscript
   :set tags+=/home/your_proj/tags
   ```

之后编码时，键入标签前几个字符后依次键入`ctrl-x ctrl-o`将罗列匹配标签列表，键入`ctrl-x ctrl-i`则补全文件名、`ctrl-x ctrl-f`则补全路径。

比如要智能补全C++标准库。首先获取C++标准库源码文件。可用如下命令安装GNU C++标准库源码文件：

```bash
sudo apt-get install libstdc++-5-dev
```

安装成功后，在 /usr/include/c++/5/ 可见到所有源码文件；接着，执行ctags生成标准库的标签文件：

```bash
cd /usr/include/c++/5
ctags -R --c++-kinds=+l+x+p --fields=+iaSl --extra=+q --language-force=c++ -f stdcpp.tags
```

GNU C++标准库源码文件中使用了_GLIBCXX_STD名称空间，标签文件里面的各个标签都嵌套在该名字空间下，所以，要让OmniCppComplete正确识别这些标签，必须显式告知其相应的名字空间名称。在.vimrc中增加如：

```vimscript
let OmniCpp_DefaultNamespaces = ["_GLIBCXX_STD"]
```

最后，在 vim 中引入该标签文件。在 .vimrc 中增加如下内容：

```vimscript
set tags+=/usr/include/c++/4.8/stdcpp.tags
```

后续就可以进行 C++ 标准库的代码补全，在某个对象名输入`.`时，vim自动显示成员列表。又比如要补全linux系统API，首先，获取linux系统API头文件：

```bash
sudo apt-get install linux-libc-dev
```

安装后在/usr/include/ 中可见相关头文件；接着执行ctags生成系统 API 的标签文件，linux源码文件中大量采用 GCC扩展语法，比如，在文件 unistd.h中几乎每个API声明中都会出现`__THROW,__nonnull`关键字，必须借由ctags的`-I`参数告之忽略某些标签，多个忽略字符串之间用逗号分割：

```bash
cd /usr/include/
ctags -R --c-kinds=+l+x+p --fields=+lS -I __THROW,__nonnull -f sys.tags
```

最后在 vim 中引入该标签文件。在 .vimrc 中增加如下内容：

```vimscript
set tags+=/usr/include/sys.tags
```

从以上可看出不论是C++标准库、boost、ACE这些重量级开发库，还是linux系统 API 均可遵循**下载源码（至少包括头文件）-执行 ctags 生产标签文件-引入标签文件**的流程实现基于标签的智能补全。唯有两种可能异常：

- 可能是源码使用了名字空间，可用OmniCppComplete插件的OmniCpp_DefaultNamespaces配置项解决；
- 还可能是源码中使用了编译器扩展语法，借助ctags的-I参数解决（具体请参见相应编译器手册，某个系统函数无法自动补全十有八九是头文件中使用使用了扩展语法，先找到该函数完整声明，再将其使用的扩展语法加入 -I 列表中，最后运行ctags重新生产新标签文件即可）。

##### 基于语义的智能补全

基于标签的补全能够很好地满足需求，但存在几个问题：一是必须定期调用ctags生成标签文件，还有就是ctags本身对C++支持有限。而基于语义的智能补全能实时探测是否有补全需求，且借助编译器进行代码分析，只要编译器支持度则不论标准库、类型推导抑或boost库中的智能指针都能补全，这样就解决了前面的问题。

Linux上有GCC和clang两大主流C++编译器，基于不同编译器，开源社区创造出GCCSense和clang_complete两个语义补全插件，选择clang_complete的原因是：使用难度低，clang采用低耦合设计，语义分析结果（ 即AST）以接口形式供外围程序使用，而GCC采用高耦合设计，必须结合补丁重新源码编译 GCC才能让GCCSense接收到其语义分析结果。clang_complete使用简单，在输入模式下依次键入要补全的字符，需要弹出补全列表时手工输入 \<leader\>tab。

[YCM](https://github.com/Valloric/YouCompleteMe)是一个随键而全、支持模糊搜索、高速补全的插件，后端调用 libclang（以获取 AST）、前端由C++开发（以提升补全效率）、外层由python封装（以成为 vim 插件），可能是安装最复杂的vim插件。安装过程见附录2。YCM集成语义、标签、OmniCppComplete以及其他等多种补全引擎。

- YCM只在两种场景下触发语义补全：一是补全标识符所在文件在buffer中（即文件已打开）；一是在对象后键入`.`、指针后键入`->`、命名空间后键入`::`。

- YCM支持标签补全，需要做两件事：一是让YCM启用标签补全引擎、二是引入 tag 文件，具体设置如下：

  ```vimscript
  " 开启 YCM 标签引擎
  let g:ycm_collect_identifiers_from_tags_files=1
  " 引入 C++ 标准库tags
  set tags+=/data/misc/software/misc./vim/stdcpp.tags
  ```

  工程代码的标签可借助indexer插件自动生成自动引入。但YCM要求tag文件必须含有`language:<X>`字段，即ctags命令行参数`--fields`必须含有`l`选项，必须通过indexer_ctagsCommandLineOptions告知indexer调用ctags时生成该字段，参见“代码导航”章节。注意引入过多tag文件会导致vim变慢，因此只引入工程自身（indexer自动引入）和C++标准库的标签（上面配置的最后一行）。

- YCM还有OmniCppComplete补全，只要在当前代码文件中#include该标识符所在头文件即可。这种补全无法使用YCM随键而全的特性，需要手工告知YCM需要补全，OmniCppComplete的默认补全快捷键不太方便，重新设定为 <leader>;，如前面配置所示：

  ```vimscript
  inoremap <leader>; <C-x><C-o>
  ```

###### 由接口快速生成实现框架

插件[vim-protodef](https://github.com/derekwyatt/vim-protodef)能根据类声明自动生成类实现的代码框架，它依赖[FSwitch](https://github.com/derekwyatt/vim-fswitch)。增加如下设置信息：

```vimscript
" 成员函数的实现顺序与声明顺序一致
let g:disable_protodef_sorting=1
```

pullproto.pl是protodef自带的 perl 脚本，默认位于~/.vim目录，由于改用pathogen管理插件，所以路径需重新设置。protodef根据文件名进行关联，比如MyClass.h 与 MyClass.cpp 是一对接口和实现文件：

```vimscript
" 设置 pullproto.pl 脚本路径
let g:protodefprotogetter='~/.vim/bundle/protodef/pullproto.pl'
" 成员函数的实现顺序与声明顺序一致
let g:disable_protodef_sorting=1
```

MyClass.cpp 中我键入 protodef 定义的快捷键 \PP，自动生成了函数框架。

上图既突显了 protodef 的优点：优点一，virtual、默认参数等应在函数声明而不应在函数定义中出现的关键字，protodef 已为你过滤；优点二：doNothing() 这类纯虚函数不应有实现的自动被 protodef 忽略。同时也暴露了 protodef 问题：printMsg(int = 16) 的函数声明变更为 printMsg(unsigned)，protodef 无法自动为你更新，它把更改后的函数声明视为新函数添加在实现文件中，老声明对应的实现仍然保留。

关于缺点，先我计划优化下 protodef 源码再发给原作者，后来想想，protodef 借助 ctags 代码分析实现的，本来就存在某些缺陷，好吧，后续我找个时间写个与 protodef 相同功能但对 C++ 支持更完善的插件，内部当然借助 libclang 啦。

另外，每个人都有自己的代码风格，比如，return 语句我喜欢

```cpp
return(TODO);
```

所以，调整了 protodef.vim 源码，把 239、241、244、246 四行改为

```vimscript
call add(full, "    return(TODO);") 
```

函数名与形参列表间习惯添加个空格

```cpp
void MyClass::getSize (void);
```

所以，把 217 行改为

```vimscript
let proto = substitute(proto, '(\_.*$', ' (' . params . Tail, '') 
```

函数功能描述、参数讲解、返回值说明的文档，要使用该功能，系统中必须先安装对应 man。安装 linux 系统函数 man，先下载（<https://www.kernel.org/doc/man-pages/download.html> ），解压后将 man1/ 至 man8/ 拷贝至 /usr/share/man/，运行 man fork 确认是否安装成功。安装 C++ 标准库 man，先下载（[ftp://GCC.gnu.org/pub/GCC/libstdc++/doxygen/](ftp://gcc.gnu.org/pub/GCC/libstdc++/doxygen/) ），选择最新 libstdc++-api-X.X.X.man.tar.bz2，解压后将 man3/ 拷贝至 /usr/share/man/，运行 man std::vector 确认是否安装成功；

vim 内置的 man.vim 插件可以查看已安装的 man，需在 .vimrc 中配置启动时自动加载该插件：

```vimscript
" 启用:Man命令查看各类man信息
source $VIMRUNTIME/ftplugin/man.vim
" 定义:Man命令查看各类man信息的快捷键
nmap <Leader>man :Man 3 <cword><CR>
```

需要查看时，在 vim 中键入输入 :Man fork 或者 :Man std::vector （注意大小写）即可在新建分割子窗口中查看到函数参考信息，为了方便，我设定了快捷键 <Leader>man，这样，光标所在单词将被传递给 :Man 命令，不用再手工键入。另外，我们编码时通常都是先声明使用 std 名字空间，在使用某个标准库中的类时前不会添加 std:: 前缀，所以 vim 取到的当前光标所在单词中也不会含有 std:: 前缀，而，C++ 标准库所有 man 文件名均有 std:: 前缀，所以必须将所有文件的 std:: 前缀去掉才能让 :Man 找到正确的 man 文件。在 libstdc++-api-X.X.X.man/man3/ 执行批量重命名以取消所有 man文件的 std:: 前缀：

```vimscript
rename "std::" "" std::\* 
```

很多人以为 rename 命令只是 mv 命令的简单封装，非也，在重命名方面，rename 太专业了，远非 mv 可触及滴，就拿上例来说，mv 必须结合 sed 才能达到这样的效果。

我认为，好的库信息参考手册不仅有对参数、返回值的描述，还应有使用范例，上面介绍的 linux 系统函数 man 做到了，C++ 标准库 man 还未达到我要求。所以，若有网络条件，我更愿意选择查看在线参考，C++ 推荐 <http://www.cplusplus.com/reference/> 、<http://en.cppreference.com/w/Cppreference:Archives> ，前者范例多、后者更新勤；UNIX 推荐 <http://pubs.opengroup.org/onlinepubs/9699919799/functions/contents.html> 、<http://man7.org/linux/man-pages/dir_all_alphabetic.html> ，前者基于最新 SUS（Single UNIX Specification，单一 UNIX 规范）、后者偏重 linux 扩展。



##### 工程管理

通常将工程相关的文档放在同个目录下，通过 NERDtree （<https://github.com/scrooloose/nerdtree> ）插件可以查看文件列表，要打开哪个文件，光标选中后回车即可在新 buffer 中打开。安装好 NERDtree 后，请将如下信息加入.vimrc中：

```vimscript
" 使用 NERDTree 插件查看工程文件。设置快捷键，速记：file list
nmap <Leader>fl :NERDTreeToggle<CR>
" 设置NERDTree子窗口宽度
let NERDTreeWinSize=32
" 设置NERDTree子窗口位置
let NERDTreeWinPos="right"
" 显示隐藏文件
let NERDTreeShowHidden=1
" NERDTree 子窗口中不显示冗余帮助信息
let NERDTreeMinimalUI=1
" 删除文件时自动删除文件对应 buffer
let NERDTreeAutoDeleteBuffer=1
```

常用操作：回车，打开选中文件；r，刷新工程目录文件列表；I（大写），显示/隐藏隐藏文件；m，出现创建/删除/剪切/拷贝操作列表。键入 <leader>fl 后，右边子窗口为工程项目文件列表。

vim 的多文档编辑涉及三个概念：buffer、window、tab，这三个事物与我们常规理解意义大相径庭。vim 把加载进内存的文件叫做 buffer，buffer 不一定可见；若要 buffer 要可见，则必须通过 window 作为载体呈现；同个看面上的多个 window 组合成一个 tab。一句话，vim 的 buffer、window、tab 你可以对应理解成视角、布局、工作区。我所用到的多文档编辑场景几乎不会涉及 tab，重点关注 buffer、window。

vim 中每打开一个文件，vim 就对应创建一个 buffer，多个文件就有多个 buffer，但默认你只看得到最后 buffer 对应的 window，通过插件 MiniBufExplorer（<https://github.com/fholgado/minibufexpl.vim> ，原始版本已停止更新且问题较多，该版本是其他人 fork 的新项目）可以把所有 buffer 罗列出来，并且可以显示多个 buffer 对应的 window。如下图所示：

<img src="buffer 列表.png" />

我在 vim 中打开了 main.cpp、CMakeLists.txt、MyClass.cpp、MyClass.h 这四个文件，最上面子窗口（buffer 列表）罗列出的 [1:main.cpp][4:CMakeLists.txt][5:MyClass.cpp][6:MyClass.h] 就是对应的四个 buffer。当前显示了 main.cpp 和 MyClass.h 的两个 buffer，分别对应绿色区域和橙色区域的 window，这下对 buffer 和 window 有概念了吧。图中关于 buffer 列表再说明两点：

- \* 表示当前有 window 的 buffer，换言之，有 * 的 buffer 是可见的；! 表示当前正在编辑的 window；
- 你注意到 buffer 序号 1 和 4 不连续的现象么？只要 vim 打开一个 buffer，序号自动增一，中间不连续有几个可能：可能一，打开了 1、2、3、4 后，用户删除了 2、3 两个 buffer，剩下 1、4；可能二，先打开了其他插件的窗口（如，tagbar）后再打开代码文件；

配置：将如下信息加入 .vimrc 中：

```vimscript
" 显示/隐藏 MiniBufExplorer 窗口
map <Leader>bl :MBEToggle<cr>
" buffer 切换快捷键
map <C-Tab> :MBEbn<cr>
map <C-S-Tab> :MBEbp<cr>
```

操作：一般通过 NERDtree 查看工程文件列表，选择打开多个代码文件后，MiniBufExplorer 在顶部自动创建 buffer 列表子窗口。通过前面配置，ctrl-tab 正向遍历 buffer，ctrl-shift-tab 逆向遍历（光标必须在 buffer 列表子窗口外）；在某个 buffer 上键入 d 删除光标所在的 buffer（光标必须在 buffer 列表子窗口内）：

<img src="多文档编辑.gif" />

默认时，打开的 window 占据几乎整个 vim 编辑区域，如果你想把多个 window 平铺成多个子窗口可以使用 MiniBufExplorer 的 s 和 v 命令：在某个 buffer 上键入 s 将该 buffer 对应 window 与先前 window 上下排列，键入 v 则左右排列（光标必须在 buffer 列表子窗口内）。如下图所示：

<img src="在子窗口中编辑多文档.gif" />

图中，通过 vim 自身的 f 名字查找 buffer 序号可快速选择需要的 buffer。另外，编辑单个文档时，不会出现 buffer 列表。

vim 的编辑环境保存与恢复是我一直想要的功能，我希望每当重新打开 vim 时恢复：已打开文件、光标位置、undo/redo、书签、子窗口、窗口大小、窗口位置、命令历史、buffer 列表、代码折叠。vim 文档说 viminfo 特性可以恢复书签、session 特性可以恢复书签外的其他项，所以，请确保你的 vim 支持这两个特性：

```bash
vim --version | grep mksession
vim --version | grep viminfo
```

如果编译 vim 时添加了 --with-features=huge 选项那就没问题。一般说来，保存/恢复环境步骤如下。

第一步，保存所有文档：

```vimscript
:wa
```

第二步，借助 viminfo 和 session 保存当前环境：

```vimscript
:mksession! my.vim
:wviminfo! my.viminfo
```

第三步，退出 vim：

```vimscript
:qa
```

第四步，恢复环境，进入 vim 后执行：

```vimscript
:source my.vim
:rviminfo my.viminfo
```

具体能保存哪些项，可由 sessionoptions 指定，另外，前面几步可以设定快捷键，在 .vimrc 中增加：

```vimscript
" 设置环境保存项
set sessionoptions="blank,buffers,globals,localoptions,tabpages,sesdir,folds,help,options,resize,winpos,winsize"
" 保存 undo 历史
set undodir=~/.undo_history/
set undofile
" 保存快捷键
map <leader>ss :mksession! my.vim<cr> :wviminfo! my.viminfo<cr>
" 恢复快捷键
map <leader>rs :source my.vim<cr> :rviminfo my.viminfo<cr>
```

这样，简化第二步、第四步操作。另外，sessionoptions 无法包含 undo 历史，你得先得手工创建存放 undo 历史的目录（如，.undo_history/）再通过开启 undofile 进行单独设置，一旦开启，每次写文件时自动保存 undo 历史，下次加载在文件时自动恢复所有 undo 历史，不再由 :mksession/:wviminfo 和 :source/:rviminfo 控制。



##### 附录：

###### 1 完整cpp.snippets例子

```vimscript
#================================= 
#预处理 
#================================= 
# #include "..." 
snippet INC 
#include "${1:TODO}"${2} 
endsnippet 
# #include <...> 
snippet inc 
#include <${1:TODO}>${2} 
endsnippet 
#================================= 
#结构语句 
#================================= 
# if 
snippet if 
if (${1:/* condition */}) { 
    ${2:TODO} 
} 
endsnippet 
# else if 
snippet ei 
else if (${1:/* condition */}) { 
    ${2:TODO} 
} 
endsnippet 
# else 
snippet el 
else { 
    ${1:TODO} 
} 
endsnippet 
# return 
snippet re 
return(${1:/* condition */}); 
endsnippet 
# Do While Loop 
snippet do 
do { 
    ${2:TODO} 
} while (${1:/* condition */}); 
endsnippet 
# While Loop 
snippet wh 
while (${1:/* condition */}) { 
    ${2:TODO} 
} 
endsnippet 
# switch 
snippet sw 
switch (${1:/* condition */}) { 
    case ${2:c}: { 
    } 
    break; 

    default: { 
    } 
    break; 
} 
endsnippet 
# 通过迭代器遍历容器（可读写） 
snippet for 
for (auto ${2:iter} = ${1:c}.begin(); ${3:$2} != $1.end(); ${4:++iter}) {
    ${5:TODO} 
} 
endsnippet 
# 通过迭代器遍历容器（只读） 
snippet cfor 
for (auto ${2:citer} = ${1:c}.cbegin(); ${3:$2} != $1.cend(); ${4:++citer}) { 
    ${5:TODO} 
} 
endsnippet 
# 通过下标遍历容器 
snippet For 
for (decltype($1.size()) ${2:i} = 0; $2 != ${1}.size(); ${3:++}$2) { 
    ${4:TODO} 
} 
endsnippet 
# C++11风格for循环遍历（可读写） 
snippet F 
for (auto& e : ${1:c}) { 
} 
endsnippet 
# C++11风格for循环遍历（只读） 
snippet CF 
for (const auto& e : ${1:c}) { 
} 
endsnippet 
# For Loop 
snippet FOR 
for (unsigned ${2:i} = 0; $2 < ${1:count}; ${3:++}$2) { 
    ${4:TODO} 
} 
endsnippet 
# try-catch 
snippet try 
try { 
} catch (${1:/* condition */}) { 
} 
endsnippet 
snippet ca 
catch (${1:/* condition */}) { 
} 
endsnippet 
snippet throw 
th (${1:/* condition */}); 
endsnippet 
#================================= 
#容器 
#================================= 
# std::vector 
snippet vec 
vector<${1:char}>	v${2}; 
endsnippet 
# std::list 
snippet lst 
list<${1:char}>	l${2}; 
endsnippet 
# std::set 
snippet set 
set<${1:key}>	s${2}; 
endsnippet 
# std::map 
snippet map 
map<${1:key}, ${2:value}>	m${3}; 
endsnippet 
#================================= 
#语言扩展 
#================================= 
# Class 
snippet cl 
class ${1:`Filename('$1_t', 'name')`} 
{ 
    public: 
        $1 (); 
        virtual ~$1 (); 
         
    private: 
}; 
endsnippet 
#================================= 
#结对符 
#================================= 
 # 括号 bracket 
snippet b "bracket" i 
(${1})${2} 
endsnippet 
# 方括号 square bracket，设定为 st 而非 sb，避免与 b 冲突
snippet st "square bracket" i 
[${1}]${2} 
endsnippet 
# 大括号 brace 
snippet br "brace" i 
{ 
    ${1} 
}${2} 
endsnippet 
# 单引号 single quote，设定为 se 而非 sq，避免与 q 冲突
snippet se "single quote" I
'${1}'${2}
endsnippet
# 双引号 quote
snippet q "quote" I
"${1}"${2}
endsnippet
# 指针符号 arrow 
snippet ar "arrow" i 
->${1} 
endsnippet 
# dot 
snippet d "dot" i 
.${1} 
endsnippet 
# 作用域 scope 
snippet s "scope" i 
::${1} 
endsnippet
```

###### 2 安装和设置YCM插件

要运行 YCM，需要vim版本至少达到7.3.598且支持 python2/3；YCM含与vim交互的插件部分，及与libclang交互的自身共享库两部分。安装过程如下：

第一步，通过 vundle 安装 YCM 插件，

```vimscript
Plugin 'Valloric/YouCompleteMe'
```

随后进入vim执行`:PluginInstall`。

第二步，下载 libclang。强烈建议下载LLVM官网的提供[Pre-built Binaries](http://llvm.org/releases/download.html)，选择适合发行套件最新版预编译二进制文件，下载并解压至 ~/downloads/clang+llvm；

第三步，编译 YCM 共享库

```bash
sudo apt-get install python-dev python3-dev libboost-all-dev libclang-dev
cd ~/downloads/
mkdir ycm_build
cd ycm_build
cmake -G "Unix Makefiles" -DUSE_SYSTEM_BOOST=ON -DPATH_TO_LLVM_ROOT=~/downloads/clang+llvm/ .\
 ~/.vim/bundle/YouCompleteMe/third_party/ycmd/cpp
cmake --build . --target ycm_core
```

在~/.vim/bundle/YouCompleteMe/third_party/ycmd中将生成 ycm_client_support.so、ycm_core.so、libclang.so 等三个共享库文件。接下来需要设置YCM：

设置一，libclang有很多参数选项，YCM为每个工程设置一个名为.ycm_extra_conf.py的私有配置文件，在此文件中写入工程的编译参数选项。下面是个完整的例子：

```python
import os 
import ycm_core 
flags = [ 
    '-std=c++11', 
    '-O0', 
    '-Werror', 
    '-Weverything', 
    '-Wno-documentation', 
    '-Wno-deprecated-declarations', 
    '-Wno-disabled-macro-expansion', 
    '-Wno-float-equal', 
    '-Wno-c++98-compat', 
    '-Wno-c++98-compat-pedantic', 
    '-Wno-global-constructors', 
    '-Wno-exit-time-destructors', 
    '-Wno-missing-prototypes', 
    '-Wno-padded', 
    '-Wno-old-style-cast',
    '-Wno-weak-vtables',
    '-x', 
    'c++', 
    '-I', 
    '.', 
    '-isystem', 
    '/usr/include/', 
] 
compilation_database_folder = '' 
if compilation_database_folder: 
  database = ycm_core.CompilationDatabase( compilation_database_folder ) 
else: 
  database = None 
SOURCE_EXTENSIONS = [ '.cpp', '.cxx', '.cc', '.c', '.m', '.mm' ] 
def DirectoryOfThisScript(): 
  return os.path.dirname( os.path.abspath( __file__ ) ) 
def MakeRelativePathsInFlagsAbsolute( flags, working_directory ): 
  if not working_directory: 
    return list( flags ) 
  new_flags = [] 
  make_next_absolute = False 
  path_flags = [ '-isystem', '-I', '-iquote', '--sysroot=' ] 
  for flag in flags: 
    new_flag = flag 
    if make_next_absolute: 
      make_next_absolute = False 
      if not flag.startswith( '/' ): 
        new_flag = os.path.join( working_directory, flag ) 
    for path_flag in path_flags: 
      if flag == path_flag: 
        make_next_absolute = True 
        break 
     if flag.startswith( path_flag ): 
        path = flag[ len( path_flag ): ] 
        new_flag = path_flag + os.path.join( working_directory, path ) 
        break 
   if new_flag: 
      new_flags.append( new_flag ) 
  return new_flags 
def IsHeaderFile( filename ): 
  extension = os.path.splitext( filename )[ 1 ] 
  return extension in [ '.h', '.hxx', '.hpp', '.hh' ] 
def GetCompilationInfoForFile( filename ): 
  if IsHeaderFile( filename ): 
    basename = os.path.splitext( filename )[ 0 ] 
    for extension in SOURCE_EXTENSIONS: 
      replacement_file = basename + extension 
      if os.path.exists( replacement_file ): 
        compilation_info = database.GetCompilationInfoForFile( replacement_file ) 
        if compilation_info.compiler_flags_: 
          return compilation_info 
    return None 
  return database.GetCompilationInfoForFile( fYCM 集成了多种补全引擎：语义补全引擎、标签补全引擎、OmniCppComplete 补全引擎、其他补全引擎。

YCM 的语义补全。YCM 不会在每次键入事件上触发语义补全，YCM 作者认为这会影响补全效率而且没什么必要（我持保留意见），YCM 只在如下两种场景下触发语义补全：一是补全标识符所在文件必须在 buffer 中（即，文件已打开）；一是在对象后键入 .、指针后键入 ->、名字空间后键入 ::。

上图中，我先尝试补全类 MyClass 失败，接着我把 MyClass 所在的文件 MyClass.h 打开后切回 main.cpp 再次补全类 MyClass 成功，然后在对象 mc 后键入 . 进行成员补全；

YCM 的标签补全。语义补全的确强大，但受限挺多，如果我要补全 STL 中的泛型算法 count_if() 岂不是还要先打开库头文件 algorithm？不用，YCM 也支持标签补全。要使用标签补全，你需要做两件事：一是让 YCM 启用标签补全引擎、二是引入 tag 文件，具体设置如下：

    " 开启 YCM 标签引擎
    let g:ycm_collect_identifiers_from_tags_files=1
    " 引入 C++ 标准库tags
    set tags+=/data/misc/software/misc./vim/stdcpp.tags

其中，工程自身代码的标签可借助 indexer 插件自动生成自动引入，但由于 YCM 要求 tag 文件中必须含有 language:<X> 字段（ctags 的命令行参数 --fields 必须含有 l 选项），所有必须通过 indexer_ctagsCommandLineOptions 告知 indexer 调用 ctags 时注意生成该字段，具体设置参见“代码导航”章节；前面章节介绍过如何生成、引入 C++ 标准库的 tag 文件，设置成正确路径即可。另外，由于引入过多 tag 文件会导致 vim 变得非常缓慢，我的经验是，只引入工程自身（indexer 自动引入）和 C++ 标准库的标签（上面配置的最后一行）。

YCM 的 OmniCppComplete 补全引擎。我要进行 linux 系统开发，打开系统函数头文件觉得麻烦（也就无法使用 YCM 的语义补全），引入系统函数 tag 文件又影响 vim 速度（也就无法使用 YCM 的标签补全），这种情况又如何让 YCM 补全呢？WOW，别担心，YCM 还有 OmniCppComplete 补全引擎，只要你在当前代码文件中 #include 了该标识符所在头文件即可。通过 OmniCppComplete 补全无法使用 YCM 的随键而全的特性，你需要手工告知 YCM 需要补全，OmniCppComplete 的默认补全快捷键为 <C-x><C-o>，不太方便，我重新设定为 <leader>;，如前面配置所示：
ilename ) 
def FlagsForFile( filename, \*\*kwargs ): 
  if database: 
    compilation_info = GetCompilationInfoForFile( filename ) 
    if not compilation_info: 
      return None 
   final_flags = MakeRelativePathsInFlagsAbsolute( 
      compilation_info.compiler_flags_, 
      compilation_info.compiler_working_dir_ ) 
  else: 
    relative_to = DirectoryOfThisScript() 
    final_flags = MakeRelativePathsInFlagsAbsolute( flags, relative_to ) 
 return { 
    'flags': final_flags, 
    'do_cache': True 
  }
```

基本上根据工程情况只需调整.ycm_extra_conf.py的flags部分，flags用于YCM调用libclang时指定的参数，通常应与构建脚本保持一致（如CMakeLists.txt）。flags 会产生两方面影响，一是影响 YCM 的补全内容、一是影响代码静态分析插件syntastic的显示结果（详见后文“静态分析器集成”）。

设置二，在 .vimrc 中增加如下配置信息：

```vimscript
" YCM 补全菜单配色
" 菜单
highlight Pmenu ctermfg=2 ctermbg=3 guifg=#005f87 guibg=#EEE8D5
" 选中项
highlight PmenuSel ctermfg=2 ctermbg=3 guifg=#AFD700 guibg=#106900
" 补全功能在注释中同样有效
let g:ycm_complete_in_comments=1
" 允许 vim 加载 .ycm_extra_conf.py 文件，不再提示
let g:ycm_confirm_extra_conf=0
" 开启 YCM 标签补全引擎
let g:ycm_collect_identifiers_from_tags_files=1
" 引入 C++ 标准库tags
set tags+=/data/misc/software/misc./vim/stdcpp.tags
" YCM 集成 OmniCppComplete 补全引擎，设置其快捷键
inoremap <leader>; <C-x><C-o>
" 补全内容不以分割子窗口形式出现，只显示补全列表
set completeopt-=preview
" 从第一个键入字符就开始罗列匹配项
let g:ycm_min_num_of_chars_for_completion=1
" 禁止缓存匹配项，每次都重新生成匹配项
let g:ycm_cache_omnifunc=0
" 语法关键字补全			
let g:ycm_seed_identifiers_with_syntax=1
```



##### [Indent Guides](https://github.com/nathanaelkane/vim-indent-guides)

Vim中可视化展示缩进层次的插件。

**特点**：

- 能检测空格和tab两种缩进风格；
- 在缩进层次上交替使用颜色；
- 可定制缩进指南的大小；
- 可定制开始的缩进层次；
- 支持高亮显示混用tab和空格的文件。

**用法**：

默认开关插件的映射是`<leader>ig`；

也可在vim内使用下面的命令：

```VimL
:IndentGuidesEnable
:IndentGuidesDisable
:IndentGuidesToggle
```

设置默认启动，在.vimrc内添加：

```VimL
let g:indent_guides_enable_on_vim_startup = 1
```

**gvim**

插件会自动检测gvim色彩模式并选择合适的颜色。下面是设定定制缩进颜色的例子，在.vimrc中添加：

```VimL
let g:indent_guides_auto_colors = 0
autocmd VimEnter,Colorscheme * :hi IndentGuidesOdd  guibg=red   ctermbg=3
autocmd VimEnter,Colorscheme * :hi IndentGuidesEven guibg=green ctermbg=4
```

或者，将下面的代码添加到色彩模式文件中：

```VimL
hi IndentGuidesOdd  guibg=red   ctermbg=3
hi IndentGuidesEven guibg=green ctermbg=4
```

**终端vim**

目前对终端vim支持有限，颜色仅按`background`设定为`dark`或`light`。

若设为`dark`，则高亮色定义为：

```VimL
hi IndentGuidesOdd  ctermbg=black
hi IndentGuidesEven ctermbg=darkgrey
```

而为`light`时，高亮定义为：

```VimL
hi IndentGuidesOdd  ctermbg=white
hi IndentGuidesEven ctermbg=lightgrey
```

当然也可以定制高亮色，见`:help indent_guides_auto_colors`。

**帮助**：

见`:help indent-guides`。



##### [Powerline](https://github.com/powerline/powerline)



##### [vim-signature](https://github.com/kshenoy/vim-signature)

要求vim以`+signs`编译，安装后会定义下面的映射，主要可以分成四类：

1. 创建标记：
   - `m[a-zA-Z]`：
   - `m,`：
   - `m.`：
2. 移除标记：
   - `dm[a-zA-Z]`：移除标记x；
   - `m-`：移除从当前行的所有标记；
   - `m<space>`：移除缓冲区的所有标记；
3. 书签跳转：
   - ]`：跳转到下个书签；
   - 



##### ctags