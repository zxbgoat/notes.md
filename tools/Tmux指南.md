##### Tmux

来自OpenBSD的优秀终端复用软件。安装完成后输入命令tmux即可开启tmux服务器，它会首先创建一个会话，而这个会话则会首先创建一个窗口，其中仅包含一个面板；也就是说，这里看到的所谓终端控制台应该称作tmux的一个面板，tmux使用C/S模型构建，主要包括以下单元模块：

- server，服务器，输入tmux命令时就开启了一个服务器；
- session，会话，一个服务器可以包含多个会话；
- window，窗口，一个会话可以包含多个窗口；
- pane，面板，一个窗口可以包含多个面板。

Tmux下`<c-b>`激活控制台，然后按下快捷键执行：

- 系统操作：
  - `?`：列出所有快捷键，按q返回；
  - `d`：脱离当前会话；这样可以暂时返回Shell界面，输入tmux attach能够重新进入之前的会话；
  - `D`：选择要脱离的会话；在同时开启了多个会话时使用；
  - `<c-z>`：挂起当前会话；
  - `s`：选择并切换会话；在同时开启了多个会话时使用；
  - `:`：进入命令行模式；此时可以输入支持的命令，例如kill-server可以关闭服务器；
  - `[`：进入复制模式；此时的操作与vi/emacs相同，按q/Esc退出；
  - `~`：列出提示信息缓存；其中包含了之前tmux返回的各种提示信息；
- 窗口操作：
  - `c`：创建新窗口；
  - `&`：关闭当前窗口；
  - `[0-9]`：切换至指定出口；
  - `p`：切换至上一窗口；
  - `n`：切换至下一窗口；
  - `l`：在前后两个窗口间互相切换；
  - `w`：通过窗口列表切换窗口；
  - `'`：重命名当前窗口；这样便于识别；
  - `.`：修改当前窗口编号；相当于窗口重新排序；
  - `f`：在所有窗口中查找指定文本；
- 面板操作：
  - `"`：将当前面板平分为上下两块；
  - `%`：将当前面板平分为左右两块；
  - `x`：关闭当前面板；
  - `!`：将当前面板置于新窗口，即新建一个窗口，其中仅包含当前面板；
  - `<c-方向键>`：以1个单元格为单位移动边缘以调整当前面板大小；
  - `<alt-方向键>`：以5个单元格为单位移动边缘以调整当前面板大小；
  - `<space>`：在预置的面板布局中循环切换；依次包括even-horizontal、even-vertical、main-horizontal、main-vertical、tiled；
  - `q`：显示面板编号；
  - `o`：在当前窗口中选择下一面板；
  - `方向键`：移动光标以选择面板；

tmux的系统级配置文件为/etc/tmux.conf，用户级配置文件为~/.tmux.conf。配置文件实际上就是tmux的命令集合，也就是说每行配置均可在进入命令行模式后输入生效。
下面是一个~/.tmux.conf的示例，其中包括了一些常用的配置：

```tmux
#此类配置可以在命令行模式中输入show-options -g查询
set-option -g base-index 1                        #窗口的初始序号；默认为0，这里设置为1
set-option -g display-time 5000                   #提示信息的持续时间；设置足够的时间以避免看不清提示，单位为毫秒
set-option -g repeat-time 1000                    #控制台激活后的持续时间；设置合适的时间以避免每次操作都要先激活控制台，单位为毫秒
set-option -g status-keys vi                      #操作状态栏时的默认键盘布局；可以设置为vi或emacs
set-option -g status-right "#(date +%H:%M' ')"    #状态栏右方的内容；这里的设置将得到类似23:59的显示
set-option -g status-right-length 10              #状态栏右方的内容长度；建议把更多的空间留给状态栏左方（用于列出当前窗口）
set-option -g status-utf8 on                      开启状态栏的UTF-8支持

#此类设置可以在命令行模式中输入show-window-options -g查询
set-window-option -g mode-keys vi    #复制模式中的默认键盘布局；可以设置为vi或emacs
set-window-option -g utf8 on         #开启窗口的UTF-8支持

#将激活控制台的快捷键由Ctrl+b修改为Ctrl+a
set-option -g prefix C-a
unbind-key C-b
bind-key C-a send-prefix

#添加自定义快捷键
bind-key z kill-session                     #按z结束当前会话；相当于进入命令行模式后输入kill-session
bind-key h select-layout even-horizontal    #按h将当前面板布局切换为even-horizontal；相当于进入命令行模式后输入select-layout even-horizontal
bind-key v select-layout even-vertical      #按v将当前面板布局切换为even-vertical；相当于进入命令行模式后输入select-layout even-vertical
```

更多信息可输入`man tmux`阅读。



Tmux是终端里的窗口管理器。前面提到的窗口管理只是 tmux 功能的一小部分，另一个很有用的功能就是，连接到远程主机之后，一旦断开，那么当前账户登录的任务就被取消了，但是使用 tmux 可以在断开之后继续工作，下次登录可以查看。其他的功能还有：

1. 窗口切换，每个窗口里还可以分割面板
2. 配置方便，可以使用脚本
3. 类似 vim 的双层操作逻辑
4. 复制粘贴缓冲区

安装的话，在 mac 下直接 `brew install tmux`，ubuntu 下则直接 `sudo apt-get install tmux`，在终端中输入 `tmux` 就可以打开一个新的 tmux session，tmux 的所有操作必须先使用一个前缀键（默认是 `ctrl + b`）进入命令模式，或者说进入控制台，就像 vim 中的 esc。



##### 基本操作

###### 信息查询

- `tmux list-keys` 列出所有可以的快捷键和其运行的 tmux 命令
- `tmux list-commands` 列出所有的 tmux 命令及其参数
- `tmux info` 流出所有的 session, window, pane, 运行的进程号等。

###### 窗口控制

先来看看在 tmux 之外如何进行控制

- session 会话：session是一个特定的终端组合。输入tmux就可以打开一个新的session
  - `tmux new -s session_name` 创建一个叫做 `session_name` 的 tmux session
  - `tmux attach -t session_name` 重新开启叫做 `session_name` 的 tmux session
  - `tmux switch -t session_name` 转换到叫做 `session_name` 的 tmux session
  - `tmux list-sessions` / `tmux ls` 列出现有的所有 session
  - `tmux detach` 离开当前开启的 session
  - `tmux kill-server` 关闭所有 session
- window 窗口：session 中可以有不同的 window（但是同时只能看到一个 window）
  - `tmux new-window` 创建一个新的 window
  - `tmux list-windows`
  - `tmux select-window -t :0-9` 根据索引转到该 window
  - `tmux rename-window` 重命名当前 window
- pane 面板：window 中可以有不同的 pane（可以把 window 分成不同的部分）
  - `tmux split-window` 将 window 垂直划分为两个 pane
  - `tmux split-window -h` 将 window 水平划分为两个 pane
  - `tmux swap-pane -[UDLR]` 在指定的方向交换 pane
  - `tmux select-pane -[UDLR]` 在指定的方向选择下一个 pane

更常用的是在 tmux 中直接通过默认前缀 `ctrl + b` 之后输入对应命令来操作，具体如下（这里只列出输入默认前缀之后需要输入的操作）：

###### 基本操作

- `?` 列出所有快捷键；按q返回
- `d` 脱离当前会话,可暂时返回Shell界面
- `s` 选择并切换会话；在同时开启了多个会话时使用
- `D` 选择要脱离的会话；在同时开启了多个会话时使用
- `:` 进入命令行模式；此时可输入支持的命令，例如 `kill-server` 关闭所有tmux会话
- `[` 复制模式，光标移动到复制内容位置，空格键开始，方向键选择复制，回车确认，q/Esc退出
- `]` 进入粘贴模式，粘贴之前复制的内容，按q/Esc退出
- `~` 列出提示信息缓存；其中包含了之前tmux返回的各种提示信息
- `t` 显示当前的时间
- `ctrl + z` 挂起当前会话

###### 窗口操作

- `c` 创建新窗口
- `&` 关闭当前窗口
- `[0-9]` 数字键切换到指定窗口
- `p` 切换至上一窗口
- `n` 切换至下一窗口
- `l` 前后窗口间互相切换
- `w` 通过窗口列表切换窗口
- `,` 重命名当前窗口，便于识别
- `.` 修改当前窗口编号，相当于重新排序
- `f` 在所有窗口中查找关键词，便于窗口多了切换

###### 面板操作

- `"` 将当前面板上下分屏（我自己改成了 `|`）
- `%` 将当前面板左右分屏（我自己改成了 `-`）
- `x` 关闭当前分屏
- `!` 将当前面板置于新窗口,即新建一个窗口,其中仅包含当前面板
- `ctrl+方向键` 以1个单元格为单位移动边缘以调整当前面板大小
- `alt+方向键` 以5个单元格为单位移动边缘以调整当前面板大小
- `q` 显示面板编号
- `o` 选择当前窗口中下一个面板
- `方向键` 移动光标选择对应面板
- `{` 向前置换当前面板
- `}` 向后置换当前面板
- `alt+o` 逆时针旋转当前窗口的面板
- `ctrl+o` 顺时针旋转当前窗口的面板
- `z` 最大化当前所在面板
- `page up` 向上滚动屏幕，q 退出
- `page down` 向下滚动屏幕，q 退出

因为 iTerm2 的支持，很多切换的操作可以直接用鼠标进行。



##### 配置

我们可以先进行一些简单的配置，修改 `~/.tmux.conf` 即可，让整个使用更方便。

```Bash
#-- base --#

set -g default-terminal "screen-256color"
set -g display-time 3000
set -g history-limit 10000
set -g base-index 1
set -g pane-base-index 1
set -s escape-time 0
set -g mouse on

#-- bindkeys --#

# split windows like vim.  - Note: vim's definition of a horizontal/vertical split is reversed from tmux's

unbind s
bind s split-window -v
bind S split-window -v -l 40
bind v split-window -h
bind V split-window -h -l 120

# navigate panes with hjkl
bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R

# key bindings for horizontal and vertical panes
unbind %
bind | split-window -h      # 使用|竖屏，方便分屏
unbind '"'
bind - split-window -v      # 使用-横屏，方便分屏

# swap panes
bind ^u swapp -U
bind ^d swapp -D

bind q killp
bind ^e last

unbind r
bind r source-file ~/.tmux.conf \; display "Configuration Reloaded!"

#-- statusbar --#

set -g status-justify centre
set -g status-left "#[fg=red]s#S:w#I.p#P#[default]"
set -g status-right '[#(whoami)#(date +" %m-%d %H:%M ")]'
set -g status-left-attr bright
set -g status-left-length 120
set -g status-right-length 120
set -g status-utf8 on
set -g status-interval 1
set -g visual-activity on
setw -g monitor-activity on
setw -g automatic-rename off

# default statusbar colors
set -g status-bg colour235 #base02
set -g status-fg colour136 #yellow
set -g status-attr default

# default window title colors
setw -g window-status-fg colour244
setw -g window-status-bg default
#setw -g window-status-attr dim

# active window title colors
setw -g window-status-current-fg colour166 #orange
setw -g window-status-current-bg default
#setw -g window-status-current-attr bright

# window title string (uses statusbar variables)
set -g set-titles-string '#T'
set -g status-justify "centre"
set -g window-status-format '#I #W'
set -g window-status-current-format ' #I #W '

# pane border
set -g pane-active-border-fg '#55ff55'
set -g pane-border-fg '#555555'

# message text
set -g message-bg colour235 #base02
set -g message-fg colour166 #orange

# pane number display
set -g display-panes-active-colour colour33 #blue
set -g display-panes-colour colour166 #orange

# clock
setw -g clock-mode-colour colour64 #green


# 修改进入命令模式按键
# remap prefix to Control + a
# set -g prefix C-a
# unbind C-b
# bind C-a send-prefix
```
