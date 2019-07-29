**collections/namedtuple**

我们知道`tuple`可以表示不变集合，例如，一个点的二维坐标就可以表示成：

```bash
>>> p = (1, 2)
```

但是，看到`(1, 2)`，很难看出这个`tuple`是用来表示一个坐标的；定义一个class又小题大做了，这时，`namedtuple`就派上了用场：

```bash
>>> from collections import namedtuple
>>> Point = namedtuple('Point', ['x', 'y'])
>>> p = Point(1, 2)
>>> p.x
1
>>> p.y
2
```

`namedtuple`是一个函数，它用来创建一个自定义的`tuple`对象，并且规定了`tuple`元素的个数，并可以用属性而不是索引来引用`tuple`的某个元素。这样一来，我们用`namedtuple`可以很方便地定义一种数据类型，它具备tuple的不变性，又可以根据属性来引用，使用十分方便。可以验证创建的`Point`对象是`tuple`的一种子类：

```bash
>>> isinstance(p, tuple)
True
```

类似的，如果要用坐标和半径表示一个圆，也可以用`namedtuple`定义：

```bash
# namedtuple('名称', [属性list]):
Circle = namedtuple('Circle', ['x', 'y', 'r'])
```



**collections/defaultdict**

使用`dict`时，如果引用的Key不存在，就会抛出`KeyError`。如果希望key不存在时，返回一个默认值，就可以用`defaultdict`：

```bash
>>> from collections import defaultdict
>>> dd = defaultdict(lambda: 'N/A')
>>> dd['key1'] = 'abc'
>>> dd['key1'] # key1存在
'abc'
>>> dd['key2'] # key2不存在，返回默认值
'N/A'
```

注意默认值是调用函数返回的，而函数在创建`defaultdict`对象时传入。除了在Key不存在时返回默认值，`defaultdict`的其他行为跟`dict`是完全一样的。



**collections/OrderedDict**

使用`dict`时，Key是无序的。在对`dict`做迭代时，我们无法确定Key的顺序。如果要保持Key的顺序，可以用`OrderedDict`：

```bash
>>> from collections import OrderedDict
>>> d = dict([('a', 1), ('b', 2), ('c', 3)])
>>> d # dict的Key是无序的
{'a': 1, 'c': 3, 'b': 2}
>>> od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
>>> od # OrderedDict的Key是有序的
OrderedDict([('a', 1), ('b', 2), ('c', 3)])
```

注意，`OrderedDict`的Key会按照插入的顺序排列，不是Key本身排序：

```bash
>>> od = OrderedDict()
>>> od['z'] = 1
>>> od['y'] = 2
>>> od['x'] = 3
>>> od.keys() # 按照插入的Key的顺序返回
['z', 'y', 'x']
```

`OrderedDict`可以实现一个FIFO（先进先出）的dict，当容量超出限制时，先删除最早添加的Key：

```python
from collections import OrderedDict
class LastUpdatedOrderedDict(OrderedDict):
    def __init__(self, capacity):
        super(LastUpdatedOrderedDict, self).__init__()
        self._capacity = capacity
    def __setitem__(self, key, value):
        containsKey = 1 if key in self else 0
        if len(self) - containsKey >= self._capacity:
            last = self.popitem(last=False)
            print 'remove:', last
        if containsKey:
            del self[key]
            print 'set:', (key, value)
        else:
            print 'add:', (key, value)
        OrderedDict.__setitem__(self, key, value)
```



**Collections/Counter**

`Counter`是一个简单的计数器，例如，统计字符出现的个数：

```bash
>>> from collections import Counter
>>> c = Counter()
>>> for ch in 'programming':
...     c[ch] = c[ch] + 1
...
>>> c
Counter({'g': 2, 'm': 2, 'r': 2, 'a': 1, 'i': 1, 'o': 1, 'n': 1, 'p': 1})
```

`Counter`实际上也是`dict`的一个子类，上面的结果可以看出，字符`'g'`、`'m'`、`'r'`各出现了两次，其他字符各出现了一次。



##### 随机数

random模块提供一个快速的伪随机数生成器；

- random函数返回一个$[0,1)$范围的随机数；
- uniform函数指定生成随机数的范围；
- seed()函数，使用同一个种子，每次生成的随机数序列都是相同的；
- randint()函数
- randrange()函数
- range()函数
- shuffle()函数
- choice()函数



##### pip

安装指定版本包：`pip install package_name==version`

查看已安装的包：`pip list`

临时改变镜像：`pip install -i http://mirror.path package_name`

永久改变镜像：

```bash
mkdir ~/.pip && cd ~/.pip
vim pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

目前国内比较流行的pip源有：

> [http://pypi.douban.com/](https://link.jianshu.com?t=http://pypi.douban.com/)       豆瓣
>
> [http://pypi.hustunique.com/](https://link.jianshu.com?t=http://pypi.hustunique.com/) 华中理工大学
>
> [http://pypi.sdutlinux.org/](https://link.jianshu.com?t=http://pypi.sdutlinux.org/) 山东理工大学
>
> [http://pypi.mirrors.ustc.edu.cn/](https://link.jianshu.com?t=http://pypi.mirrors.ustc.edu.cn/) 中国科学技术大学
>
> [http://mirrors.aliyun.com/pypi/simple/](https://link.jianshu.com?t=http://mirrors.aliyun.com/pypi/simple/)  阿里云
>
> [https://pypi.tuna.tsinghua.edu.cn/simple/](https://link.jianshu.com?t=https://pypi.tuna.tsinghua.edu.cn/simple/) 清华大学



##### virtualenv

指定python版本：`virtualenv -p /path/to/python3 project_name`

激活虚拟环境：`source project_name/bin/activate`

退出虚拟环境：`deactivate`



##### OpenAI Gym编程指南

**基础**：强化学习中有两个概念：环境（外部世界）和代理（所写的算法）。代理将行动发送给环境，而环境则用观测和激励来回应。核心Gym接口是[Env](https://github.com/openai/gym/blob/master/gym/core.py)，是统一环境接口。下面是应当知晓的`Env`方法：

- `reset(self)`：重置环境状态，返回观测；
- `step(self, action)`：将环境向前执行一步，返回观测、激励、完成、信息；
- `render(self, model='human', close=False)`：渲染环境的一帧。

**安装**：先安装依赖包

```bash
apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```

然后

```bash
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[all]'
```





##### 安装mupen64plus模拟器

安装sdl2.0、libpng、freetype、zlib库

```bash
sudo apt-get install libsdl2-2.0-0 libsdl2-dev\
                     libpng12-0 libpng12-dev\
                     libfreetype6 libfreetype6-dev\
                     zlib1g zlib1g-dev
```

安装编译所需的工具：

```bash
sudo apt-get install nasm
```

安装mupen64plus模拟器



##### 安装Mupen64Plus Gym环境

安装依赖包：

```bash
sudo apt-get install libjson-c-dev libjsoncpp-dev
sudo pip install numpy gym pyyaml termcolor mss
```

安装mupen64plus-input-bot

```bash
#!/bin/bash
mkdir mupen64plus-src && cd "$_"
git clone https://github.com/mupen64plus/mupen64plus-core
git clone https://github.com/kevinhughes27/mupen64plus-input-bot
cd mupen64plus-input-bot
make all
sudo make install
```



##### 包安装路径

- dist-packages是Debian系统特定的，像Ubuntu，由Debian包管理器安装的模块就会安装到这个路径中：

  ```bash
  /usr/lib/pythonVersion/dist-packages
  ```

  即用`sudo apt-get install`安装的包都在这个目录中。

- 因为`pip`和`asy_install`是从包管理器安装的，因此也用dost-packages，但存放在：

  ```bash
  /usr/local/lib/pythonVersion/dist-packages
  ```

  因此用`sudo pip install`安装的包在这个目录中。

- 由[Debian Python Wiki](http://wiki.debian.org/Python)的描述，从手动从源码安装Python，它使用site-packages这个目录，这使得两种安装保持分离，尤其是Debian和Ubuntu的许多系统工具都依赖于Python的系统版本。

  那么手动从源码安装的包也是存储在这个目录里面吗？

##### 寻找包

- **sys.path**：python通过搜寻在`sys.path`列表中的路径引入包，它会查找安装在这些位置的任意包。

  ```python
  import sys, os
  print '\n'.join(sys.path)
  ```

  根据python docs，`sys.path`由当前工作目录、接着是`PYTHONPATH`环境变量列出的目录、然后是由`site`模块控制的安装依赖的默认路径组成。

  `site`模块会在启动Python时自动导入，这种自动导入也可以使用解释器的`-s`选项禁止；

  **操作sys.path**：可以在python会话中操作`sys.path`来改变寻找包的方式。比如：

  ```python
  sys.path.append(home_dir)
  ```

  **模块的`__file__`属性**：在导入模块时，通常可通过模块的`__file__`属性来查看模块在文件系统中的位置：

  ```python
  import numpy
  print numpy.__file__
  ```

  但不适用于静态链接到解释器的C模块，因此`print sys__file__`就无效。

- **imp模块**：python通过`imp`模块来显示其整个导入系统。`imp.find_module`可以用来寻找模块：

  ```python
  import imp
  print imp.find_module('numpy')
  ```

  也可以使用`imp.load-source`来导入任意python源码。

  ```python
  home_dir = os.path.expanduser("~")
  my_module_file = os.path.join(home_dir, "hi.py")
  hi = imp.load_source('hi', my_module_file)
  ```

  传输`'hi'`仅是设定模块的`__name__`属性设定。

- **Ubuntu Python**：ubuntu系统自带的python位于`/usr/bin/python`，而从源码编译的python则位于`/usr/local/bin/python`中。使用ubuntu的python，包安装在`/usr/local/lib/python/dist-packages`中，而新安装的python则将包装在`/usr/local/lib/python/site-packages`。在Ubuntu的python中会将`sys.path`路径硬编码进`site`模块。



如果想定义成私有属性，则需在属性名前加2个下划线'__'。

在Python中有一些内置的方法，这些方法命名都有比较特殊的地方（其方法名以2个下划线开始然后以2个下划线结束）。类中最常用的就是构造方法和析构方法。 　　构造方法**init**(self,....)：在生成对象时调用，可以用来进行一些初始化操作，不需要显示去调用，系统会默认去执行。构造方法支持重载，如果用户自己没有重新定义构造方法，系统就自动执行默认的构造方法。 　　析构方法**del**(self)：在释放对象时调用，支持重载，可以在里面进行一些释放资源的操作，不需要显示调用。 　　还有其他的一些内置方法，比如 **cmp**( ), **len( )**等。下面是常用的内置方法：

```
内置方法     说明
 __init__(self,...)     初始化对象，在创建新对象时调用
 __del__(self)     释放对象，在对象被删除之前调用
 __new__(cls,*args,**kwd)     实例的生成操作
 __str__(self)     在使用print语句时被调用
 __getitem__(self,key)     获取序列的索引key对应的值，等价于seq[key]
 __len__(self)     在调用内联函数len()时被调用
 __cmp__(stc,dst)     比较两个对象src和dst
 __getattr__(s,name)     获取属性的值
 __setattr__(s,name,value)     设置属性的值
 __delattr__(s,name)     删除name属性
 __getattribute__()     __getattribute__()功能与__getattr__()类似
 __gt__(self,other)     判断self对象是否大于other对象
 __lt__(slef,other)     判断self对象是否小于other对象
 __ge__(slef,other)     判断self对象是否大于或者等于other对象
 __le__(slef,other)     判断self对象是否小于或者等于other对象
 __eq__(slef,other)     判断self对象是否等于other对象
 __call__(self,*args)     把实例对象作为函数调用
```



argparse是Python标准库中的命令行解析模块。下面是一个示例`prog.py` ：

```python
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')
args = parser.parse_args()
print(args.accumulate(args.integers))
```

运行结果：

```bash
$ python prog.py 1 2 3 4
4
$ python prog.py 1 2 3 4 --sum
10
```



##### 1.创建解析器

使用argparse的第一步是创建`ArgumentParser`解析器对象。

```python
parser = argparse.ArgumentParser(description='Process some integers.')
```

所有参数都必须以关键词参数传递，其参数概括如下：

- `prog`：程序名（默认：`sys.argv[0]`）；
- `usage`：描述程序使用方法的字符串（默认：从添加到解析器的参数产生）；
- `description`：显示在参数帮助前的文本（默认：无）；
- `epilog`：显示在参数帮助后的文本（默认：无）；
- `parent`：两个参数也须包含进来的`ArgumentParser`对象列表；
- `formatter_class`：定制帮助输出的类；
- `prefix_chars`：可选参数前缀的字符集（默认：'-'）；
- `argument_default`：参数的全局默认值（默认：`None`）；
- `conflict_handler`：解决冲突选项的策略（通常不必）；
- `add_help`：添加`-h/--help`选项到解析器（默认：`True`）；
- `allow_abbrev`：在缩写清晰时，允许缩写长选项（默认：`True`）。



##### 2.添加参数

```python
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')
```

`ArgumentParser.add_argument()`定义如何解析单个命令行参数，它的参数有：

- `name`或`flags`：名称或选项字符串列表，如`foo`或`-f, --foo`；函数`add_argument`必须知道是否需要可选参数或位置参数，因此最先传递的参数必须是一些标识，或单个参数名，比如：

  ```python
  >>> parser.add_argument('-f', '--foo')
  >>> parser.add_argument('bar')
  >>> parser.parse_args(['BAR', '--foo', 'FOO']) #这也是在交互环境中使用argparse的方式
  Namespace(bar='BAR', foo='FOO')
  ```

  当调用`parse_args()`函数时，会将`-`前缀识别为可选，其余为位置。

- `action`：当在命令行遇到这个参数时采取的基本行为类型；`ArgumentParser`对象用行动连接命令行参数，提供的行为有：

  - `store`：存储参数值，为默认行为；

  - `store_const`：存储由关键词`const`指明的值，通常与确定某些类别或标识的可选参数使用，如

    ```python
    >>> parser.add_argument('--foo', action='store_const', const=42)
    >>> parser.parse_args(['--foo'])
    Namespace(foo=42)
    ```

  - `store_true`和`store_false`：`store_const`用于存储`True`或`False`的特例，另外还分别创建默认值`False`和`True`：

    ```python
    >>> parser.add_argument('--foo', action='store_true')
    >>> parser.add_argument('--bar', action='store_false')
    >>> parser.add_argument('--baz', action='store_false')
    >>> parser.parse_args('--foo --bar'.split())
    Namespace(foo=True, bar=False, baz=True)
    ```

  - `append`：存储一个列表，追加每个参数值，在允许一个选项被知名多次时很有用，如：

    ```python
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument('--foo', action='append')
    >>> parser.parse_args('--foo 1 --foo 2'.split())
    Namespace(foo=['1', '2'])
    ```

  - `append_const`：存储一个列表，追加每个由`const`关键词指明的值（注意`const`关键词默认为`None`），在多个参数需要存储常数到同一列表时很有用，如：

    ```python
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument('--str', dest='types', action='append_const', const=str)
    >>> parser.add_argument('--int', dest='types', action='append_const', const=int)
    >>> parser.parse_args('--str --int'.split())
    Namespace(types=[<class 'str'>, <class 'int'>])
    ```

  - `count`：计数一个关键词出现次数，比如要增加冗余程度：

    ```python
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument('--verbose', '-v', action='count')
    >>> parser.parse_args(['-vvv'])
    Namespace(verbose=3)
    ```

  - `help`：打印所有选项的完整帮助信息然后退出，默认添加自动添加到解析器中。

  - `version`：期望关键词`version=`，调用时打印版本信息并退出：

    ```python
    >>> parser = argparse.ArgumentParser(prog='PROG')
    >>> parser.add_argument('--version', action='version', version='%(prog)s 2.0')
    >>> parser.parse_args(['--version'])
    PROG 2.0
    ```

  也可以通过传递一个`Action`子类或其他实现同样接口的对象来自己指定任意行为，如：

  ```python
  >>> class FooAction(argparse.Action):
  ...     def __init__(self, option_strings, dest, nargs=None, **kwargs):
  ...         if nargs is not None:
  ...             raise ValueError("nargs not allowed")
  ...         super(FooAction, self).__init__(option_strings, dest, **kwargs)
  ...     def __call__(self, parser, namespace, values, option_string=None):
  ...         print('%r %r %r' % (namespace, values, option_string))
  ...         setattr(namespace, self.dest, values)
  ...
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('--foo', action=FooAction)
  >>> parser.add_argument('bar', action=FooAction)
  >>> args = parser.parse_args('1 --foo 2'.split())
  Namespace(bar=None, foo=None) '1' None
  Namespace(bar='1', foo=None) '2' '--foo'
  >>> args
  Namespace(bar='1', foo='2')
  ```

  更多信息可见[`Action`](https://docs.python.org/3/library/argparse.html#argparse.Action)。

- `nargs`：要消耗的命令行参数个数；`ArgumentParser`对象通常将单个命令行参数与单个行为连结，`nargs`参数将不同数目的命令行参数与单个行为连结，支持的值有：

  - `N`，N是一个整数，N个命令行的参数会收集到一个列表，如：

    ```python
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument('--foo', nargs=2)
    >>> parser.add_argument('bar', nargs=1)
    >>> parser.parse_args('c --foo a b'.split())
    Namespace(bar=['c'], foo=['a', 'b'])
    ```

    注意`nargs=1`产生只有一个项目的列表，这与默认不同。

  - `?`：若可能会消耗命令行的一个参数并产生单个项目。若命令行中没有出现参数，会产生默认值。注意对可选参数，若有选项字串而无跟随命令行参数，则会产生一个常量值。比如：

    ```python
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument('--foo', nargs='?', const='c', default='d')
    >>> parser.add_argument('bar', nargs='?', default='d')
    >>> parser.parse_args(['XX', '--foo', 'YY'])
    Namespace(bar='XX', foo='YY')
    >>> parser.parse_args(['XX', '--foo'])
    Namespace(bar='XX', foo='c')
    >>> parser.parse_args([])
    Namespace(bar='d', foo='d')
    ```

    `nargs=？`的一种更常用法是允许可选的输入和输出文件：

    ```python
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument('infile', nargs='?', type=argparse.FileType('r'),
    ...                     default=sys.stdin)
    >>> parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
    ...                     default=sys.stdout)
    >>> parser.parse_args(['input.txt', 'output.txt'])
    Namespace(infile=<_io.TextIOWrapper name='input.txt' encoding='UTF-8'>,
              outfile=<_io.TextIOWrapper name='output.txt' encoding='UTF-8'>)
    >>> parser.parse_args([])
    Namespace(infile=<_io.TextIOWrapper name='<stdin>' encoding='UTF-8'>,
              outfile=<_io.TextIOWrapper name='<stdout>' encoding='UTF-8'>)
    ```

  - `*`：出现的所有命令行参数都收集到一个列表中，可以没有参数。通常超过一个带`nargs=*`的位置参数并没有什么意义，但多个带`nargs=*`的可选参数是可以的：

    ```python
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument('--foo', nargs='*')
    >>> parser.add_argument('--bar', nargs='*')
    >>> parser.add_argument('baz', nargs='*')
    >>> parser.parse_args('a b --foo x y --bar 1 2'.split())
    Namespace(bar=['1', '2'], baz=['a', 'b'], foo=['x', 'y'])
    ```

    `+`：将所有出现的参数收集到一个列表中，必须至少有一个。

    ```python
    >>> parser = argparse.ArgumentParser(prog='PROG')
    >>> parser.add_argument('foo', nargs='+')
    >>> parser.parse_args(['a', 'b'])
    Namespace(foo=['a', 'b'])
    >>> parser.parse_args([])
    usage: PROG [-h] foo [foo ...]
    PROG: error: the following arguments are required: foo
    ```

  - `argparse.REMAINDER`：所有剩余命令行参数收集到一个列表，通常在置换到其他命令行功能时使用：

    ```python
    >>> parser = argparse.ArgumentParser(prog='PROG')
    >>> parser.add_argument('--foo')
    >>> parser.add_argument('command')
    >>> parser.add_argument('args', nargs=argparse.REMAINDER)
    >>> print(parser.parse_args('--foo B cmd --arg1 XX ZZ'.split()))
    Namespace(args=['--arg1', 'XX', 'ZZ'], command='cmd', foo='B')
    ```

  若未提供`nargs`关键词参数，消费的参数个数由`action`决定，通常是消耗单个命令行参数并产生单项（而非列表）。

- `const`：一些`action`和`nargs`选择需要的常数，即存储并非从命令行读取但为多个`ArgumentParser`所要求的常量，最常见的两种用法是：

  - 当`add_argument()`用`action=store_const`或`action=append_const`调用时，这些行为将`const`值添加到`parse_args()`返回对象的一个特征中；
  - 当`add_argument()`用可选字串（像`-f`或`-foo`）和`nargs=?`调用时，创建一个后跟0或1个命令行参数的可选参数。在解析命令行参数时若遇到选项字串而没跟参数，就会假设是常量值。

  当使用`store_const`、`append_const`时，，必须给定`const`关键词参数，对其他行为，则默认为`None`。

- `default`：若参数在命令行缺失产生的值；所有可选参数和一些位置参数能被省略，默认值为`None`的`default`关键词参数指明当这些命令行参数未出现时的值。对可选参数，`default`值在选项字串未出现时

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('--foo', default=42)
  >>> parser.parse_args(['--foo', '2'])
  Namespace(foo='2')
  >>> parser.parse_args([])
  Namespace(foo=42)
  ```

  若`default`是字串，解析器将其作为命令行参数解析，特别是，若提供`type`参数，则在设置命名空间返回值特征时，解析器会应用转换。即：

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('--length', default='10', type=int)
  >>> parser.add_argument('--width', default=10.5, type=int)
  >>> parser.parse_args()
  Namespace(length=10, width=10.5)
  ```

  对带`nargs`的位置参数参数就等价于`?`或`*`，`default`值在没有命令行参数出现的时候使用：

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('foo', nargs='?', default=42)
  >>> parser.parse_args(['a'])
  Namespace(foo='a')
  >>> parser.parse_args([])
  Namespace(foo=42)
  ```

  提供`default=argparse.SUPPRESS`使命令行未出现时，没有特征添加：

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('--foo', default=argparse.SUPPRESS)
  >>> parser.parse_args([])
  Namespace()
  >>> parser.parse_args(['--foo', '1'])
  Namespace(foo='1')
  
  ```

- `type`：命令行参数需要被转换的类型；`ArgumentParser`对象默认将命令行参数作为字串读入，而`type`关键词参数则允许执行任意类型检查和转换命令行内建类型和函数能直接作为`type`参数的的值：

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('foo', type=int)
  >>> parser.add_argument('bar', type=open)
  >>> parser.parse_args('2 temp.txt'.split())
  Namespace(bar=<_io.TextIOWrapper name='temp.txt' encoding='UTF-8'>, foo=2)
  
  ```

  为方便使用多种类型文件，argparse提供工厂函数`FileType`取`open()`函数的`mode=`、`bufsize=`、`encoding=`和`errors=`参数：

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('bar', type=argparse.FileType('w'))
  >>> parser.parse_args(['out.txt'])
  Namespace(bar=<_io.TextIOWrapper name='out.txt' encoding='UTF-8'>)
  
  ```

  `type=`可以去任何输入单个字串返回转换后类型的可调用函数：

  ```python
  >>> def perfect_square(string):
  ...     value = int(string)
  ...     sqrt = math.sqrt(value)
  ...     if sqrt != int(sqrt):
  ...         msg = "%r is not a perfect square" % string
  ...         raise argparse.ArgumentTypeError(msg)
  ...     return value
  ...
  >>> parser = argparse.ArgumentParser(prog='PROG')
  >>> parser.add_argument('foo', type=perfect_square)
  >>> parser.parse_args(['9'])
  Namespace(foo=9)
  >>> parser.parse_args(['7'])
  usage: PROG [-h] foo
  PROG: error: argument foo: '7' is not a perfect square
  
  ```

- `choices`：参数允许值的容器；一些命令行参数仅能从一个限制集合选取，这能通过传递一个`choices`关键词参数的容器对象解决，当解析命令行时会检查参数值：

  ```python
  >>> parser = argparse.ArgumentParser(prog='game.py')
  >>> parser.add_argument('move', choices=['rock', 'paper', 'scissors'])
  >>> parser.parse_args(['rock'])
  Namespace(move='rock')
  >>> parser.parse_args(['fire'])
  usage: game.py [-h] {rock,paper,scissors}
  game.py: error: argument move: invalid choice: 'fire' (choose from 'rock',
  'paper', 'scissors')
  
  ```

  相比于`type`，可能是更方便的类型检查器。注意`choice`容器中的包含值是在`type`的转换以后执行的，因此`choice`容器中的对象类型必须匹配`type`所确定的。

- `required`：命令行选项是否能被省略（仅可选参数）；通常假定`-f`和`--bar`这样的标识总是可以省略，若要使其必须，使`required=True`：

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('--foo', required=True)
  >>> parser.parse_args(['--foo', 'BAR'])
  Namespace(foo='BAR')
  >>> parser.parse_args([])
  usage: argparse.py [-h] [--foo FOO]
  argparse.py: error: option --foo is required
  
  ```

  通常认为这是一个糟糕的方式。

- `help`：参数作用的简单描述；`help`值是包含简单参数描述的字串，当在命令行中使用`-h`或`--help`时，会显示出来：

  ```python
  >>> parser = argparse.ArgumentParser(prog='frobble')
  >>> parser.add_argument('--foo', action='store_true',
  ...                     help='foo the bars before frobbling')
  >>> parser.add_argument('bar', nargs='+',
  ...                     help='one of the bars to be frobbled')
  >>> parser.parse_args(['-h'])
  usage: frobble [-h] [--foo] bar [bar ...]
  
  positional arguments:
   bar     one of the bars to be frobbled
  
  optional arguments:
   -h, --help  show this help message and exit
   --foo   foo the bars before frobbling
  
  ```

  `help`字串能包含多种格式化指定器来避免像程序名或参数默认值的重复，包含程序名`%(prog)s`和大多数关键词参数，如`%(default)s`、`%(type)s`等：

  ```python
  >>> parser = argparse.ArgumentParser(prog='frobble')
  >>> parser.add_argument('bar', nargs='?', type=int, default=42,
  ...                     help='the bar to %(prog)s (default: %(default)s)')
  >>> parser.print_help()
  usage: frobble [-h] [bar]
  
  positional arguments:
   bar     the bar to frobble (default: 42)
  
  optional arguments:
   -h, --help  show this help message and exit
  
  ```

  因为帮助字串支持%-格式，若需要字面的`%`，需用`%%`来转义。`argparse`支持特定选项的帮助静默，通过设置`help`值为`argparse.SUPRESS`：

  ```python
  >>> parser = argparse.ArgumentParser(prog='frobble')
  >>> parser.add_argument('--foo', help=argparse.SUPPRESS)
  >>> parser.print_help()
  usage: frobble [-h]
  
  optional arguments:
    -h, --help  show this help message and exit
  
  ```

- `metavar`：参数在用法消息中的名称；当`ArgumentParser`产生帮助信息的时候，需要一些方法来引用每个希望的参数。默认使用`dest`值作为咩个对象的名称。对位置参数行为，默认直接使用`dest`值，对可选参数行为，默认使用大写的`dest`值。因此`dest='bar'`的单个位置参数会被引用为`bar`，而单个后跟一个命令行参数的可选参数`--foo`会被引用为`FOO`：

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('--foo')
  >>> parser.add_argument('bar')
  >>> parser.parse_args('X --foo Y'.split())
  Namespace(bar='X', foo='Y')
  >>> parser.print_help()
  usage:  [-h] [--foo FOO] bar
  
  positional arguments:
   bar
  
  optional arguments:
   -h, --help  show this help message and exit
   --foo FOO
  
  ```

  使用`metavar`可制定替代名：

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('--foo', metavar='YYY')
  >>> parser.add_argument('bar', metavar='XXX')
  >>> parser.parse_args('X --foo Y'.split())
  Namespace(bar='X', foo='Y')
  >>> parser.print_help()
  usage:  [-h] [--foo YYY] XXX
  
  positional arguments:
   XXX
  
  optional arguments:
   -h, --help  show this help message and exit
   --foo YYY
  
  ```

  注意`metavar`仅改变展示的名字，`parse_args()`内的特征名依然由`dest`值决定：

  ```python
  >>> parser = argparse.ArgumentParser(prog='PROG')
  >>> parser.add_argument('-x', nargs=2)
  >>> parser.add_argument('--foo', nargs=2, metavar=('bar', 'baz'))
  >>> parser.print_help()
  usage: PROG [-h] [-x X X] [--foo bar baz]
  
  optional arguments:
   -h, --help     show this help message and exit
   -x X X
   --foo bar baz
  
  ```

- `dest`：添加到`parse_args()`方法返回对象中特征的名称； 大多数`ArgumentParser`行为添加一些值作为`parse_args()`返回对象的特征，特征的名称由`dest`关键词参数确定。对位置参数，`dest`提供为`add_argument()`的地一个参数：

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('bar')
  >>> parser.parse_args(['XXX'])
  Namespace(bar='XXX')
  
  ```

  对可选参数行为，`dest`的值为第一个长选项字串去掉前置`--`，若无长选项字串，则从地一个短选项字串去掉前置`-`获得。任何内部`-`都会被转化为`_`字串：

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('-f', '--foo-bar', '--foo')
  >>> parser.add_argument('-x', '-y')
  >>> parser.parse_args('-f 1 -x 2'.split())
  Namespace(foo_bar='1', x='2')
  >>> parser.parse_args('--foo 1 -y 2'.split())
  Namespace(foo_bar='1', x='2')
  
  ```

  `dest`允许定制提供的特征名：

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('--foo', dest='bar')
  >>> parser.parse_args('--foo XXX'.split())
  Namespace(bar='XXX')
  
  ```

- `Action`类：`Action`类实现`Action`API，后者是一个返回一个处理来自命令参数的可调用函数的可调用函数。一个遵循这样API的对象可以作为`action`参数传递到`add_argument()`。

  `class argparse.Action(option_strings, dest, nargs=None, const=None, default=None, type=None, choices=None, required=False, help=None, metavar=None)`

  `Action`对象被`ArgumentParser`用于表示从一个或多个来自命令行字串解析单个参数的信息，必须接受两个位置参数加传递到`ArgumentParser.add_argument()`的任何关键词参数，除了`action`本身。

  `Action`实例（或任何到`action`的返回值）应该有定义的`dest`、`option_strings`、`default`、`type`、`required`、`help`等特征。最简单的确保定义这些特征的方法就是调用`Action.__init__`。

  `Action`实例应该是可调用的，因此必须覆盖`__call__`方法，其后者应该接受4个参数：

  - `parser`：包含这个行为的`ArgumentParser`对象；
  - `namespace`：`Namespace`对象会被`parse_args()`返回，大多数行为使用`setattr()`来添加特征到此对象。
  - `values`：关联的命令行参数，带任何类型转换应用。类型转换由到`add_argument()`的`type`关键词参数指定；
  - `option_string`：可选字串用于调用这个行为，`option_string`参数是可选的，在行为关联到一个位置参数时会缺席。

  `__call__`方法可以执行任意行为，但通常是基于`dest`和`values`在`namespace`设置特征。



##### 3.解析参数

`ArgumentParser.parse_args(args=None, namespace=None)`将参数字串转换为对象并将其赋为命名空间的特征。有两个参数：

- `args`：需要解析的字串列表，默认从`sys.argv`取得；
- `namespace`：要取得这些特征的对象，默认是新的空`Namespace`对象。

前面对`add_argument()`的调用确定创建什么样的对象以及如何赋值。

- 选项值语义：

  `parse_args()`支持多种指定选项值的方法，最简单的是选项和值作为两个参数传递：

  ```python
  >>> parser = argparse.ArgumentParser(prog='PROG')
  >>> parser.add_argument('-x')
  >>> parser.add_argument('--foo')
  >>> parser.parse_args(['-x', 'X'])
  Namespace(foo=None, x='X')
  >>> parser.parse_args(['--foo', 'FOO'])
  Namespace(foo='FOO', x=None)
  
  ```

  对长选项，选项和值可以作为单个命令行参数传递，用`=`分隔：

  ```python
  >>> parser.parse_args(['--foo=FOO'])
  Namespace(foo='FOO', x=None)
  
  ```

  对短选项，选项和值可以拼接起来：

  ```python
  >>> parser.parse_args(['-xX'])
  Namespace(foo=None, x='X')
  
  ```

  多个短选项可以连在一起仅用单个`-`前缀，只要仅最后一个（或没有）选项需要值：

  ```python
  >>> parser = argparse.ArgumentParser(prog='PROG')
  >>> parser.add_argument('-x', action='store_true')
  >>> parser.add_argument('-y', action='store_true')
  >>> parser.add_argument('-z')
  >>> parser.parse_args(['-xyzZ'])
  Namespace(x=True, y=True, z='Z')
  
  ```

- 包含`-`的参数

  有些情况本身很模糊，比如命令行参数`-1`既可以是指定选项也也可以是提供位置参数。``parse_arg()`方法这里会很小心：若看起来像负数，位置参数仅能以`-`开始，并且解析器没有选项看起来是负数：

  ```python
  >>> parser = argparse.ArgumentParser(prog='PROG')
  >>> parser.add_argument('-x')
  >>> parser.add_argument('foo', nargs='?')
  
  >>> # no negative number options, so -1 is a positional argument
  >>> parser.parse_args(['-x', '-1'])
  Namespace(foo=None, x='-1')
  
  >>> # no negative number options, so -1 and -5 are positional arguments
  >>> parser.parse_args(['-x', '-1', '-5'])
  Namespace(foo='-5', x='-1')
  
  >>> parser = argparse.ArgumentParser(prog='PROG')
  >>> parser.add_argument('-1', dest='one')
  >>> parser.add_argument('foo', nargs='?')
  
  >>> # negative number options present, so -1 is an option
  >>> parser.parse_args(['-1', 'X'])
  Namespace(foo=None, one='X')
  
  >>> # negative number options present, so -2 is an option
  >>> parser.parse_args(['-2'])
  usage: PROG [-h] [-1 ONE] [foo]
  PROG: error: no such option: -2
  
  >>> # negative number options present, so both -1s are options
  >>> parser.parse_args(['-1', '-1'])
  usage: PROG [-h] [-1 ONE] [foo]
  PROG: error: argument -1: expected one argument
  
  ```

  若有位置参数必须以`-`开始并且看起来并不像附属，可以插入伪参`--`告知`parse_args()`气候的都是一个位置参数：

  ```python
  >>> parser.parse_args(['--', '-f'])
  Namespace(foo='-f', one=None)
  
  ```

- 参数简写（前缀匹配）

  `parse_args()`默认允许长选项简写为一个前缀，只要简写非歧义（前缀匹配单个选项）：

  ```python
  >>> parser = argparse.ArgumentParser(prog='PROG')
  >>> parser.add_argument('-bacon')
  >>> parser.add_argument('-badger')
  >>> parser.parse_args('-bac MMM'.split())
  Namespace(bacon='MMM', badger=None)
  >>> parser.parse_args('-bad WOOD'.split())
  Namespace(bacon=None, badger='WOOD')
  >>> parser.parse_args('-ba BA'.split())
  usage: PROG [-h] [-bacon BACON] [-badger BADGER]
  PROG: error: ambiguous option: -ba could match -badger, -bacon
  
  ```

  这个特征可以通过设置`allow_abbrev`为`False`来关闭。

- 超越`sys.argv`

  若希望就诶系非`sys.argv`的参数，可以通过传递一个字串列表到`parse_args()`实现。

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument(
  ...     'integers', metavar='int', type=int, choices=range(10),
  ...     nargs='+', help='an integer in the range 0..9')
  >>> parser.add_argument(
  ...     '--sum', dest='accumulate', action='store_const', const=sum,
  ...     default=max, help='sum the integers (default: find the max)')
  >>> parser.parse_args(['1', '2', '3', '4'])
  Namespace(accumulate=<built-in function max>, integers=[1, 2, 3, 4])
  >>> parser.parse_args(['1', '2', '3', '4', '--sum'])
  Namespace(accumulate=<built-in function sum>, integers=[1, 2, 3, 4])
  
  ```

- Namespace对象

  `class argparse.Namespace`，默认由`parse_args()`使用来创建保存特征对象并返回；这个类特意被设得很简单，只是一个可读字串表达子类的对象。若希望有像字典的特征视点，可以使用`vars()`：

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('--foo')
  >>> args = parser.parse_args(['--foo', 'BAR'])
  >>> vars(args)
  {'foo': 'BAR'}
  
  ```

  也可以使一个`ArgumentParser`将特征赋到一个已有对象而非而非一个新的`Namespace`对象。可以通过`namesapce=`关键词指定：

  ```python
  >>> class C:
  ...     pass
  ...
  >>> c = C()
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('--foo')
  >>> parser.parse_args(args=['--foo', 'BAR'], namespace=c)
  >>> c.foo
  'BAR'
  
  ```

##### 4.其他效用

- 子命令：`ArgumentParser.add_subparsers([title][, description][, prog][, parser_class][, action][, option_string][, dest][, help][, metavar])`

  很多程序将其功能分到一些子命令中，比如`svn`可以调用像`svn checkout`、`svn update`、`svn commit`等子命令。这样分割功能在程序执行不同功能需要不同类型命令行参数时非常有用。`ArgumentParser`通过`add_subparsers()`方法支持这样子命令的创建。通常此方法被无参调用并返回特殊的行为对象。这个对象有单个方法`add_parser()`，后者输入命令名和任意`ArgumentParser`构造器函数，返回一个`ArgumentParser`对象。其参数描述为：

  - `title`：在帮助输出中子解析器的 标题。若提供描述默认是

- 部分解析：`ArgumentParser.parse_known_args(args=None, namespace=None)`，有时一个脚本仅解析命令行参数的一些，



#### Sacred

#### 1. 快速入门

##### Hello World：

```python
from sacred import Experiment
ex = Experiment()
@ex.automain
def my_main():
    print('Hello world!')

```

这里做了3件事：

- 从`sacred`导入`Experiment`；
- 创建experiment实例`ex`；
- 用`@ex.automain`用装饰希望运行的函数；

experiment可以从命令行运行：

```bash
> python h01_hello_world.py
INFO - 01_hello_world - Running command 'my_main'
INFO - 01_hello_world - Started
Hello world!
INFO - 01_hello_world - Completed after 0:00:00

```

这个experiment已经有了完整的命令行界面，可以用来控制日志层次或在一个数据库中自动保存关于运行的信息，但没有配置的话(configuration)的话很受限。

##### 首个Configuration

添加一些配置到程序：

```python
from sacred import Experiment
ex = Experiment('hello_config')

@ex.config
def my_config():
    recipient = "world"
    message = "Hello %s!" % recipient

@ex.automain
def my_main(message):
    print(message)

```

这里做了：

- 添加`my_config`函数并用`@ex.config`装饰；
- 在函数内定义变量`message`；
- 添加`message`参数到函数`main`并使用。

当运行experiment时，sacred会运行`my_config`函数并将所有变量从其局部视野放进experiment中。然后哪里所有定义的变量都能用在`main`函数中。可以通过命令行界面打印配置：

```python
> python hello_config.py print_config
INFO - hello_config - Running command 'print_config'
INFO - hello_config - started
Configuration:
  message = 'Hello world!'
  recipient = 'world'
  seed = 746486301
INFO - hello_config - finished after 0:00:00.

```

注意sacred如何挑选`message`和`recipent`变量。既然实验有了配置，可以从命令行界面改变：

```python
> python hello_config.py with recipient="that is cool"
INFO - hello_config - Running command 'my_main'
INFO - hello_config - started
Hello that is cool!
INFO - hello_config - finished after 0:00:00.

```



#### 2. Experiment总览

`Experiment`是sacred框架的中心类。

##### 创建Experiment

创建一个`Experiment`只需将其初始化并添加main方法：

```python
from sacred import Experiment
ex = Experiment()

@ex.main
def my_main():
    pass

```

用`@ex.main`装饰的函数就是试验的主函数，运行试验就会被执行，也用于确定试验的源文件。这里更推介使用`@automain`，若执行文件它会自动运行试验，等价于下面：

```python
@ex.main
def my_main():
    pass

if __name__ == '__main__':
    ex.run_commandline()

```

需要将`automain`函数放在文件的最后，才会这样的效果，否则任何在其下面的会在试验运行中未定义。

##### 运行Experiment

最简单的运行试验的方式是通过命令行，这要求使用`automain`（或等家形式）。也可以直接在python中运行，这在希望用不同配置运行多次时很有效。假定要运行的试验在文件`my_experiment.py`中，可以将其导入并运行：

```python
from my_experiment import ex
r = ex.run()

```

在交互环境中sacred试验默认会失败，这是一种特意的安全方法，若需要可通过传递`interactive=True`关闭这种防卫：

```python
ex = Experiment('jupyter_ex', interactive=True)

```

`run`函数会接收`config_update`来指定这次运行的配置如何变化。它需要是（可能嵌套）包含所有希望更新值的字典：

```python
from my_experiment import ex
r = ex.run(config_update={'foo':23})

```

每次运行一个`Experiment`（这也是`ex.run`返回的对象）时会创建一个`Run`对象，它保存一些关于运行的信息（比如最终配置和后面的结果），并对后面所有观测Experiment事件的发行负责。在`Experiment`运行时可以在任何`Captured Functions`内通过接收特殊的`_run`参数来访问它。这也用于保存定制信息。

##### 配置

有多种增加配置到试验的方法，最但的是通过Config范围：

```python
@ex.config
def my_config():
    foo = 42
    bar = 'baz'

```

这个函数的局部变量会被收集形成试验的配置，这样定义配置的方式就能充分利用python的力量，参数甚至能够互相依赖。注意：仅可JSON序列化（比如数字、字串、列表、元组、字典）的变量才能成为配置的一部分，其他变量会被忽略。也可以整个从文件中加载配置。

##### 捕获函数

要使用一个配置值，需要捕获一个函数并将其接收为一个参数。每次调用此函数时Sacred会尝试从配置中填补缺失的参数。

```python
from sacred import Experiment
ex = Experiment('my_experiment')

@ex.config
def my_config():
    foo = 42
    bar = 'baz'

@ex.capture
def some_function(a, foo, bar=10):
    print(a, foo, bar)

@ex.main
def my_main():
    some_function(1, 2, 3)     #  1  2   3
    some_function(1)           #  1  42  'baz'
    some_function(1, bar=12)   #  1  42  12
    some_function()            #  TypeError: missing value for 'a'

```

注意配置值优先于默认值。

##### 观测一个试验

Sacred中的试验收集很多关于运行的信息，比如：

- 开始和结束的时间；
- 使用的配置；
- 结果或发生的任何错误；
- 运行机器的基本信息；
- 试验依赖的包及其版本信息；
- 所有导入的本地源文件；
- 用`ex.open_resource`打开的文件；
- 用`ex.add_artifact`添加的文件。

可以使用observer界面获取这些信息：

```python
from sacred.observers import MongoObserver
ex.observer.append(MongoObserver.create())

```

目前`MongoObserver`是sacred仅有的舶来observer，它连接到一个MongoDB并将所有信息放入一个被称为`experiments`的收集中。也可以从命令行添加observer：

```bash
>>>python my_experiment.py -m my_database

```

##### 捕获stdout/stderr

Sacred尝试不或所有输出并将信息传输到observer。这种行为是可配置的，有三种不同的模式：

- `no`模式：不捕获任何输出，若未添加observer到试验，这是默认行为；
- `sys`模式：sacred捕获所有写到`sys.stdout`和`sys.stderr`的输出，比如`print`语句、栈追踪以及日志等；但不会捕获系统调用、C-扩展或子进程。这是windows的默认行为；
- `fd`模式：捕获文件描述符层次的输出，并包含所有程序和子进程的输出，这是Linux和MaxOS的默认行为。

这种模式可以从命令行或在Setting中设置。捕获的输出包含所有打印的字符和类似文件而非终端的行为，要阻止捕获的输出保存每次和每个写到终端的输出，可以添加一个捕获输出过滤器到(captured output filter)到试验：

```python
from sacred.utils import apply_backspaces_and_linefeeds
ex.captured_out_filter = apply_backspaces_and_linefeeds

```

这里`apply_backspaces_and_linefeeds`是一个像终端一样解释所有回车和换行符并返回修改后文本的简单函数。任何以输入字串输出（修改）字串的函数都能用为`captured_out_filter`。

##### 试验中断或失败

若运行被中断（比如Ctrl-C）或发生一个异常，Sacred会收集栈追踪和失败时间并向observer报告。结果条目的状态会设置为`INTERRUPTED`或`FAIL`。

有是试验会不是因抛出异常而失败，比如断电、内核崩溃等，，这种情况的失败不会记录到数据库中，其状态依然是`RUNNING`。这样的运行失败通过调查其心跳最容易检测出：每个运行中的试验会在固定间隔向observer报告（默认每10秒），并随着捕获的输出和信息字典更新心跳时间。因此若时间留在比间隔更前，就认为这个运行是失效的。

若发生异常，sacred默认通过除去所有sacred内置的调用来过滤栈追踪。栈追踪也存储在数据库中（如果添加了合适的observer）。若希望使用一个调试器，则需要关闭栈追踪过滤，有两种选择：

- 通过`-d`标识失效；
- Sacred同样也支持通过`-D`标识直接添加一个事后检验的`pdb`调试器。

有时需要一些自定的原因来中断试验，比如有限的计算预算等。Sacred提供了一种特殊的基异常`sacred.utils.SacredInterrupt`用来提供定制的状态编码。若产生了一个源于此的异常，则被中断试验的状态就会设为此编码。刚才提到的超时情况可有编码为`TIMEOUT`的`sacred.utils.TimeoutInterrupt`异常。可以通过创建一个继承自`sacred.utils.SacredInterrupt`的定制异常并定义一个`STATUS`成员来使用任何状态编码：

```python
from sacred.utils import SacredInterrupt
class CustomInterrupt(SacredInterrupt):
    STATUS = 'MY_CUSTOM_STATUS'

```

#### 3. 配置



#### collections

#### `namedtuple(typename, fieldnames, verbose=False, rename=False)`

返回一个名字为typename的元组子类，这个子类能用来创建有能用属性查看并可索引和迭代域的类元组对象，其实例也有帮助文档（有类型名和域名）和一个`__repr__()`方法来以`name=value`的形式列出元组内容：

- `fieldname`是单个字串，每个域名都用空格和/或逗号分开，如`'x y'`或`'x, y'`，或者`fieldname`也可以是字串序列如`['x', 'y']`，除下划线开始的任意合法python标识符都可以为域名；
- 若`rename`为真，非法的域名都会自动用位置名代替，如`['abc', 'def', 'ghi', 'abc']`会被转换为`['abc', '_1', 'ghi', '_3']`；
- 若`verbose`为真，就会在构建后打印其类定义，这个选项已过时，直接打印`_source`属性更简单。

```python
>>> Point = namedtuple('Point', ['x', 'y'])
>>> p = Point(11, y=22)     # instantiate with positional or keyword arguments
>>> p[0] + p[1]             # indexable like the plain tuple (11, 22)
33
>>> x, y = p                # unpack like a regular tuple
>>> x, y
(11, 22)
>>> p.x + p.y               # fields also accessible by name
33
>>> p                       # readable __repr__ with a name=value style
Point(x=11, y=22)

```

命名元组没有每元组字典，因此是轻量的且无需比正常元组更多的存储，它在将域名赋给由`csv`或`sqlite3`模块返回的结果元组时特别有用：

```python
EmployeeRecord = namedtuple('EmployeeRecord', 'name, age, title, department, paygrade')

import csv
for emp in map(EmployeeRecord._make, csv.reader(open("employees.csv", "rb"))):
    print(emp.name, emp.title)

import sqlite3
conn = sqlite3.connect('/companydata')
cursor = conn.cursor()
cursor.execute('SELECT name, age, title, department, paygrade FROM employees')
for emp in map(EmployeeRecord._make, cursor.fetchall()):
    print(emp.name, emp.title)

```

除继承自元组的方法外，命名元组还支持三个额外的方法和两个属性，方法和属性名以单个下划线开始：

##### somenamedtuple._make(iterable)

从一个已有序列或可迭代中创建一个实例的类方法：

```python
>>> t = [11, 22]
>>> Point._make(t)
Point(x=11, y=22)

```

##### somenamedtuple._asdict()

返回一个将域名映射到对应值的有序字典(`OrderedDict`)：

```python
>>> p = Point(x=11, y=22)
>>> p._asdict()
OrderedDict([('x', 11), ('y', 22)])

```

##### somenamedtuple._replace(**kwargs)

返回一个将特定域用新值代替的新命名字典实例：

```python
>>> p = Point(x=11, y=22)
>>> p._replace(x=33)
Point(x=33, y=22)

>>> for partnum, record in inventory.items():
...     inventory[partnum] = record._replace(price=newprices[partnum], timestamp=time.now())

```

##### somenamedtuple._source

用于创建此命名元组类的纯python代码字串，源码创建命名元组的自文档，能被打印、用`exec()`执行、或者保存到一个文件中并被导入。

##### somenamedtuple._fields

列出域名的元组，在内省或从已有命名元组中创建新明明元组时很有用：

```python
>>> p._fields            # view the field names
('x', 'y')

>>> Color = namedtuple('Color', 'red green blue')
>>> Pixel = namedtuple('Pixel', Point._fields + Color._fields)
>>> Pixel(11, 22, 128, 255, 0)
Pixel(x=11, y=22, red=128, green=255, blue=0)
```

##### 实战指导

获得一个名字以字串形式存储的域，使用`getattr()`函数：

```python
>>> getattr(p, 'x')
11
```

将字典转换为命名元组，使用双星操作符：

```pyhton]
>>> d = {'x': 11, 'y': 22}
>>> Point(**d)
Point(x=11, y=22)
```

因为命名元组也是一个正常python类，用子类添加或改变功能就很简单，下面是如何添加一个计算域和一个定宽打印形式：

```python
>>> class Point(namedtuple('Point', 'x y')):
...     __slots__ = ()
...     @property
...     def hypot(self):
...         return (self.x ** 2 + self.y ** 2) ** 0.5
...     def __str__(self):
...         return 'Point: x=%6.3f  y=%6.3f  hypot=%6.3f' % (self.x, self.y, self.hypot)


>>> for p in Point(3, 4), Point(14, 5/7):
...     print(p)
Point: x= 3.000  y= 4.000  hypot= 5.000
Point: x=14.000  y= 0.714  hypot=14.018
```

上面的子类设置`__slots__`为一个空元组，这通过字典实例的创建使得存储要求很低。

要添加新、存储的域，可以从`_fields`属性创建新命名元组类型：

```python
>>> Point3D = namedtuple('Point3D', Point._fields + ('z',))
```

可以直接添加赋值到`__doc__`域来定制文档：

```python
>>> Book = namedtuple('Book', ['id', 'title', 'authors'])
>>> Book.__doc__ += ': Hardcover book in active collection'
>>> Book.id.__doc__ = '13-digit ISBN'
>>> Book.title.__doc__ = 'Title of first printing'
>>> Book.authors.__doc__ = 'List of authors sorted by last name'
```

默认值能够通过使用`__replace()`实现来定制实例原型：

```python
>>> Account = namedtuple('Account', 'owner balance transaction_count')
>>> default_account = Account('<owner name>', 0.0, 0)
>>> johns_account = default_account._replace(owner='John')
>>> janes_account = default_account._replace(owner='Jane')
```

