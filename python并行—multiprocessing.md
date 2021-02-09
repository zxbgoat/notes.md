#### 1 概述

[`multiprocessing`](https://docs.python.org/zh-cn/3/library/multiprocessing.html#module-multiprocessing) 是一个用于产生进程的包，具有与 [`threading`](https://docs.python.org/zh-cn/3/library/threading.html#module-threading) 模块相似API。 [`multiprocessing`](https://docs.python.org/zh-cn/3/library/multiprocessing.html#module-multiprocessing) 包同时提供本地和远程并发，使用子进程代替线程，有效避免 [Global Interpreter Lock](https://docs.python.org/zh-cn/3/glossary.html#term-global-interpreter-lock) 带来的影响。因此， [`multiprocessing`](https://docs.python.org/zh-cn/3/library/multiprocessing.html#module-multiprocessing) 模块允许程序员充分利用机器上的多核。

[`multiprocessing`](https://docs.python.org/zh-cn/3/library/multiprocessing.html#module-multiprocessing) 模块还引入了在 [`threading`](https://docs.python.org/zh-cn/3/library/threading.html#module-threading) 模块中没有的API。一个主要的例子就是 [`Pool`](https://docs.python.org/zh-cn/3/library/multiprocessing.html#multiprocessing.pool.Pool) 对象，它提供了一种快捷的方法，赋予函数并行化处理一系列输入值的能力，可以**将输入数据分配给不同进程处理**（数据并行）。下面的例子演示了在模块中定义此类函数的常见做法，以便子进程可以成功导入该模块。这个数据并行的基本例子使用了 [`Pool`](https://docs.python.org/zh-cn/3/library/multiprocessing.html#multiprocessing.pool.Pool) 。

```python
from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))
```



#### 2 `Process` 类

在 [`multiprocessing`](https://docs.python.org/zh-cn/3/library/multiprocessing.html#module-multiprocessing) 中，通过创建一个 [`Process`](https://docs.python.org/zh-cn/3/library/multiprocessing.html#multiprocessing.Process) 对象然后调用它的 `start()` 方法来生成进程。 [`Process`](https://docs.python.org/zh-cn/3/library/multiprocessing.html#multiprocessing.Process) 和 [`threading.Thread`](https://docs.python.org/zh-cn/3/library/threading.html#threading.Thread) API 相同。 一个简单的多进程程序示例是：

```python
from multiprocessing import Process

def f(name):
    print('hello', name)

if __name__ == '__main__':
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()
```

