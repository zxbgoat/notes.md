通过C能很简单地添加新的 Python 内置模块，以下两件不能用 Python 直接做的事，可以通过 *extension modules* 来实现：实现新的内置对象类型；调用 C 的库函数和系统调用。为了支持扩展，Python API（应用程序编程接口）定义了一系列函数、宏和变量，可以访问 Python 运行时系统的大部分内容。Python 的 API 可以通过在一个 C 源文件中引用 `"Python.h"` 头文件来使用。



##### 简单例子

创建一个扩展模块 `spam` 并且想要创建对应 C 库函数 `system()`的 Python 接口。这个函数接受一个以 null 结尾的字符串参数并返回一个整数。 我们希望可以在 Python 中以如下方式调用此函数:

```python
>>> import spam
>>> status = spam.system("ls -l")
```

首先创建一个 `spammodule.c` 文件。（传统上，如果一个模块叫 `spam`，则对应实现它的 C 文件叫 `spammodule.c`；如果这个模块名字非常长，比如 `spammify`，则这个模块的文件可以直接叫 `spammify.c`。）文件中开始的两行是：

```c
#define PY_SSIZE_T_CLEAN
#include <Python.h>
```

这会导入 Python API。由于 Python 可能会定义一些能在某些系统上影响标准头文件的预处理器定义，因此在包含任何标准头文件之前，你 *必须* 先包含 `Python.h`。推荐总是在 `Python.h` 前定义 `PY_SSIZE_T_CLEAN` 。所有在 `Python.h` 中定义的用户可见的符号都具有 `Py` 或 `PY` 前缀，已在标准头文件中定义的那些除外。 考虑到便利性，也由于其在 Python 解释器中被广泛使用，`"Python.h"` 还包含了一些标准头文件: `<stdio.h>`，`<string.h>`，`<errno.h>` 和 `<stdlib.h>`。 如果后面的头文件在你的系统上不存在，它还会直接声明函数 `malloc()`，`free()` 和 `realloc()`。下面添加C函数到扩展模块，当调用 `spam.system(string)` 时会做出响应：

```c
static PyObject *
spam_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}
```

有个直接翻译参数列表的方法(举个例子，单独的 `"ls -l"` )到要传递给C函数的参数。C函数总是有两个参数，通常名字是 *self* 和 *args* 。

对模块级函数， *self* 参数指向模块对象；对于对象实例则指向方法。

*args* 参数是指向一个 Python 的 tuple 对象的指针，其中包含参数。 每个 tuple 项对应一个调用参数。 这些参数也全都是 Python 对象 --- 要在我们的 C 函数中使用它们就需要先将其转换为 C 值。 Python API 中的函数 [`PyArg_ParseTuple()`](https://docs.python.org/zh-cn/3/c-api/arg.html#c.PyArg_ParseTuple) 会检查参数类型并将其转换为 C 值。 它使用模板字符串确定需要的参数类型以及存储被转换的值的 C 变量类型。 细节将稍后说明。

[`PyArg_ParseTuple()`](https://docs.python.org/zh-cn/3/c-api/arg.html#c.PyArg_ParseTuple) 在所有参数都有正确类型且组成部分按顺序放在传递进来的地址里时，返回真(非零)。其在传入无效参数时返回假(零)。在后续例子里，还会抛出特定异常，使得调用的函数可以理解返回 `NULL` (也就是例子里所见)。