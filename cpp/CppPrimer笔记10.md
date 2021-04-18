#### 1 概述



#### 2 初识泛型算法

##### 2.1 只读算法

|     算法     | 功能 | 备注 |
| :----------: | :--: | :--: |
|    `find`    |      |      |
|   `count`    |      |      |
| `accumulate` | 求和 |      |
|   `equal`    |      |      |
|  `find_if`   |      |      |

##### 2.2 写容器元素算法

|   算法    | 功能 | 备注 |
| :-------: | :--: | :--: |
|  `fill`   |      |      |
| `fill_n`  |      |      |
|  `copy`   |      |      |
| `replace` |      |      |

##### 2.3 重排容器元素算法

|     算法      | 功能 | 备注 |
| :-----------: | :--: | :--: |
|    `sort`     |      |      |
|   `unique`    |      |      |
|  `partition`  |      |      |
| `stable_sort` |      |      |



#### 3 定制操作

##### 3.1 向算法传递函数

##### 3.2 lambda表达式

一个lambda表达式表示一个可调用的代码单元，可以理解为一个未命名内联函数，它具有一个返回类型、一个参数列表和一个函数体，可以定义在函数内部。它的形式如下：
$$
[capture\ list]\ (parameter\ list)\ ->\ return\ type\ \{\ function\ body\ \}
$$
lambda必须使用尾置返回来制定返回类型。

lambda表达式可以忽略参数列表和返回类型，但必须包含捕获列表和函数体：

```cpp
auto f = [] { return 42; };
cout << f() << endl;
```

传递参数

```cpp
[] (const string& a, const string& b) { return a.size() < b.size(); };

stable_sort(words.begin(), words.end(),
            [] (const string& a, const string& b) {return a.size() < b.size();});
```

使用捕获列表

```cpp
int sz = 5;

[sz] (const string& a) { return s.size() < sz; };
```



#### 4 再探迭代器



#### 5 泛型算法结构

算法所要求的