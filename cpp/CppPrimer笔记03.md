#### 2 标准库类型string

##### 2.1 定义和初始化

string可以使用默认初始化、拷贝初始化、直接初始化等多种方法初始化：

```cpp
string s0;                   // 默认初始化，s0是空串
string s1 = s0;              // 拷贝初始化，s1是s0的副本
string s2(s0);               // 直接初始化，s3的值是s0的值；
string s3 = "value";         // 拷贝初始化，s3是字面值"value"的副本，除最后的空字符；
string s4("value");          // 直接初始化，效果等价于上面
string s5(10, 'c');          // 直接初始化，s5的内容是"cccccccccc"
string s6 = sring(10, 'c');  // 拷贝初始化，等价于string temp(10, 'c'); string s6 = temp
```

##### 2.2 string对象上的操作

下面列出了string的一些操作：

```cpp
os << s;         // 将s写到输出流os中，返回os
is >> s;         // 从is中读取字符串赋给s，忽略开头的空白（空格符、换行符、制表符），直到遇见下一处空白，返回is
getline(is, s);  // 从is种读取一行赋给s，返回is
s1 + s2;         // 
s1 += s2;
s1 == s2;
s1 != s2;
s1 <= s2;
s1 >= s2;
```

可以使用`while(cin >> s)`读取未知数量的输入。

使用`getline`能够读取一整：

- 参数为一个输入流和一个`string`对象；
- 函数从给定输入流读取内容，直到遇到换行符为止（换行符也被读入）；
- 然后将所读内容保存到`string`对象中（不存换行符）；
- 若输入的一开始就是换行符，则得到的是空`string`；
- 返回它的流参数。

```cpp
string line;
while(getline(cin, line)) cout << line << endl;
```

`string`类和其他大多数标准库类型都定义了几种配套的类型，体现了标准库机器无关的特性：

- `string`对象的`size()`方法返回的`string::size_type`类型的值，就是其中一种；
- 它是一种无符号类型的值，而且能存下任何`string`对象的大小；
- 所有用于存放`string`类`size`函数返回的值的变量，都应该定义为这种类型（可以使用`auto`或`decltype`推断）；
- `size`方法返回的是无符号整数，不能在表达式中混用带符号和无符号的数，比如`n`是一个负值，则`s.size()<n`肯定为`true`。