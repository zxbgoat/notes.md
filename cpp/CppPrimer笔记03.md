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

可以使用`while(cin >> s)`读取未知数量的。