在`eigen`中，所有的向量和矩阵都是`Matrix`模版类的对象，向量就是一种特殊的矩阵，它们的行或列位1。



##### 1 Matrix的前3个模版参数

`Matrix`类取6个模版参数，其中后三个有默认值，暂且不表；前三个必要的参数是：

```cpp
Matrix<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>;
```

`Eigen`为很多常用案例提供了很多别名，如`Matrix4f`为`typedef Matrix<float, 4, 4> Matrix4f`。



##### 2 向量

在`Eigen`中，向量就是特殊的矩阵，其行或列为1。其中最常见的是列向量，一般简称为向量。

比如类型`Vector3f`定义为`typedef Matrix<float, 3, 1> Vector3f;`，是一个包含3个浮点数元素的**列向量**。

而行向量类型`RowVector2i`的定义为`typedef Matrix<int, 1, 2> RowVector2i`。



##### 3 特殊的Dynamic值

`Eigen`也支持维度在编译时未知的矩阵，参数`RowsAtCompileTime`和`ColsAtCompileTime>`都可以取`Dynamic`值，表示此维度在编译时未知，因此必须按照运行时变量来处理。在`Eigen`中，这样的大小被称为动态大小；而在编译时已知的大小为静态大小。例如类型`MatrixXd`表示动态大小的双精度浮点矩阵，其定义为：

```cpp
typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
```

类似地将`VectorXi`定义为：

```cpp
typedef Matrix<double, Dynamic, 1> VectorXi;
```



##### 4 构造器

`Eigen`提供了默认的构造器，既不执行动态内存分配，也不初始化矩阵参数。比如：

- `Matrix3f a;`是一个$3\times3$矩阵，包含平坦的参数为初始化的`float[9]`数组；
- `MatrixXf b;`是一个动态大小的矩阵，当前大小为$0\times0$，尚未为其分配参数数组。

构造器也可以输入大小，它们按照给定的大小分配参数数组，但并不初始化这些参数：

- `MatrixXf a(10,15);`是一个$10\times15$动态矩阵，包括已分配但未初始化的参数；
- `VectorXf b(30);`是一个大小为30的动态向量，包括已分配但未初始化的参数。

为在动态大小和静态大小矩阵间提供一致的API，可以在固定大小矩阵上使用这些构造器，只是不起作用，因此这样的操作是合法的：

```cpp
Matrix3f a(3,3);
```

最后，`Eigen`提供了一些构造器来初始化大小最大为4的固定大小的向量参数：

```cpp
Vector2d a(5.0, 6.0);
Vector3d b(5.0, 6.0, 7.0);
Vector4d c(5.0, 6.0, 7.0, 8.0);
```



##### 5 参数存取器

`Eigen`中最主要的参数存取器是重载的圆括号操作符，注意语法`m(index)`并不局限于向量，对一般向量也适用。表示基于索引的参数数组入口，它取决于存储顺序。所有矩阵默认都是列为主存储，但可以设为行为主。

```cpp
int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << "Here is the matrix m:\n" << m << std::endl;
  VectorXd v(2);
  v(0) = 4;
  v(1) = v(0) - 1;
  std::cout << "Here is the vector v:\n" << v << std::endl;
}
```

操作符`[]`在向量中也重载为基于索引的入口，但需要注意C++中`[]`不支持一个以上的参数，因此将`[]`操作符限定于向量。



##### 6 逗号初始化

矩阵和向量参数可以使用都好初始化语法设定：

```cpp
Matrix3f m;
m << 1, 2, 3,
     4, 5, 6,
     7, 8, 9;
cout << m;
```



##### 7 调整大小

当前矩阵的大小可以通过`rows()`、`cols()`和`size()`方法获得，它们分别返回行数、列数和参数个数。可以通过`resize()`方法调整一个动态大小矩阵的大小：

```cpp
int main()
{
  MatrixXd m(2,5);
  m.resize(4,3);
  std::cout << "The matrix m is of size "
            << m.rows() << "x" << m.cols() << std::endl;
  std::cout << "It has " << m.size() << " coefficients" << std::endl;
  VectorXd v(2);
  v.resize(5);
  std::cout << "The vector v is of size " << v.size() << std::endl;
  std::cout << "As a matrix, v is of size "
            << v.rows() << "x" << v.cols() << std::endl;
}
```

如果矩阵的大小没有改变，则`resize()`方法并不执行任何操作；否则，它可能是破坏性的：参数的值可能会改变；若需要一个不改变参数的保守版本`resize()`方法，可以使用`conservativeResize()`方法。

为了API调用的一致性，所有这些方法在固定大小的矩阵上也是合法的，但试图改变一个固定大小的矩阵会触发错误。



##### 8 赋值于调整大小

赋值是一种将一个矩阵赋值到另一个的行为，`Eigen`会自动调整等号左边矩阵的大小来匹配右边矩阵的大小：

```cpp
MatrixXf a(2,2);
std::cout << "a is of size " << a.rows() << "x" << a.cols() << std::endl;
MatrixXf b(3,3);
a = b;
std::cout << "a is now of size " << a.rows() << "x" << a.cols() << std::endl;
```

当然如果左边大小是固定的，则是无法调整大小的。