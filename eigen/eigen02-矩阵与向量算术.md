`Eigen`通过重载算术运算符或使用特殊函数来提供矩阵和向量的算术运算。对于`Matrix`类（向量和矩阵），操作符仅被重载为支持线性代数的操作，例如`matrix1*matrix2`表示矩阵与矩阵相乘，`vector * scalar`则不被允许。



##### 1 加减法

运算符两侧的矩阵必须大小和类型都相同，因为`Eigen`不会自动执行类型提升。支持：

- 二元`+`操作如`matrixa + matrixb`；
- 二元`-`操作如`matrixa - matrixb`；
- 一元取负`-`操作如`-matrixa`；
- 复合`+=`操作如`matrixa += matrixb`；
- 复合`-=`操作如`matrixa -= matrixb`；

```cpp
int main()
{
  Matrix2d a;
  a << 1, 2, 3, 4;
  MatrixXd b(2,2);
  b << 2, 3, 1, 4;
  std::cout << "a + b =\n" << a + b << std::endl;
  std::cout << "a - b =\n" << a - b << std::endl;
  std::cout << "Doing a += b;" << std::endl;
  a += b;
  std::cout << "Now a =\n" << a << std::endl;
  Vector3d v(1,2,3);
  Vector3d w(1,0,0);
  std::cout << "-v + w - v =\n" << -v + w - v << std::endl;
}
```



##### 2 与标量乘除

与标量的乘除十分简单，支持的操作有：

- 二元乘法操作如`matrix * scalar`；
- 二元乘法操作如`scalar * matrix`；
- 二元除法操作如`matrix / scalar`；
- 复合乘法如`matrix *= scalar`；
- 复合除法如`matrix /= scalar`；

```cpp
int main()
{
  Matrix2d a;
  a << 1, 2,
       3, 4;
  Vector3d v(1,2,3);
  std::cout << "a * 2.5 =\n" << a * 2.5 << std::endl;
  std::cout << "0.1 * v =\n" << 0.1 * v << std::endl;
  std::cout << "Doing v *= 2;" << std::endl;
  v *= 2;
  std::cout << "Now v =\n" << v << std::endl;
}
```



##### 3 表达式模版的注意点

`Eigen`中算术操作符本身并不执行任何计算，它们返回**表达式对象**来描述要执行的计算，实际的计算发生在后面，即对整个表达式进行评估的时候，通常在`operator=`中。尽管这听起来很重，但任意现代的编译器都能优化掉这种抽象，得到完美优化后的代码。例如，当执行：

```cpp
VectorXf a(50), b(50), c(50), d(50);
...
a = 3*b + 4*c + 5*d;
```

`Eigen`将其编译为单个循环，这样这些数组仅遍历一次，简单而言（忽略SIMD优化）看起来是这样：

```cpp
for(int i = 0; i < 50; ++i)
  a[i] = 3*b[i] + 4*c[i] + 5*d[i];
```

因此无需担心使用大表达式，它使得`Eigen`有更多的机会进行优化。



##### 4 转置与共轭

矩阵或向量的转置、共轭以及共轭转置分别可以通过成员函数`transpose()`、`conjugate()`和`adjoint()`获得。

```cpp
MatrixXcf a = MatrixXcf::Random(2,2);
cout << "Here is the matrix a\n" << a << endl;
cout << "Here is the matrix a^T\n" << a.transpose() << endl;
cout << "Here is the conjugate of a\n" << a.conjugate() << endl;
cout << "Here is the matrix a^*\n" << a.adjoint() << endl;
```

对于实矩阵，`conjugate()`无操作，`adjoint()`等价于`transpose()`。

`transpose()`和`adjoint()`仅返回代理对象而不执行实际的转置。



##### 4 

