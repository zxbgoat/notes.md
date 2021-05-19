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

`transpose()`和`adjoint()`仅返回代理对象而不执行实际的转置。若执行`b = a.transpose()`，则转置在结果写到`b`的时候才发生；不过当执行`a = a.transpose()`时，`Eigen`在转置结束前就将结果写入`a`中，因此这样不会得到正确的结果：

```cpp
Matrix2i a; a << 1, 2, 3, 4;
cout << "Here is the matrix a:\n" << a << endl;
a = a.transpose(); // !!! do NOT do this !!!
cout << "and the result of the aliasing effect:\n" << a << endl;
```

得到的结果是：

```bash
Here is the matrix a:
1 2
3 4
and the result of the aliasing effect:
1 2
2 4
```

这就是所谓的别名（aliasinig）问题，在`saaertions`未禁用的调试模式下，这种缺陷会被自动检测。要执行就地转置，可以使用`transposeInPlace`方法，如：

```cpp
MatrixXf a(2,3); a << 1, 2, 3, 4, 5, 6;
cout << "Here is the initial matrix a:\n" << a << endl;
a.transposeInPlace();
cout << "and after being transposed:\n" << a << endl;
```

同样也有相应的`adjointInPlace`方法。



##### 5 矩阵相乘

矩阵相乘依然使用`*`操作符，而向量是特殊的矩阵，因此矩阵与向量的乘法、向量与向量的外积都是矩阵乘法的特殊情况，都通过下面两种运算符处理：

- 二元操作符`*`如`a * b`；
- 复合运算符`*=`如`a *= b`。

```cpp
int main()
{
  Matrix2d mat;
  mat << 1, 2,
         3, 4;
  Vector2d u(-1,1), v(2,0);
  std::cout << "Here is mat*mat:\n" << mat*mat << std::endl;
  std::cout << "Here is mat*u:\n" << mat*u << std::endl;
  std::cout << "Here is u^T*mat:\n" << u.transpose()*mat << std::endl;
  std::cout << "Here is u^T*v:\n" << u.transpose()*v << std::endl;
  std::cout << "Here is u*v^T:\n" << u*v.transpose() << std::endl;
  std::cout << "Let's multiply mat by itself" << std::endl;
  mat = mat*mat;
  std::cout << "Now mat is mat:\n" << mat << std::endl;
}
```

注意，`Eigen`将矩阵乘法作为特例处理，因此`m = m * m`中会引入临时变量`temp = m * m; m = temp;`，因此不会出现上面的别名问题。而如果确信矩阵乘积能被安全地计算到目的矩阵，不会产生别名问题，则可以使用`noalias()`方法来避免临时变量：`c.noalias() = a * b;`。



##### 6 点积与叉积

需要使用`dot()`和`cross()`方法来获取点积和叉积，当然点积也可以通过`u.adjoint() * v`作为$1\times1$矩阵来获得：

```cpp
int main()
{
  Vector3d v(1,2,3);
  Vector3d w(0,1,2);
  cout << "Dot product: " << v.dot(w) << endl;
  double dp = v.adjoint()*w; // automatic conversion of the inner product to a scalar
  cout << "Dot product via a matrix product: " << dp << endl;
  cout << "Cross product:\n" << v.cross(w) << endl;
}
```

记住：叉积仅适用于大小为3的向量，而点积则适用于任意大小的向量，当使用复数时，`Eigen`的点积在第一个变量上共轭线性、第二个变量上线性。



##### 7 基本算术归约操作

`Eigen`也支持一些归约操作来将矩阵或向量归约为单个值对这些矩阵参数求和（通过`sum()`方法）、求积（通过`prod()`方法）、取最大值（通过`maxCoeff()`方法）、取最小值（通过`minCoeff()`方法）、求迹（通过`trace()`方法，矩阵对角线元素的和）。

```cpp
int main()
{
  Eigen::Matrix2d mat;
  mat << 1, 2,
         3, 4;
  cout << "Here is mat.sum():       " << mat.sum()       << endl;
  cout << "Here is mat.prod():      " << mat.prod()      << endl;
  cout << "Here is mat.mean():      " << mat.mean()      << endl;
  cout << "Here is mat.minCoeff():  " << mat.minCoeff()  << endl;
  cout << "Here is mat.maxCoeff():  " << mat.maxCoeff()  << endl;
  cout << "Here is mat.trace():     " << mat.trace()     << endl;  // mat.trace() equals to mat.diagonal().sum()
}
```

`minCoeff()`和`maxCoeff()`方法同样也有一些返回矩阵最值元素坐标的变体：

```cpp
Matrix3f m = Matrix3f::Random();
std::ptrdiff_t i, j;
float minOfM = m.minCoeff(&i,&j);
cout << "Here is the matrix m:\n" << m << endl;
cout << "Its minimum coefficient (" << minOfM << ") is at position (" << i << "," << j << ")\n\n"; 
RowVector4i v = RowVector4i::Random();
int maxOfV = v.maxCoeff(&i);
cout << "Here is the vector v: " << v << endl;
cout << "Its maximum coefficient (" << maxOfV  << ") is at position " << i << endl;
```



##### 8 操作的合法性

`Eigen`会检查操作的合法性：

- 如果可能的话会在编译时进行检查，产生编译错误；
- 但在检查动态大小时，则使用运行时断言。
