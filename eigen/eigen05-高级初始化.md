##### 1 逗号初始化

`Eigen`提供了逗号初始化器，可以方便地设置矩阵或数组的所有参数。只需列出这些参数，从左上角开始，按照自左向右自上而下的顺序。对象大小需要预先确定好，参数太多或太少都会使`Eigen`抱怨。

```cpp
Matrix3f m;
m << 1, 2, 3,
     4, 5, 6,
     7, 8, 9;
std::cout << m;
```

此外，初始列表的元素本身可以是矩阵或向量，一个常用的作用就是将向量或矩阵连接起来。下面展示了如何连接两个行向量，记住在使用逗号初始化器前需要设置其大小：

```cpp
RowVectorXd vec1(3);
vec1 << 1, 2, 3;
std::cout << "vec1 = " << vec1 << std::endl;
RowVectorXd vec2(4);
vec2 << 1, 4, 9, 16;
std::cout << "vec2 = " << vec2 << std::endl;
RowVectorXd joined(7);
joined << vec1, vec2;
std::cout << "joined = " << joined << std::endl;
```

可以使用同样的方法来用块结构初始化矩阵：

```cpp
MatrixXf matA(2, 2);
matA << 1, 2, 3, 4;
MatrixXf matB(4, 4);
matB << matA, matA/10, matA/10, matA;
std::cout << matB << std::endl;
```

逗号初始化器也能用来填充`m.row(i)`这样的块表达式：

```cpp
Matrix3f m;
m.row(0) << 1, 2, 3;
m.block(1,0,2,2) << 4, 5, 7, 8;
m.col(2).tail(2) << 6, 9;                   
std::cout << m;
```



##### 2 特殊矩阵与数组

`Matrix`和`Array`类有`Zero()`这样的静态方法，可以用来将所有元素初始化为0。它有三个变体：

- 第一个没有参数，只能用于静态大小的对象；
- 第二个取一个参数，初始化一维动态大小的对象；
- 第三个取两个参数，初始化二维动态大小的对象。

```cpp
std::cout << "A fixed-size array:\n";
Array33f a1 = Array33f::Zero();
std::cout << a1 << "\n\n";
std::cout << "A one-dimensional dynamic-size array:\n";
ArrayXf a2 = ArrayXf::Zero(3);
std::cout << a2 << "\n\n";
std::cout << "A two-dimensional dynamic-size array:\n";
ArrayXXf a3 = ArrayXXf::Zero(3, 4);
std::cout << a3 << "\n";
```

类似地：

- `Constant(value)`将所有参数设置为`value`，若需要指定大小，额外的参数放在`value`前面，如`MatrixXd::Constant(rows, cols, value)`；
- `Random()`方法将矩阵或数组元素填充为随机值；
- `Identity()`单位矩阵仅针对`Matrix`类；
- `LinSpace(size, low, high)`方法仅针对向量或一维数组，它产生指定大小的向量，值均匀分布在`low`和`high`之间；

```cpp
ArrayXXf table(10, 4);
table.col(0) = ArrayXf::LinSpaced(10, 0, 90);
table.col(1) = M_PI / 180 * table.col(0);
table.col(2) = table.col(1).sin();
table.col(3) = table.col(1).cos();
std::cout << "  Degrees   Radians      Sine    Cosine\n";
std::cout << table << std::endl;
```

上例显式，像`LinSpace()`这样的返回对象可以赋值给变量。`Eigen`定义了`setZero()`、`MatrixBase::setIdentity()`、`DenseBase::setLinSpace()`来方便地实现这些。

```cpp
const int size = 6;
MatrixXd mat1(size, size);
mat1.topLeftCorner(size/2, size/2)     = MatrixXd::Zero(size/2, size/2);
mat1.topRightCorner(size/2, size/2)    = MatrixXd::Identity(size/2, size/2);
mat1.bottomLeftCorner(size/2, size/2)  = MatrixXd::Identity(size/2, size/2);
mat1.bottomRightCorner(size/2, size/2) = MatrixXd::Zero(size/2, size/2);
std::cout << mat1 << std::endl << std::endl;
 
MatrixXd mat2(size, size);
mat2.topLeftCorner(size/2, size/2).setZero();
mat2.topRightCorner(size/2, size/2).setIdentity();
mat2.bottomLeftCorner(size/2, size/2).setIdentity();
mat2.bottomRightCorner(size/2, size/2).setZero();
std::cout << mat2 << std::endl << std::endl;
 
MatrixXd mat3(size, size);
mat3 << MatrixXd::Zero(size/2, size/2), MatrixXd::Identity(size/2, size/2),
        MatrixXd::Identity(size/2, size/2), MatrixXd::Zero(size/2, size/2);
std::cout << mat3 << std::endl;
```

所有预定义矩阵、向量和数组对象可以在[这里](https://eigen.tuxfamily.org/dox/group__QuickRefPage.html)查阅。



##### 3 作为临时对象

如上所述，`Zero()`和`Constant()`这样的静态方法能够在声明或赋值右边时初始化变量，而实际上它们返回的是所谓的**表达式对象**，只有在需要时才被求值为矩阵或数组，因此这种语法不会招致任何开销。这种表达式可用作临时对象：

```cpp
int main()
{
  MatrixXd m = MatrixXd::Random(3,3);
  m = (m + MatrixXd::Constant(3,3,1.2)) * 50;
  cout << "m =" << endl << m << endl;
  VectorXd v(3);
  v << 1, 2, 3;
  cout << "m * v =" << endl << m * v << endl;
}
```

表达式`m + MatrixXd::Constant(3,3,1.2)`构造了一个$3\times3$的矩阵表达式；逗号初始化器也能用于构建临时对象：

```cpp
MatrixXf mat = MatrixXf::Random(2, 3);
std::cout << mat << std::endl << std::endl;
mat = (MatrixXf(2,2) << 0, 1, 1, 0).finished() * mat;
std::cout << mat << std::endl;
```

这里的`.finished()`函数需要在临时子矩阵逗号初始化完成后获得实际的矩阵对象。