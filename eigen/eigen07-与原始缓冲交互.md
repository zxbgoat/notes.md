本文解释如何使用原始C/C++数组，这在多种环境中十分有效，尤其是从其他库导入向量或矩阵时。



##### 1 简介

有时可能需要在`Eigen`内将一些预定义的数组作为向量和数组进行使用，一种方法是复制这些数据，但更好的方法是将这些内存作为Eigen类型进行重用。通过`Map`类，可以很容易地完成这个操作。



##### 2 Map类型

一个`Map`对象拥有其与Eigen对等的定义类型，注意在这种默认情况下，一个`Map`仅需一个模版参数：

```cpp
Map<Matrix<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>>
```

要构建`Map`变量，需要两个额外的信息，一个指向定义这些数据的内存区域的地址，以及需要的矩阵或向量的形状。例如定义一个float类型的形状在编译时确定的矩阵，需要执行：

```cpp
Map<MatrixXf> mf(pf, rows, cols);
```

而一个大小固定的只读整型向量可以是：

```cpp
Map<const Vector4i> mi(pi);
```

注意Map没有默认构造器，必须传递一个指针来初始化对象，但也可以绕过这个要求，详见下面。Map足够灵活，可以满足多种不同的数据表达，它还有两个可选的模版参数：

```cpp
Map<typename MatrixType, int MapOptions, typename StrideType>
```

- `MapOptions`参数指定指针是否是对齐的，默认是不对齐的；
- `StrideType`允许使用`Stride`类来自定义数组内存的布局，一个示例就是指定数据以行优先的格式组织数据。

```cpp
int array[8];
for(int i = 0; i < 8; ++i) array[i] = i;
cout << "Col-major:\n" << Map<Matrix<int, 2, 4> >(array) << endl;
cout << "Row-major:\n" << Map<Matrix<int, 2, 4, RowMajor> >(array) << endl;
cout << "Row-major using stride:\n" << Map<Matrix<int,2,4>, Unaligned, Stride<1,4> >(array) << endl;
```

运行结果为：

```bash
Column-major:
0 2 4 6
1 3 5 7
Row-major:
0 1 2 3
4 5 6 7
Row-major using stride:
0 1 2 3
4 5 6 7
```

`Stride`可以比这更加灵活，详见[`Map`](https://eigen.tuxfamily.org/dox/classEigen_1_1Map.html)和[`Stride`](https://eigen.tuxfamily.org/dox/classEigen_1_1Stride.html)类。



##### 3 使用Map变量

可以像其他的Eigen类型一样使用`Map`对象。

```cpp
typedef Matrix<float,1,Dynamic> MatrixType;
typedef Map<MatrixType> MapType;
typedef Map<const MatrixType> MapTypeConst;   // a read-only map
const int n_dims = 5;
  
MatrixType m1(n_dims), m2(n_dims);
m1.setRandom();
m2.setRandom();
float *p = &m2(0);  // get the address storing the data for m2
MapType m2map(p,m2.size());   // m2map shares data with m2
MapTypeConst m2mapconst(p,m2.size());  // a read-only accessor for m2
 
cout << "m1: " << m1 << endl;
cout << "m2: " << m2 << endl;
cout << "Squared euclidean distance: " << (m1-m2).squaredNorm() << endl;
cout << "Squared euclidean distance, using map: " <<
  (m1-m2map).squaredNorm() << endl;
m2map(3) = 7;   // this will change m2, since they share the same array
cout << "Updated m2: " << m2 << endl;
cout << "m2 coefficient 2, constant accessor: " << m2mapconst(2) << endl;
/* m2mapconst(2) = 5; */   // this yields a compile-time error
```

所有的Eigen函数都可以像其他Eigen类型一样接收Map对象，但当编写自己的函数输入Eigen类型时，这并不会自动发生：`Map`类型与它对应的`Dense`并不是等价的，详见[编写输入Eigen类型为参数的函数](https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html)。



##### 4 改变映射的数组

在声明之后可以改变`Map`对象的数组：

```cpp
int data[] = {1,2,3,4,5,6,7,8,9};
Map<RowVectorXi> v(data,4);
cout << "The mapped vector v is: " << v << "\n";
new (&v) Map<RowVectorXi>(data+4,5);
cout << "Now v is: " << v << "\n";
```

这个操作并没有调用内存分配器，因为语法指定了存储结果的地址。这个语法使得声明映射数组地址未知的`Map`对象成为可能。

```cpp
Map<Matrix3f> A(NULL);  // don't try to use this matrix yet!
VectorXf b(n_matrices);
for (int i = 0; i < n_matrices; i++)
{
  new (&A) Map<Matrix3f>(get_matrix_pointer(i));
  b(i) = A.trace();
}
```

