`Eigen`并没有提供直观的方法来取切片或重塑矩阵，但这些特性很容易就能使用`Map`类计算。



##### 1 重塑

重塑（reshape）操作在于改变矩阵形状的同时，保持相同的参数。这里并未直接改变输入矩阵本身，这对于编译时大小不可行，而是使用`Map`在存储上创建一个不同的视图：

```cpp
MatrixXf M1(3,3);    // Column-major storage
M1 << 1, 2, 3,
      4, 5, 6,
      7, 8, 9;
Map<RowVectorXf> v1(M1.data(), M1.size());
cout << "v1:" << endl << v1 << endl;
Matrix<float,Dynamic,Dynamic,RowMajor> M2(M1);
Map<RowVectorXf> v2(M2.data(), M2.size());
cout << "v2:" << endl << v2 << endl;
```

注意输入矩阵的存储顺序如何改变现行视图下参数的顺序，下面是另一示例：

```cpp
MatrixXf M1(2,6);    // Column-major storage
M1 << 1, 2, 3,  4,  5,  6,
      7, 8, 9, 10, 11, 12;
Map<MatrixXf> M2(M1.data(), 6,2);
cout << "M2:" << endl << M2 << endl;
```



##### 2 切片

切片在于取一些分布于矩阵内的行、列或元素：

```cpp
RowVectorXf v = RowVectorXf::LinSpaced(20,0,19);
cout << "Input:" << endl << v << endl;
Map<RowVectorXf,0,InnerStride<2> > v2(v.data(), v.size()/2);
cout << "Even:" << v2 << endl;
```

也可以根据实际存储顺序适当使用`outer-stride`或`inner-stride`来每三列取一列：

```cpp
MatrixXf M1 = MatrixXf::Random(3,8);
cout << "Column major input:" << endl << M1 << "\n";
Map<MatrixXf,0,OuterStride<> > M2(M1.data(), M1.rows(), (M1.cols()+2)/3, OuterStride<>(M1.outerStride()*3));
cout << "1 column over 3:" << endl << M2 << "\n";
 
typedef Matrix<float,Dynamic,Dynamic,RowMajor> RowMajorMatrixXf;
RowMajorMatrixXf M3(M1);
cout << "Row major input:" << endl << M3 << "\n";
Map<RowMajorMatrixXf,0,Stride<Dynamic,3> > M4(M3.data(), M3.rows(), (M3.cols()+2)/3,
                                              Stride<Dynamic,3>(M3.outerStride(),3));
cout << "1 column over 3:" << endl << M4 << "\n";
```

