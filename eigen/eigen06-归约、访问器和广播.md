#### 1 归约

在`Eigen`中，归约是一种输入矩阵或数组，返回单个标量值的函数：

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
  cout << "Here is mat.trace():     " << mat.trace()     << endl;
}
```

##### 1.1 范数计算

向量的平方范数（即欧式$l^2$）通过`squaredNorm()`方法获得，它等价于向量与自身的点积；`Eigen`也提供了`norm()`方法，返回`squaredNorm()`的平方根。这些操作也可以在矩阵上执行，这种情况下一个$n \times p$矩阵可视为大小为$n\times p$的向量，因此`norm()`方法返回"Frobenius"或"Hilbert-Schmidt"范数。这里避免说一个矩阵的$l^2$范数，因为它表示不同的意思。

若想要获得其他逐元素的$l^p$范数，使用`lpNorm<p>()`方法，若希望获得$l^{\infty}$范数模版参数`p`可以取特殊`infinity`，也即参数重的最大绝对值。

```cpp
int main()
{
  VectorXf v(2);
  MatrixXf m(2,2), n(2,2);
  
  v << -1,
       2;
  
  m << 1,-2,
       -3,4;
 
  cout << "v.squaredNorm() = " << v.squaredNorm() << endl;
  cout << "v.norm() = " << v.norm() << endl;
  cout << "v.lpNorm<1>() = " << v.lpNorm<1>() << endl;
  cout << "v.lpNorm<Infinity>() = " << v.lpNorm<Infinity>() << endl;
 
  cout << endl;
  cout << "m.squaredNorm() = " << m.squaredNorm() << endl;
  cout << "m.norm() = " << m.norm() << endl;
  cout << "m.lpNorm<1>() = " << m.lpNorm<1>() << endl;
  cout << "m.lpNorm<Infinity>() = " << m.lpNorm<Infinity>() << endl;
}
```

norm运算符：1-norm和$\infty$-norm矩阵操作符norm可以这样方便地计算:

```cpp
int main()
{
  MatrixXf m(2,2);
  m << 1,-2,
       -3,4;
 
  cout << "1-norm(m)     = " << m.cwiseAbs().colwise().sum().maxCoeff()
       << " == "             << m.colwise().lpNorm<1>().maxCoeff() << endl;
  cout << "infty-norm(m) = " << m.cwiseAbs().rowwise().sum().maxCoeff()
       << " == "             << m.rowwise().lpNorm<1>().maxCoeff() << endl;
}
```

##### 1.2 布尔归约

下面的归约在布尔值上操作:

- `all()`返回`true`，如果矩阵或数组中所有值的计算结果都是`true`；
- `any()`返回`true`，如果矩阵或数组中存在值的计算结果都是`true`；
- `count()`返回矩阵或数组中计算结果为`true`的参数个数。

这些通常与`Array`提供的逐元素比较和相等性操作符一起使用，例如`array > 0`的结果是一个与`array`大小相同的`Array`对象，因此`(array > 0).all()`测试手是否所有值都大于0：

```cpp
int main()
{
  ArrayXXf a(2,2);
  
  a << 1,2,
       3,4;
 
  cout << "(a > 0).all()   = " << (a > 0).all() << endl;
  cout << "(a > 0).any()   = " << (a > 0).any() << endl;
  cout << "(a > 0).count() = " << (a > 0).count() << endl;
  cout << endl;
  cout << "(a > 2).all()   = " << (a > 2).all() << endl;
  cout << "(a > 2).any()   = " << (a > 2).any() << endl;
  cout << "(a > 2).count() = " << (a > 2).count() << endl;
}
```



#### 2 访问器

访问器在需要获得矩阵或数组内某一参数的位置时十分有用，最简单的示例就是`maxCoeff(&i, &j)`和`minCoeff(&i, &j)`。传递的参数是存储行位置和列位置的指针，这些值的类型应为`Index`：

```cpp
int main()
{
  Eigen::MatrixXf m(2,2);
  m << 1, 2,
       3, 4;
  //get location of maximum
  MatrixXf::Index maxRow, maxCol;
  float max = m.maxCoeff(&maxRow, &maxCol);
  //get location of minimum
  MatrixXf::Index minRow, minCol;
  float min = m.minCoeff(&minRow, &minCol);
  cout << "Max: " << max <<  ", at: " <<
     maxRow << "," << maxCol << endl;
  cout << "Min: " << min << ", at: " <<
     minRow << "," << minCol << endl;
}
```

这两个函数也返回最小值和最大值的值。



#### 3 部分归约

部分归约是指可以对矩阵或数组进行逐行或逐列归约的操作，返回对应的列或行向量。部分归约通过`colwise()`和`rowwise()`应用。

```cpp
int main()
{
  Eigen::MatrixXf mat(2,4);
  mat << 1, 2, 6, 9,
         3, 1, 7, 2;
  std::cout << "Column's maximum: " << std::endl
   << mat.colwise().maxCoeff() << std::endl;
  std::cout << "Row's maximum: " << std::endl
   << mat.rowwise().maxCoeff() << std::endl;
}
```

注意：逐列操作返回行向量，逐行操作返回列向量。

