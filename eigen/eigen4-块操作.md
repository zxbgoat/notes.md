块就是矩阵或数组的一个矩形部分，块表达式既可以作为左值、也可以作为右值使用。与通常的`Eigen`表达式类似，这种抽象在编译器优化下的运行时消耗为0。



##### 1 使用块操作

`Eigen`中最常用的块操作是`.block()`方法，它有两个版本，语法分别如下：

| 块操作                           | 构建一个动态大小块表达式的版本 | 构建一个静态大小块表达式的版本 |
| -------------------------------- | ------------------------------ | ------------------------------ |
| 从`(i,j)`开始，大小为`(p,q)`的块 | `matrix(i, j, p, q)`           | `matrix<p, q>(i, j)`           |

这两种版本都可以用在静态大小或动态大小的矩阵或数组上，这两者在语义上是等价的，唯一的区别在于当块比较小时，静态大小的版本通常会更快。

```cpp
int main()
{
  Eigen::MatrixXf m(4,4);
  m <<  1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
       13,14,15,16;
  cout << "Block in the middle" << endl;
  cout << m.block<2,2>(1,1) << endl << endl;
  for (int i = 1; i <= 3; ++i)
  {
    cout << "Block of size " << i << "x" << i << endl;
    cout << m.block(0,0,i,i) << endl << endl;
  }
}
```

上面的代码中，`.block()`作为右值使用，也就是仅读取了它们的值；但是块也能作为左值被赋值：

```cpp
int main()
{
  Array22f m;
  m << 1,2,
       3,4;
  Array44f a = Array44f::Constant(0.6);
  cout << "Here is the array a:" << endl << a << endl << endl;
  a.block<2,2>(1,1) = m;
  cout << "Here is now a with m copied into its central 2x2 block:" << endl << a << endl << endl;
  a.block(0,0,2,3) = a.block(2,1,2,3);
  cout << "Here is now a with bottom-right 2x3 block copied into top-left 2x3 block:" << endl << a << endl << endl;
}
```

虽然`block()`方法能用于任何块操作，但也有其他一些特例提供了更特定的API或/和更好的性能。关于性能的话题，重要的就是在编译时给`Eigen`足够尽可能多的信息。比如，若块是矩阵中的一列，可以使用特定的`.col()`函数，从而给它优化的机会。



##### 2 行与列

单独的行或列是块中的特殊情况，`Eigen`提供了`.col()`和`.row()`方法来更方便地处理它们：

- $i^{th}$行：`matrix.row(i)`；
- $i^{th}$列：`matrix.col(j)`。

```cpp
int main()
{
  Eigen::MatrixXf m(3,3);
  m << 1,2,3,
       4,5,6,
       7,8,9;
  cout << "Here is the matrix m:" << endl << m << endl;
  cout << "2nd Row: " << m.row(1) << endl;
  m.col(2) += 3 * m.col(0);
  cout << "After adding 3 times the first column into the third column, the matrix m is:\n";
  cout << m << endl;
}
```



##### 3 关于顶角的操作

`Eigen`也为包含矩阵或数组一个顶角或边的块提供了特定的方法，如`.topLeftCorner()`方法可用于指向矩阵左上角的块，不同的可能性如下表总结所示：

| 块操作                | 构建一个动态大小块表达式的版本   | 构建一个静态大小块表达式的版本     |
| --------------------- | -------------------------------- | ---------------------------------- |
| 左上角的$p\times q$块 | `matrix.topLeftCorner(p, q)`     | `matrix.topLeftCorner<p, q>()`     |
| 左下角的$p\times q$块 | `matrix.bottomLeftCorner(p, q)`  | `matrix.bottomLeftCorner<p, q>()`  |
| 右上角的$p\times q$块 | `matrix.topRightCorner(p, q)`    | `matrix.topRightCorner<p, q>()`    |
| 右下角的$p\times q$块 | `matrix.bottomRightCorner(p, q)` | `matrix.bottomRightCorner<p, q>()` |
| 前$p$行               | `matrix.topRows(p)`              | `matrix.topRows<p>()`              |
| 后$q$行               | `matrix.bottomRows(q)`           | `matrix.bottomRows<q>()`           |
| 前$p$列               | `matrix.leftCols(p)`             | `matrix.leftCols<p>()`             |
| 后$q$列               | `matrix.rightCols(q)`            | `matrix.rightCols<q>()`            |

```cpp
int main()
{
  Eigen::Matrix4f m;
  m << 1, 2, 3, 4,
       5, 6, 7, 8,
       9, 10,11,12,
       13,14,15,16;
  cout << "m.leftCols(2) =" << endl << m.leftCols(2) << endl << endl;
  cout << "m.bottomRows<2>() =" << endl << m.bottomRows<2>() << endl << endl;
  m.topLeftCorner(1,3) = m.bottomRightCorner(3,1).transpose();
  cout << "After assignment, m = " << endl << m << endl;
}
```



##### 4 向量的块操作

`Eigen`也提供了一些为向量或一维数组设计的块操作：

| 块操作                   | 构建一个动态大小块表达式的版本 | 构建一个静态大小块表达式的版本 |
| ------------------------ | ------------------------------ | ------------------------------ |
| 前$n$个元素              | `vector.head(n)`               | `vector.head<n>()`             |
| 后$n$个元素              | `vector.tail(n)`               | `vector.tail<n>()`             |
| 以$i$开始，包含$n$个元素 | `vector.segment(i, n)`         | `vector.segment<n>(i)`         |

```cpp
int main()
{
  Eigen::ArrayXf v(6);
  v << 1, 2, 3, 4, 5, 6;
  cout << "v.head(3) =" << endl << v.head(3) << endl << endl;
  cout << "v.tail<3>() = " << endl << v.tail<3>() << endl << endl;
  v.segment(1,4) *= 2;
  cout << "after 'v.segment(1,4) *= 2', v =" << endl << v << endl;
}
```

