##### 1 基本线性解法

**问题**：有一个方程系统，可以写为单个矩阵方程：$Ax = b$，其中$A$和$b$都是矩阵（$b$也可以是向量），需要找到方程的解$x$。

**方法**：可以从多种分解中选择，取决于矩阵$A$看起来像什么，以及更注重速度还是精度。下面以一个在所有情况下都有效的方法开始，是一个很好的折衷：

```cpp
#include <Eigen/Dense>

int main()
{
   Matrix3f A;
   Vector3f b;
   A << 1,2,3,  4,5,6,  7,8,10;
   b << 3, 3, 4;
   cout << "Here is the matrix A:\n" << A << endl;
   cout << "Here is the vector b:\n" << b << endl;
   Vector3f x = A.colPivHouseholderQr().solve(b);
   cout << "The solution is:\n" << x << endl;
}
```

在这个示例中，`colPivHouseholderQr()`方法返回`ColPivHouseholderQr`类的一个对象，因为这个矩阵类型是`Matrix3f`，这一行可以替换为：

```cpp
ColPivHouseholderQR<Matrix3f> dec(A);
Vector3f x = dec.solve(b);
```

这里，`ColPivHouseholderQR`是一个以列为轴的QR分解，它在这里是一个很好的折衷，因为它对所有矩阵都有效，并且速度也很快。下面是可以选择的分解方法，取决于矩阵的形式和所做的权衡：

|                             分解                             |               方法                |     要求     | 速度（中小矩阵） | 速度（大型矩阵） | 精度 |
| :----------------------------------------------------------: | :-------------------------------: | :----------: | :--------------: | :--------------: | :--: |
| [PartialPivLU](https://eigen.tuxfamily.org/dox/classEigen_1_1PartialPivLU.html) |          partialPivLu()           |     可逆     |        ++        |        ++        |  +   |
| [FullPivLU](https://eigen.tuxfamily.org/dox/classEigen_1_1FullPivLU.html) |            fullPivLu()            |      无      |        -         |        --        | +++  |
| [HouseholderQR](https://eigen.tuxfamily.org/dox/classEigen_1_1HouseholderQR.html) |          householderQr()          |      无      |        ++        |        ++        |  +   |
| [ColPivHouseholderQR](https://eigen.tuxfamily.org/dox/classEigen_1_1ColPivHouseholderQR.html) |       colPivHouseholderQr()       |      无      |        +         |        -         | +++  |
| [FullPivHouseholderQR](https://eigen.tuxfamily.org/dox/classEigen_1_1FullPivHouseholderQR.html) |      fullPivHouseholderQr()       |      无      |        -         |        --        | +++  |
| [CompleteOrthogonalDecomposition](https://eigen.tuxfamily.org/dox/classEigen_1_1CompleteOrthogonalDecomposition.html) | completeOrthogonalDecomposition() |      无      |        +         |        -         | +++  |
| [LLT](https://eigen.tuxfamily.org/dox/classEigen_1_1LLT.html) |               llt()               |     正定     |       +++        |       +++        |  +   |
| [LDLT](https://eigen.tuxfamily.org/dox/classEigen_1_1LDLT.html) |              ldlt()               | 正定或半负定 |       +++        |        +         |  ++  |
| [BDCSVD](https://eigen.tuxfamily.org/dox/classEigen_1_1BDCSVD.html) |             bdcSvd()              |      无      |        -         |        -         | +++  |
| [JacobiSVD](https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html) |            jacobiSvd()            |      无      |        -         |       ---        | +++  |

所有这些分解都提供了一个像上例一样使用的`solve()`方法。比如，若矩阵是正定的，根据上表LLT或LDLT是一个很好的选择：

```cpp
int main()
{
   Matrix2f A, b;
   A << 2, -1, -1, 3;
   b << 1, 2, 3, 1;
   cout << "Here is the matrix A:\n" << A << endl;
   cout << "Here is the right hand side b:\n" << b << endl;
   Matrix2f x = A.ldlt().solve(b);
   cout << "The solution is:\n" << x << endl;
}
```



##### 2 检查是否存在解

仅当解在一定误差范围时，才考虑其为有效。因此`Eigen`让用户自己检查：

```cpp
int main()
{
   MatrixXd A = MatrixXd::Random(100,100);
   MatrixXd b = MatrixXd::Random(100,50);
   MatrixXd x = A.fullPivLu().solve(b);
   double relative_error = (A*x - b).norm() / b.norm(); // norm() is L2 norm
   cout << "The relative error is:\n" << relative_error << endl;
}
```



##### 3 计算特征值与特征向量

