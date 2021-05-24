`Array`类提供一般意义上的数组，它提供了方便的方法来执行逐元素操作。



##### 1 Array类

Array是一个与Matrix类取同样模版参数的类，与Matrix类似`Eigen`也为一些常见Array提供了别名，并：

- 采用了`ArrayNt`的方式为一维数组定义别名，其中`N`和`t`分别表示大小和类型；
- 采用`ArrayNMt`的格式来为二维数组定义别名。

```cpp
Array<float, Dynamic, 1> ArrayXf;
Array<float, 3, 1> Array3f;
typedef Array<double, Dynamic, Dynamic> ArrayXXd;
typedef Array<double, 3, 3>  Array33d;
```



##### 2 获取数组内的值



##### 3 加减法

```cpp
int main()
{
  ArrayXXf a(3,3);
  ArrayXXf b(3,3);
  a << 1,2,3,
       4,5,6,
       7,8,9;
  b << 1,2,3,
       1,2,3,
       1,2,3;
       
  // Adding two arrays
  cout << "a + b = " << endl << a + b << endl << endl;
 
  // Subtracting a scalar from an array
  cout << "a - 2 = " << endl << a - 2 << endl;
}
```



##### 4 数组乘法



##### 5 其他逐元素操作



##### 6 在`array`和`matrix`之间转换

