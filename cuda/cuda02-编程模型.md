#### 1 核心

CUDA C++通过允许开发者定义C++函数——核心（kernel）来扩展C++，当核心被调用时，会被N个不同的CUDA线程执行N次，与普通C++函数只执行一次相对。核心：

- 通过`__global__`声明说明符和CUDA线程数来定义；
- 通过`<<<...>>>`执行配置语法来调用；
- 每个执行此核心的线程都有一个线程ID，可以在核心内通过内置变量`threadIdx`获得。

下例展示了向量加法：

```cpp
// kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
  ...
  // kernel invocation
  VecAdd<<<1, N>>>(A, B, C);
  ...
}
```



#### 2 线程层次

`threadIdx`是一个3元素向量，因此线程能用一维、二维或三维线程索引来表示，形成一维、二维或三维块的线程，称为线程块。

线程索引和线程ID之间的相互关联是十分直接的：

- 对一维的块，两者是相同的；
- 对大小为$(D_x,D_y)$的二维块，索引为$(x,y)$的线程其ID为$(x+yD_x)$；
- 对大小为$(D_x,D_y,D_z)$的三维块，索引为$(x,y,z)$的线程其ID为$x+yD_x+zD_xD_y$。

下例展示了两个矩阵GPU上的两个显卡相加：

```cpp
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
  int i = threadIdx.x;
  int j = threadIdx.y;
  C[i][j] = A[i][j] + B[i][j];
}

int main()
{
  ...
  // kernel invocation with one block of N*N*1 threads
  int numBlocks = 1;
  dim3 threadPerBlock(N, N);
  VecAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
  ...
}
```

因为一个块的所有线程都位于同一个处理器核，共享有限的核内存储资源，因此每个块的线程数都有限制，在当前GPU中，一个线程块最多包含1024个线程。

但是一个核心能被多个相同形状的线程块执行多次，因此线程总数等于每个块的线程数乘以总共的块数。块也被组织为一维、二维或三维网格的线程块，如下图所示，一个网格内的线程数量通常由所处理的数据决定。

<img src='figures/04.png' />

`<<<...>>>`语法中指定的每个块的线程数和每个网格的的块数可以使用`int`或`dim3`类型：

- 每个网格内的块可以通过一个一维、二维或三维的唯一索引标识，它在核心内通过内置变量`blockIdx`获得；
- 线程块的维度则可以在核心中通过内置变量`blockDim`变量获得。

下例扩展了矩阵相加的示例来处理多个线程块：

```cpp
// kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < N && j < N) C[i] = A[i] + B[i];
}

int main()
{
  ...
  // kernel invocation
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(N/threadsPerBlock.x, N/threadsPerBlock.y);
  VecAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
  ...
}
```

尽管可以任意取之，但通常将块的大小定为$16\times16$。

线程块需要能被独立执行：

- 必须能以任意顺序对它们进行执行，并行或串行；
- 允许在任意数量的核中以任意顺序调度，使开发者编写依据核数扩展的代码。

一个块内的线程可以通过共享内存的共享数据相互协作，并通过同步执行来协调内存的获取。具体而言就是，可以调用`__syncthreads()`内置函数指定同步点，`__syncthreads()`函数类似于一个栅栏，块内的所有线程必须等待，直到允许继续执行。除此之外，[Cooperative Groups API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)提供了丰富的线程同步原语。

