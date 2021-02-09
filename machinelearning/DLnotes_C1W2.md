##### Logistic回归代价函数的解释

$$
\hat y = \sigma(\mathbf w^T\mathbf x+b)\qquad\text{where }\sigma(z) = \frac1{1+e^{-z}}
$$

要将$\hat y$表示为$\hat y = p(y=1\mid x)$，即
$$
\begin{aligned}
&\text{if}\quad y=1:\qquad p(y\mid x) = \hat y\\
&\text{if}\quad y=0:\qquad p(y\mid x) = 1-\hat y
\end{aligned}
$$
通过下面的式子将两者结合起来：
$$
p(y\mid x) = \hat y^y (1-\hat y)^{1-y}\quad y \in \{0,1\}
$$
而$\log$函数为严格单调递增函数，
$$
\begin{aligned}
\log p(y\mid x) &= y\log\hat y + (1-y)\log(1-\hat y)\\
&= -\mathcal L(\hat y, y)
\end{aligned}
$$
这只是单个样本的损失(loss)函数，那整个数据集上的损失函数呢？
$$
p(\text{labels in training set}) = \prod_{i=1}^m p\left(y^{(i)}\mid x^{(i)}\right)\\
$$
对这个式子进行最大似然估计，寻找到观察到这些样本概率最大的参数。因$\log$是严格单调增函数，两边同取$\log$：
$$
\begin{aligned}
\log p(\text{labels in training set}) 
&= \sum_{i=1}^m y\log\hat y +(1-y)\log(1-\hat y)\\
&= -\sum_{i=1}^m \mathcal L(\hat y, y)
\end{aligned}
$$
因此，logistic回归在训练集上的要最小化的代价(cost)函数：
$$
J(\mathbf w, b) =  \frac1m \sum_{i=1}^m \mathcal L(\hat y, y)
$$
就是在假设训练样本都是独立同分布(iid.)的条件下，用logistic模型训练集上执行极大似然估计。