问题描述：已知文档A的关键词向量组$\mathbf{A = [a_1, \cdots, a_m]}$及其对应的权重$\mathbf {w_a} = [w_{\mathbf a_1}, \dots, w_{\mathbf w_m}]$，文档B的关键词向量组$\mathbf{B = [b_1, \cdots, b_n]}$及其对应的权重$\mathbf{w_b}=[w_{\mathbf b_1}, \cdots, w_{\mathbf b_n}]$，以及一个向量相似度评价函数$f_{sim}(\mathbf{x,y}) \in \mathbb R$，给出一个文档A和B之间相似度评价的标准。

##### 想法1

求出$[\mathbf{A , B}]=[\mathbf{a_1, \cdots, a_m, b_1, \cdots, b_n}]$，即两个矩阵拼接起来的的极大线形无关组，不妨设其为$\mathbf C=[\mathbf{c_1, \cdots, c_l}]$，则$\mathbf A$、$\mathbf B$中的每个向量都可以映射到由$\mathbf C$向量组张成的空间$\mathcal C$中，假设$\mathbf A$中每个向量在$\mathcal C$中的坐标为$\mathbf{x_1, \cdots, x_m}$，$\mathbf B$中每个向量在$\mathcal C$中的坐标为$\mathbf{y_1, \cdots, y_n}$，这样便将所有向量都映射到了同一个空间中。

将权重分别归一化，即$w_{\mathbf x_i} = \frac{w_{\mathbf a_i}}{\sum_{k=1}^m w_{\mathbf a_k}}, i=1,\cdots,n;\ \ w_{\mathbf y_j} = \frac{w_{\mathbf b_j}}{\sum_{k=1}^n w_{\mathbf b_k}}, j=1,\cdots,m$，可将相似度定义为：
$$
\text{Sim}_{\mathbf{A,B}} = f_{sim}\left(\sum_{i=1}^m w_{\mathbf x_i} \mathbf x_i,\ \sum_{j=1}^n w_{\mathbf y_j} \mathbf y_j\right)
$$
或者：
$$
\mathbf M_S =
\begin{bmatrix}
f_{sim}(\mathbf{x_1, y_1}) & \cdots & f_{sim}(\mathbf{x_1,y_n}) \\
\vdots & & \vdots \\
f_{sim}(\mathbf{x_m,y_1}) & \cdots & f_{sim}(\mathbf {x_m,y_n})
\end{bmatrix},\
\mathbf M_C = 
\begin{bmatrix}
w_{\mathbf x_1} w_{\mathbf y_1} &\cdots & w_{\mathbf x_1} w_{\mathbf y_n} \\
\vdots & & \vdots \\
w_{\mathbf x_m} w_{\mathbf y_1} & \cdots & w_{\mathbf x_m} w_{\mathbf y_n}
\end{bmatrix}
$$
则可将相似度定义为：
$$
\text{Sim}_{\mathbf{A,B}} = \sum_{i=1}^m \sum_{j=1}^n {\mathbf M_S}_{ij} {\mathbf M_C}_{ij}
$$

##### 想法2

参考Word Movers's Distance(WMD)方法。将文档表示为关键词向量的加权点云。两篇文章的A和B之间的距离是文档A中的关键词迁移到文档B点云的最小累积距离。

假设字典长为$n$，将文档表示为正规化的词袋(nBOW)向量，$\mathbf d \in \mathcal R^n$，假设$\mathbf d$中位置$i$的单词在文档中的权重为$w_i$，则$d_i = \frac{w_i}{\sum_{j=1}^n} w_j$，因此可以将文档表示为词分布的$n-1$维单纯形。

令文档A和文档B的nBOW分别为$\mathbf{d_A}$，$\mathbf{d_B}$，设$\mathbf T \in \mathcal R^{n \times n}$表示$\mathbf{d_A,\ d_B}$的词之间的距离矩阵：
$$
\mathbf T_{ij} = f_{sim} (\mathbf{w_{d_A}}_i,\ \mathbf{w_{d_B}}_j)
$$
其中$\mathbf{w_{d_A}}_i$表示$\mathbf{d_A}$中第$i$个位置对应词的词向量；$\mathbf{w_{d_B}}_i$类似。于是，文档A和B之间的距离就可以表示为：
$$
\begin{aligned}
\text{Sim}_{\mathbf{A,B}}=\min_{\mathbf T \ge 0} &\sum_{i,j=1}^n \mathbf T_{ij}c(i,j) \\
\text{subject to: } &\sum_{j=1}^n \mathbf T_{ij} = {d_{\mathbf A}}_i\ \ \ \ \forall i  \in \{1,\cdots,n\} \\
&\sum_{i=1}^n \mathbf T_{ij} =  {d_{\mathbf B}}_j,\ \ \ \forall j \in \{1,\cdots,n\}
\end{aligned}
$$
参考文献：[Form Word Embedding To Document Distances](http://proceedings.mlr.press/v37/kusnerb15.pdf)