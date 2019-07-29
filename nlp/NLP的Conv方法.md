#### 基于GatedCNN的语言模型

本文展现一个语言模型的卷积方法，它引入一个新的门控机制(Gating Mechanism)来简化梯度的反向传播，尽管更简单但其表现超过了LSTM风格的门控。在WikiText-103上取得最高水平，也在Google Billion Word基准上获得了最优单GPU结果。在隐含因素重要的设置里，这个模型取得了数量级的加速。

##### 1.简介

传统语言模型受困于数据的稀疏性，难以表达大量文本，因此也无法表达长依赖。神经语言模型通过将单词嵌入连续空间中解决了这个问题。本文介绍门控卷积神经网络(Gated Convolutional Networks)并应用于语言模型。卷积神经网络能被堆叠起来表示大范围上下文并用更抽象的特征在越来越大的上下文抽取分层(hierarchical)特征。这就使得可以用上下文范围为$N$、核(kernel)宽为$k$的$\mathcal O\left( \frac{N}{k} \right)$的操作来表示长期依赖关系。将输入进行分层分析与经典的语法形式相似。分层结构也简化了学习，因对给定的上下文范围而言，非线性的数量相对于链式结构是减少的，因此缓和了梯度弥散问题。

现代硬件非常适于高度可并行的模型，而卷积网络非常服从这种计算范式。另外模型的门控线性单元通过为深度结构在获得非线性能力时提供线性梯度路径减缓了梯度弥散问题。为评估模型处理长范围依赖的能力，在模型以整段为条件的WikiText-103基准上进行了检验，最终显示门控线性单元比LSTM风格的门控正确率更高、收敛更快。

##### 2.方法

本文介绍一种新的将循环连接替换为门控时间卷积的语言模型。神经语言模型为$w_0,\dots,w_n$的每个词产生$\mathbf H=\left[ \mathbf h_1,\dots,\mathbf h_N \right]$的上下文表示以预测下个单词$P(w_i|\mathbf h_i)$。我们的方法对输入进行卷积来得到$\mathbf H=f*w$，因此没有时间依赖，这就使得其易于将一句话的每个词并行化。这个过程会将每个上下文计算为一些前面单词的函数。相比于循环神经网络，上下文的范围有限但在实践中可以表达足够的上下文来表现很好。

<img src="NLP的Conv方法/GatedCNN.png" width="500px" align="middle" />

Figure1展示了模型架构。词语用存储在查询表$\mathbf D^{|\mathcal V|\times m}$中的向量嵌入来表示，其中$|\mathcal V|$是词典中的单词数，$m$是嵌入维数，模型输入是一列表示为嵌入$\mathbf E=[\mathbf D_{w_0},\dots,\mathbf D_{w_N}]$的序列单词$w_0,\dots,w_N$。通过
$$
h_l(\mathbf X) = (\mathbf X * \mathbf W+\mathbf b) \otimes \sigma(\mathbf X * \mathbf V + \mathbf c)
$$
来计算隐层$h_0,\dots,h_L$。其中$\mathbf X \in \mathbb R^{N\times m}$是$h_l$层的输入，可能是此嵌入或前面层的输出，$\mathbf W \in \mathbb R^{k\times m\times n}$，$\mathbf b \in \mathbb R^{n}$，$\mathbf V \in \mathbb R^{k\times m\times n}$，$\mathbf c \in \mathbb R^{n}$是学习参数，$\sigma$是sigmoid函数，$\otimes$是矩阵间的元素级乘法。