#### [情感聊天机器ECM](https://arxiv.org/abs/1704.01074)

情感智能是对话系统或会话代理成功的关键因素。本文提出的情感聊天机器(Emotional Chatting Machine)能够产生无论内容层次（相关性和语法性）还是情感层次（一致的情感表达）都合适的回应。就目前所知，这时首个考虑情感因素的大规模对话生成。ECM用三种方式处理情感因子：通过嵌入情感类别为情感的高层抽象建模，获取含蓄的内部情感状态变化，以及使用带外部情感词汇的明确情感表达。

##### 1.简介

要创造以人类水平与用户表现和交谈的机器人需要机器人理解人类的认知行为，而人类行为中最重要的一个就是表达和理解情感和影响。作为人类智能必不可少的部分，情感智能定义为察觉、结合、理解和控制情感的能力。要创造以人类水平与用户交流的机器，就必须使机器拥有情感智能。

本文提出了一个在大规模序列到序列生成的编码器-解码器框架中能有情感地反馈用户的模型，主要的贡献为：

- 提出了在大规模序列到序列对话生成中结合情感影响的端到端框架，同时提出了三种新机制：情感类别嵌入、内部存储、和外部存储。
- 展现了ECM能产生比传统seq2seq模型更自然和情感智能的回应，未来的同感计算机代理和情感交互明显可以以此为基础施行。

##### 2.相关工作

**情感智能**：已有一些向对话系统或会话代理赋予情感智能的尝试。已证实在人类与智能代理交互过程中，察觉人类情感的迹象并能适当反馈的能力能丰富交流的体验，比如增加用户满意度，得到更积极的感觉，减少许多冷场，将对话行为调整为用户情感状态。但这些都是受物理发现的启发，要么是基于规则的，要么受限于小规模数据。

**基于大规模序列到序列的对话生成**：随着序列到序列生成模型在机器翻译中的成功，这些模型也应用在了对话生成，包括[神经回应机器](http://www.aclweb.org/anthology/P15-1152)，[层次循环编码-解码神经网络](https://www.researchgate.net/publication/280221106_Hierarchical_Neural_Network_Generative_Models_for_Movie_Dialogues)等。但已有的工作关注的都是改进生成回应的内容质量。为避免无用或通用的回答提出了最大互信息(Maximum Mutual Information, MMI)规则作为最大似然估计的替代。集束搜索也广泛地用于产生有意义和多样化的回应。其他改进内容质量的努力还包括结合进解码器附加主题词语，人物角色信息，或其他获得的回应等。另一条路径是使用强化学习来对聊天机器人对话中的长期延迟奖励建模。处理未知词语也可以改进生成质量。但还没有工作致力于生成内容相关和情感一致的回复。

**基于存储器的网络**：存储网络与神经图灵机用存储结构改进了对大范围序列建模的能力。本文的模型采用了一个动态存储器来模拟内部情感状态的变化，和一个静态的存储器来存储情感词语的字典。

##### 3.背景：序列到序列生成

先介绍基于序列到序列(seq2seq)学习的[通用编码-解码框架](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)。这里编码器和解码器使用GRU实现。编码器奖文本序列$\mathbf X=(x_1,x_2,\cdots,x_n)$转化为隐式表达$\mathbf h=(h_1,h_2,\cdots,h_n)$，可以简单定义为：
$$
h_t = \mathbf{GRU}(h_{t-1}, x_t)
$$

解码器输入上下文向量$c_t$和前一个解码词语嵌入$e(y_{t-1})$，并使用另一个GRU更新其状态$s_t$：
$$
s_t = \mathbf{GRU}(s_{t-1},[c_t;e(y_{t-1})])
$$
其中$[c_t;e(y_{t-1})]$是两个向量连接，作为GRU的输入。上下文向量$c_t$设计用于在解码时动态关注输入文本的关键信息，它依赖于解码器的前一个状态$s_{t-1}$：
$$
\begin{aligned}
c_t &= \sum_{k=1}^n \alpha_{tk} h_k \\
\alpha_{tk} &= \frac{\exp(e_{tk})}{\sum_{j=1}^n \exp(e_{tj})} \\
e_{tk} &= v_a^T \tanh(W_as_{t-1}+U_ah_k)
\end{aligned}
$$
其中$\alpha_{tk}$可视为每个文本隐状态$h_k$和解码器的状态$s_{t-1}$的相似度。