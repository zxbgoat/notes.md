##### 间隔与支持向量

给定样本集$D=\{(\mathbf x_1, y_1),(\mathbf x_2, y_2),\dots,(\mathbf x_m,y_m)\}, y_i\in\{-1,+1\}$，分类学习最基本的思想就是在基于训练集$D$的样本空间找到一个划分超平面，将不同的样本分开。且直观上应该找那些位于两类训练样本正中间的划分超平面，因为其对训练样本的局部扰动“容忍性”最好。在样本空间中，划分超平面可以通过如下线性方程描述：
$$
\mathbf w^T\mathbf x + b = 0
$$
其中$\mathbf w=(w_1,w_2,\dots,w_d)$为法向量，决定了超平面的方向；$b$为位移项，决定了超平面与原点间的距离；可将超平面记为$(\mathbf w,b)$。样本空间中任意点$\mathbf x$到超平面$(\mathbf w,b)$的距离可以写为：
$$
r = \frac{\left\vert \mathbf w^T\mathbf x + b \right\vert}{\Vert \mathbf w \Vert}
$$
