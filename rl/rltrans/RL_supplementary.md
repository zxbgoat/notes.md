##### MDP

二、关于最优策略的理解：

1. 策略偏序关系定义为，当且仅当对所有的$s\in\mathcal S$都有$v_\pi(s)\ge v_{\pi'}(s)$时，则$\pi\ge\pi'$；

2. 总存在一个或多个策略优于或等于其他所有策略，这就是最优策略，记为$\pi_*$；

3. 所有最优策略都有同样的最优状态价值函数$v_*(s)\dot=\max_\pi v_\pi(s)$；

4. 所有最优策略都有同样的最优行动价值函数，表示在状态$s$采取行动$a$后遵循最优策略所能得到的期望回报：
   $$
   \begin{eqnarray}
   q_*(s,a) &\dot=& \max_\pi q_\pi(s,a)\\
   &=& \max_\pi \mathbb E_\pi\bigl[ G_t \mid S_t=s, A_t=a \bigr]\\
   &=& \max_\pi\mathbb E_\pi\bigl[ R_{t+1} + \gamma G_{t+1} \mid S_t=s, A_t=a \bigr]\\
   &=& \max_\pi \mathbb E_\pi\bigl[ R_{t+1} + \gamma G_{t+1} \mid S_{t+1}, S_t=s,A_t=a \bigr]\tag{1.7}\\
   &=& \mathbb E\bigl[ R_{t+1}+\gamma v_*(S_{t+1}) \mid S_t=s,A_t=a \bigr] \tag{1.8}
   \end{eqnarray}
   $$
   (1.7)是因为条件为$S_t=s,A_t=a$时就可以引出后继状态$S_{t+1}$；还有这里形式是$\mathbb[G_{t+1}\mid S_t=s,A_t=a]$，回报与状态行动不在同一时间步，因此不是$q(s,a)$。注意这里是采取行动$a$后再遵循最优策略，在状态$s$采取行动$a$很可能并不属于最优策略。



三、状态价值函数的贝尔曼最优性方程：
$$
\begin{eqnarray}
v_*(s) &=& \max_{a\in\mathcal A(s)}q_{\pi_*}(s,a)\tag{1.9}\\
&=& \max_{a} \mathbb E\bigl[R_{t+1} +\gamma v_*(S_{t+1}) \mid S_t=s, A_t=a\bigr]\tag{1.10}\\
&=& \max_a \sum_{s',r} p(s',r \mid s,a)\left[ r+\gamma v_*(s') \right]\tag{1.11}
\end{eqnarray}
$$
行动价值函数的贝尔曼最优性方程为：
$$
\begin{eqnarray}
q_*(s,a) &=& \mathbb E\bigl[ R_{t+1}+\gamma v_*(S_{t+1}) \mid S_t=s,A_t=a \bigr] \tag{1.12}\\
&=& \mathbb E\left[ R_{t+1}+\gamma \max_{a'}q_*(S_{t+1},a') \middle | S_t=s,A_t=a \right]\tag{1.13}\\
&=& \sum_{s',r} p(s',r \mid s,a) \left[ r+\gamma\max_{a'}q_*(s',a') \right]\tag{1.14}
\end{eqnarray}
$$
(1.10)和(1.12)可由(1.8)获得。最优状态价值函数、最优行动价值函数和最优策略之间的关系为：
$$
\begin{eqnarray}
v_*(s) &=& \max_a q_*(s,a)\tag{1.15}\\
q_*(s,a) &=& \sum_{s',r} p(s',r \mid s,a) \left[ r+\gamma v_*(s') \right]\tag{1.16}\\
\pi_*(s) &=& \arg\max_a q_*(s,a)\tag{1.17}\\
\pi_*(s) &=& \arg\max_a \sum_{s',r} p(s',r \mid s,a) [r+\gamma v_*(s')]\tag{1.18}
\end{eqnarray}
$$
(1.15)可由(1.9)获得，(1.16)可由(1.11)获得。策略就是将状态映射为动作的函数，因此获得最后两个公式。由(1.17)，在获得所有$q_*$后，取每个状态$s$所有行动$a$中$q_*(s,a)$最大者即可确定策略；由(1.18)，在获得所有$v_*$后，通过向前一步搜索（不需要知道$q_*$），也可确定策略。



##### DP

一、任意策略$\pi$的状态价值函数$v_\pi$计算公式为：
$$
\begin{eqnarray}
v_\pi(s) &=& \mathbb E_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t=s]\\
&=& \mathbb E_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t=s] \tag{2.1}\\
&=& \sum_a \pi(a \mid s) \sum_{s',r} p(s',r \mid s,a) \bigl[ r+\gamma v_\pi(s') \bigr] \tag{2.2}
\end{eqnarray}
$$
将最后两个改成迭代式为
$$
\begin{eqnarray}
v_{k+1}(s) &\dot=& \mathbb E_\pi\bigl[R_{t+1} + \gamma v_k(S_{t+1}) \mid S_t=s\bigr]\\
&=& \sum_a\pi(a\mid s)\sum_{s',r} p(s',r \mid s,a) \bigl[ r+\gamma v_k(s') \bigr] \tag{2.3}
\end{eqnarray}
$$
对应计算$q_\pi$的公式为：
$$
\begin{eqnarray}
q_\pi(s,a) &=& \mathbb E_\pi [R_{t+1} + \gamma G_{t+1} \mid S_t=s, A_t=a]\\
&=& \mathbb E_\pi \bigl[R_{t+1} + \gamma \mathbb E_\pi[v_\pi(S_{t+1})] \mid S_t=s, A_t=a\bigr]\\
&=& \mathbb E_\pi \left[ R_{t+1} + \gamma \sum_{A_{t+1}}\pi(A_{t+1}\mid S_{t+1})q_\pi(S_{t+1},A_{t+1}) \middle | S_t=s, A_t=a \right] \tag{2.4}\\
&=& \sum_{s',r} p(s',r \mid s,a) \left[ r+\gamma\sum_{a'}\pi(s'\mid a')q_\pi(s',a') \right] \tag{2.5}
\end{eqnarray}
$$
最后的迭代公式为：
$$
\begin{eqnarray}
q_{k+1}(s,a) &=& \mathbb E_\pi \left[ R_{t+1} + \gamma \sum_{A_{t+1}}\pi(A_{t+1}\mid S_{t+1})q_k(S_{t+1},A_{t+1}) \middle | S_t=s, A_t=a \right]\\
&=& \sum_{s',r} p(s',r \mid s,a) \left[ r+\gamma\sum_{a'}\pi(s'\mid a')q_k(s',a') \right] \tag{2.6}
\end{eqnarray}
$$

二、策略改善定理的理解

若已确定了某一策略$\pi$的状态-价值函数$v_\pi$，希望了解在某个状态$s$选择行为$a\neq\pi(s)$会得到与原来相比怎样的结果，则可以由(1.6)，即：
$$
\begin{eqnarray}
q_\pi(s,a) &\dot=& \mathbb E_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t=s,A_t=a] \tag{2.7}\\
&=& \sum_{s',r} p(s',r \mid s,a) \bigl[ r+\gamma v_\pi(s') \bigr]
\end{eqnarray}
$$
计算在$s$选择$a$然后遵循策略$\pi$所获得的价值$q_\pi(s,a)$，再与$v_\pi(s)$比较，若$q_\pi(s,a)>v_\pi(s)$，则在$s$选择$a$，其余状态不变所形成的新策略$\pi'$必然是比$\pi$更好的策略。

(2.7)的理解可参照(1.8)。因$v_{\pi'}(s)>v_\pi(s)$，而所有其他状态价值的计算都会直接或间接地涉及到它，作为和的一部分，因此必大于等于原来的状态价值，因此在所有的状态上$\pi'$的价值都大于$\pi$，因此是更好的策略。另外，一旦确定好策略$\pi$后，就有$\forall s\in \mathcal S, v_\pi(s)=q_\pi(s,\pi(s))$。

策略改善定理：若$\pi$和$\pi'$为两个确定性策略并对$\forall s \in \mathcal S$满足：
$$
q_\pi(s, \pi'(s)) \ge v_\pi(s) \tag{2.8}
$$
则策略$\pi'$必然是比$\pi$更好，即对$\forall s \in \mathcal S$，都有：
$$
v_{\pi'}(s) \ge v_\pi(s) \tag{2.9}
$$
并且若有任一状态在(2.8)严格不等，则在(2.9)中至少存在一个严格不等式，因此很自然就想到下面的贪心策略：
$$
\begin{eqnarray}
\pi_*(s) &=& \arg\max_a q_\pi(s,a) \tag{2.10}\\
&=& \arg\max_a \mathbb E[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t=s,A_t=a] \tag{2.11}\\
&=& \arg\max_a \sum_{s',r} p(s',r \mid s,a) [r+\gamma v_\pi(s')] \tag{2.12}
\end{eqnarray}
$$
这样通过**对原策略价值函数贪心**的改善原策略的过程，就是策略改善。若新策略$\pi'$等优但不优于$\pi$，则$v_{\pi'}=v_\pi$，并由(1.17)，对$\forall s \in \mathcal S$有：
$$
v_{\pi'}(s) = \max_a \sum_{s',r} p(s',r \mid s,a)\left[ r+\gamma v_*(s') \right]
$$
这等价于贝尔曼最优性方程(1.11)，因此必有$v_{\pi'} = v_\pi$，因此$\pi=\pi'=\pi_*$。因此**策略改善必给出一个严格更优的策略，除非原策略已是最优**。

三、基于(2.6)和(2.10)实现策略迭代的行动价值函数版，包括改善多个最优策略死循环问题：
$$
\bbox[5px,border:2px solid]
{\begin{aligned}
 &\underline{\text{Policy Iteration with }q_\pi}\\
 \\
&1.\text{ Initialization}\\
&\quad V(s,a) \in \mathbb R \text{ arbitrarily}, \forall s\in\mathcal S \text{ and }a\in\mathcal A(s)\\
&\quad \pi(s) \in \mathcal A(s) \text{ arbitrarily}, \forall s\in\mathcal S\\
\\
&2.\text{ Policy Evaluation}\\
&\quad\text{Loop:}\\
&\quad\qquad \Delta \leftarrow 0\\
&\quad\qquad \text{Loop for each }s \in \mathcal S:\\
&\quad\qquad\qquad \text{Loop for each }a \in \mathcal A(s):\\
&\quad\qquad\qquad\qquad v \leftarrow V(s,a)\\
&\quad\qquad\qquad\qquad V(s,a) \leftarrow \sum_{s',r} p(s',r \mid s,a) \left[ r+\gamma\sum_{a'}\pi(s'\mid a')V(s',a') \right]\\
&\quad\qquad\qquad\qquad \Delta = \max\bigl(\Delta, \vert v-V(s,a) \vert\bigr)\\
&\quad\text{until }\Delta < \theta\\
\\
&3.\text{ Policy Improvement}\\
&\quad policy\text-stable \leftarrow true\\
&\quad \text{Loop for each } s \in \mathcal S:\\
&\quad\qquad old\text-value \leftarrow V\bigl(s,\pi(s)\bigr)\\
&\quad\qquad \pi(s) \leftarrow \arg\max_a V(s,a)\\
&\quad\qquad \text{If }old\text-value \neq V\bigl(s,\pi(s)\bigr), \text{ then }policy\text-stable\leftarrow false\\
&\quad \text{If }policy\text-stable, \text{then stop and return }V \approx v_*; \text{ else go to }2
\end{aligned}}
$$
改善的基本思想是用价值函数代替行动来判别策略是否稳定，因所有最优策略的价值函数都相同。

四、确定性策略，随机性策略，$\varepsilon$-松弛策略

$\varepsilon$-松弛策略要求，在每个状态$s$至少以$\frac\varepsilon{\vert\mathcal A(s)\vert}$的概率选择$\mathcal A(s)$中的每个行动；

确定性策略就是在每个状态都会给出固定的行动的策略；

随机性策略就是在每个状态从所有行动的分布中抽样的行动；

预测问题就是评估一个策略的价值函数；

控制问题就是改善任务的策略直到最优。

五、行动的价值迭代
$$
\begin{eqnarray}
q_{k+1}(s,a) &=& \mathbb E\left[ R_{t+1}+\gamma \max_{A_{k+1}}q_k(S_{t+1},A_{k+1}) \middle | S_t=s,A_t=a \right]\\
&=& \sum_{s',r} p(s',r \mid s,a) \left[ r+\gamma\max_{a'}q_k(s',a') \right] \tag{2.13}
\end{eqnarray}
$$
依据(2.13)可获得算法为：
$$
\bbox[5px,border:2px solid]
{\begin{aligned}
 &\underline{\text{Value Iteration with }q_\pi}\\
 \\
&\text{Initialize }V(s,a), \text{ for } \forall s\in\mathcal S\text{ and }a\in\mathcal A(s), \text{ arbitrarily}\\
\\
&\text{Loop:}\\
&\qquad\Delta \leftarrow 0\\
&\qquad\text{Loop for each }s \in \mathcal S:\\
&\qquad\qquad v \leftarrow V(s,a)\\
&\qquad\qquad V(s,a) \leftarrow \sum_{s',r} p(s',r \mid s,a) \left[ r+\gamma\max_{a'}V(s',a') \right]\\
&\qquad\qquad\Delta \leftarrow \max(\Delta,\vert v-V(s,a)\vert)\\
&\text{until }\Delta < \theta\\
\\
&\text{Output a deterministic policy},\pi\approx\pi_*, \text{such that}\\
&\qquad\pi(s) \leftarrow \arg\max_aV(s,a)
\end{aligned}}
$$

六：证明$v_\pi(s) = \mathbb E_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t=s]$
$$
\begin{eqnarray}
v_\pi(s)
&=& \mathbb E_\pi[R_{t+1}+\gamma G_{t+1} \mid S_t=s]\\
&=& \sum \pi(S_{t+1} \mid S_t=s)\mathbb E[R_{t+1} + \gamma G_{t+1} \mid S_{t+1}]\\
&=& \sum \pi(S_{t+1} \mid S_t=s)[\mathbb E(R_{t+1}) + \gamma v_\pi(S_{t+1})]\\
&=& \mathbb E_\pi[R_{t+1}+\gamma v_\pi(S_{t+1}) \mid S_t=s]
\end{eqnarray}
$$
