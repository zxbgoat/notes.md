**习题7.1**：n-步误差就是$t+n-1$时刻的误差，在$t+n$时刻获得；并且有$V_t(s)=V_{t+1}(s)=\cdots=V_{t+n}(s)$，因此：
$$
\begin{eqnarray}
&&G_{t:t+n} - V_{t+n-1}(S_t)\\
&=& R_{t+1} + \gamma G_{t+1:t+n} - V_{t+n-1}(S_t)\\
&=& R_{t+1} + \gamma V_t(S_{t+1}) - V_{t+n-1}(S_t) - \gamma V_t(S_{t+1}) + \gamma G_{t+1:t+n}\\
&=& R_{t+1} + \gamma V_t(S_{t+1}) - V_t(S_t) - \gamma V_t(S_{t+1}) + \gamma G_{t+1:t+n}\\
&=& \delta_t  + \gamma[R_{t+2} + \gamma V_{t+1}(S_{t+2}) - V_t(S_{t+1})] + \gamma^2G_{t+2:t+n} - \gamma^2V_{t+1}(S_{t+2})\\
&=& \delta_t + \gamma[R_{t+2} + \gamma V_{t+1}(S_{t+2}) - V_{t+1}(S_{t+1})] + \gamma^2G_{t+2:t+n} - \gamma^2V_{t+1}(S_{t+2})\\
&=& \delta_t + \gamma\delta_{t+1} + \gamma^2[R_{t+3} + \gamma V_{t+2}(S_{t+3})-V_{t+1}(S_{t+2})] + \gamma^3G_{t+3:t+n} - \gamma^3V_{t+2}(S_{t+3})\\
&=& \delta_t + \gamma\delta_{t+1} + \cdots + \gamma^{n-1}\delta_{t+n-1} + \gamma^nG_{t+n:t+n} - \gamma^nV_{t+n-1}(S_{t+n})\\
&=& \sum_{k=0}^{n-1} \gamma^k\delta_{t+k+1}
\end{eqnarray}
$$
因此也满足公示(6.6)。



**习题7.2**：价值逐步变化的n-步方法，此时的n-步误差就是：
$$
\begin{eqnarray}
&&G_{t:t+n} - V_{t+n-1}(S_t)\\
&=& R_{t+1} + \gamma G_{t+1:t+n} - V_{t+n-1}(S_t)\\
&=& R_{t+1} + \gamma V_t(S_{t+1}) -V_t(S_t) + \gamma G_{t+1:t+n} - \gamma V_t(S_{t+1}) + V_t(S_t) - V_{t+n-1}(S_t)\\
&=& \delta_t + \gamma G_{t+1:t+n} - \gamma V_t(S_{t+1}) + V_t(S_t) - V_{t+n-1}(S_t)\\
&=& \delta_t + \gamma\bigl[R_{t+2}+\gamma V_{t+1}(S_{t+2})-V_{t+1}(S_{t+1})\bigr] +\gamma^2G_{t+2:t+n} - \gamma^2V_{t+1}(S_{t+2}) \\
&&\qquad+ \bigl[\gamma V_{t+1}(S_{t+1}) - \gamma V_t(S_{t+1})\bigr] + V_t(S_t) - V_{t+n-1}(S_t)\\
&=& \delta_t + \gamma\delta_{t+1} +\gamma^2G_{t+2:t+n} - \gamma^2V_{t+1}(S_{t+2}) +\gamma\delta'_t + V_t(S_t) - V_{t+n-1}(S_t)\\
&=& \delta_t + \gamma\delta_{t+1} +\gamma^2G_{t+2:t+n} - \gamma^2V_{t+1}(S_{t+2}) +\gamma\delta'_t + V_t(S_t) - V_{t+n-1}(S_t)\\
\end{eqnarray}
$$
使用TD误差和来代替(7.2)中的误差，于是新算法的更新为：
$$
V_{t+n}(S_t) = V_{t+n-1}(S_t) + \alpha\left[\sum_{k=1}^n \gamma^{k-1}\delta_{t+k}\right]
$$



**习题7.3**



**习题7.4**：这个算法具体的步骤为：
$$
\bbox[5px,border:2px solid]
{\begin{aligned}
  &\underline{\mathbf{Per\text-Decision\ n\text-step\ TD\ for\ estimating\ }V\approx v_\pi}\\
  \\
  &\text{Input: a policy }\pi\\
  &\text{Initialize }V(s)\text{ arbitrarily },s\in\mathcal S\\
  &\text{Parameters: step size }\alpha\in(0,1],\text{ a positive integer n}\\
  &\text{All store and access operations (for }S_t\text{ and }R_t\text{) can take their index mod }n+1\\
  \\
  &\text{Loop for each episode:}\\
  &\qquad \text{Initialize and store }S_0\neq \text{terminal}\\
  &\qquad T \leftarrow \infty\\
  &\qquad \text{For }t=0,1,2,\dots\text{:}\\
  &\qquad|\qquad\text{if }t < T,\text{ then:}\\
  &\qquad|\qquad\qquad \text{Take an action according to }b(\bullet\mid S_t)\\
  &\qquad|\qquad\qquad \text{Observe and store next reward as }R_{t+1}\text{ and the next state as }S_{t+1}\\
  &\qquad|\qquad\qquad \text{If }S_{t+1}\text{ is terminal, then }T \leftarrow t+1\\
  &\qquad|\qquad \tau \leftarrow t-n+1\quad(\tau\text{ is the time whose states's estimate is being updated})\\
  &\qquad|\qquad\text{if }\tau\ge0\text{:}\\
  &\qquad|\qquad\qquad G\leftarrow\sum_{i=\tau+1}^{\min(\tau+n, T)}\gamma^{i-\tau-1}R_i\\
  &\qquad|\qquad\qquad \text{If }\tau+n < T, \text{then: }G\leftarrow G+\gamma^nV(S_{\tau+n})\qquad\qquad\qquad(G_{\tau:\tau+n})\\
  &\qquad|\qquad\qquad V(S_\tau) \leftarrow V(S_\tau) + \alpha[G - V(S_\tau)]\\
  &\qquad \text{Until }\tau=T-1
\end{aligned}}
$$
