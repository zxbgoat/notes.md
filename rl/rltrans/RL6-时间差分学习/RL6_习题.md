**习题6.1**  $V$在一节中变化时，比如TD(0)更新：
$$
V_{t+1}(S_t) \leftarrow V_t(S_t) + \alpha\Bigl[ R_{t+1}+\gamma V_t{(S_{t+1})} -V_t(S_t) \Bigr]
$$
因需在$t+1$步时才获得$R_{t+1}$和$S_{t+1}$，此时$S_{t+1}$价值还未更新，TD误差为：
$$
\delta_t \dot= R_{t+1} + \gamma V_t(S_{t+1}) - V_t(S_t)
$$
同理MC误差为：
$$
\begin{eqnarray}
G_t - V_t(S_t)
&=& R_{t+1} + \gamma G_{t+1} - V_t(S_t) +\gamma V_t(S_{t+1}) - \gamma V_t(S_{t+1})\\
&=& \delta_t +\gamma\bigl(G_{t+1}-V_t(S_{t+1})\bigr)\\
&=& \delta_t + \gamma\bigl(R_{t+2}+\gamma G_{t+2} - V_t(S_{t+1})+\gamma V_{t+1}(S_{t+2})-\gamma V_{t+1}(S_{t+2})\bigr)\\
&=& \delta_t + \gamma\Bigl[\bigl( R_{t+2}+\gamma V_{t+1}(S_{t+2})-V_{t+1}(S_{t+1})\bigr) + \gamma\bigl(G_{t+2}-V_{t+1}(S_{t+2})\bigr) +\bigl(V_{t+1}(S_{t+1})-V_t(S_{t+1})\bigr) \Bigr]\\
&=& \delta_t + \gamma\delta_{t+1} + \gamma^2\bigl[G_{t+2}-V_{t+1}(S_{t+2})\bigr] + \gamma\alpha\delta_t\\
&=& \delta_t+\gamma\delta_{t+1} + \gamma^2\delta_{t+2} +\gamma^3\bigl[G_{t+3}-V_{t+2}(S_{t+3})\bigr] +\gamma\alpha\delta_t +\gamma^2\alpha\delta_{t+1}\\
&=& \sum_{k=t}^{T-1}\gamma^{k-t}\delta_k + \alpha\sum_{k=t+1}^T\gamma^{k-t}\delta_{k-1}
\end{eqnarray}
$$
因此当$\alpha$很小时(6.6)也近似成立。



**习题6.2** 



**习题6.3**：从$\mathtt A$直接到了0最左变的终结状态。