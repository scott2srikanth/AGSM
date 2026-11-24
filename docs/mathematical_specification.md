# Mathematical specification

For input embedding $x_t$ and previous context $h$, define
$$
u=\mathrm{LN}(W_x x_t+W_h h),\quad
q=\operatorname{normalize}(W_q u),\quad
k=\operatorname{normalize}(W_k u),\quad v=W_v u.
$$
Address normalization divides by a norm floored at $10^{-6}$. The fast state $S$ is initialized to zero. From the previous state compute
$$
r=S q,\quad e=v-S k,\quad \rho=\frac{\lVert e\rVert_2}{\max(\lVert v\rVert_2,10^{-6})}.
$$
The denominator uses a floor in place of adding epsilon. A shared GELU feature network conditions five sigmoid outputs $g,a,\gamma,\beta,\lambda$. The controller receives $u,h,r,v,\log(1+\rho),\lVert S\rVert_F$, with optional ablation of the last two features.

$$
S'=\lambda S + (g\beta)e k^\top,\quad
\tilde h=\operatorname{GRUCell}([u,r],h),\quad h'=h+a(\tilde h-h).
$$

Optional clipping rescales the fast matrix to a Frobenius-norm cap. Reads and delta updates use FP32 for low-precision storage and FP64 for FP64 memory. Padding keeps both recurrent tensors unchanged at that position. Fast-only and slow-only ablations keep the disabled bank at its zero initialization and preserve tensor interfaces.

Let $f=[u,h',\gamma r]$, $z_0=W_z f$, and $z_j=\operatorname{LN}(z_{j-1}+\mathrm{MLP}([z_{j-1},f]))$. The controller's shared context and $z_j$ parameterize each conditional stop probability $p_j$. The stop mass at depth $j$ is $m_j=s_{j-1}p_j$, where $s_0=1$ and $s_j=s_{j-1}-m_j$. The last allowed depth receives all remaining mass. Training predicts from $\sum_j m_j z_j$ and penalizes $\sum_j j m_j$. Every depth is executed during soft training.

A cumulative-additive ACT scheme and a conditional stopping distribution are both possible. The conditional form makes mass conservation and the expected-depth interpretation explicit. Inference stops a row once cumulative mass meets the threshold and assigns that row's remaining probability mass to the current representation. The next MLP invocation runs only on active rows. Inference expected depth is computed using the resulting truncated distribution. Fixed-depth mode assigns unit mass to the last depth. Zero depth returns the initial projection.

The objective adds weighted expected depth, mean effective write strength $g\beta$, and squared excess fast/slow norms to the task loss. Weights and norm thresholds are resolved from YAML. Synthetic classification and retrieval use cross-entropy at labeled positions. The regression loss uses scalar squared error. The language model predicts the next byte with cross-entropy.
