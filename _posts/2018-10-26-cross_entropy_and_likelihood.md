
---
title: "Cross entropy, Likelihood 그리고 mse"
date: 2018-10-30 08:26:28 -0400
categories: .

---

###### Information

$$\begin{align}
\mathbb{I}(X=x_k) & \triangleq - \log p(X=x_k) \\
& = - \log p_k
\end{align}$$

###### Entropy

$$\begin{align}
\mathbb{H}(X) &\triangleq \mathbb{E} \big[ \mathbb{I}(X=x_k) \big] \\
& = -\sum_{k=1}^K p(X=x_k) \log p(X=x_k) \\
& = -\sum_{k=1}^K p_k \log p_k \\
\end{align}$$

###### Cross Entropy

$$\begin{align}
\mathbb{H}(q, p) &\triangleq \mathbb{E} \big[ \mathbb{I}(X=x_k) \big] \\
& = -\sum_{k=1}^K q(X=x_k) \log p(X=x_k) \\
& = -\sum_{k=1} q_k \log p_k \\
\end{align}$$

> $\frac{1}{N}\text{NLL}$ = cross entropy

>  - $X$: random variable for categorical events, $\{ c_1, \dots, c_i, \dots, c_n \}$.
 - $p$: hypothetical distribution for $X$, $p(X=c_i) = y_i$
 - $\text{data}$: total $N$ times of trial, which is $\{ k_1 \text{ times of } c_1,  \dots, k_n \text{ times of } c_n\}$
 - $q$: empirical distribution for $X$, $q(X=c_i) = \frac{k_i}{N} = y'_i$

> - likelihood for $data$ under a $model$ : 
$$P[data \mid model] = y_1^{k_1} \cdots y_i^{k_i} \cdots y_n^{k_n}$$     
> - negative log likelihood(divided by $N$): 
$$\begin{align}
-\frac{1}{N} \log P[data \mid model] &= -\frac{1}{N} \bigg( k_1 \log y_1 + k_2 \log y_2 + \cdots + k_n \log y_n \bigg) \\
& = -\frac{1}{N} \sum_i k_i \log y_i \\
& = -\sum_i y'_i \log y_i \\
& = \mathbb{H}(y', y)
\end{align}$$

> 예시(범주형)
- 각 면의 나올 확률이 동일하지 않은 동전을 5번 던진다고 해 보자. 동전을 던지기 전 예상으로는 앞면이 나올 확률이 $3/7$, 뒷면이 나올 확률이 $4/7$이다. 동전을 5번 던진 후 데이터와 비교하여 예상했던 확률의 cross entropy 혹은 negative log likelihood를 생각해보자. (loss라고 봐도 무방하다. 예상이 실제 확률에 가까울수록 이 값이 작아질 것이다.) 5번 던진 결과 앞면이 1번 뒷면이 4번 나왔다면, 실험으로 얻어진 확률은 $(1/5, 4/5)$이다.
- 이때의 cross entropy는 아래와 같다.
$$\begin{align}
\text{NLL} &= -\frac{1}{5} \bigg( 1 \cdot \log \frac{3}{7} + 4 \cdot \log \frac{4}{7} \bigg) \\
&\approx 0.617
\end{align}$$
- 만약 최초 예상이 (실험 결과에 조금 더 가깝게) 앞면이 나올 확률은 $2/7$, 뒷면이 나올 확률은 $5/7$이라면 $\text{NLL}$은 아래와 같다. 앞서 결과와 비교해보면 $\text{NLL}$ 값이 조금 더 작다. 즉 $\text{NLL}$을 최소화하면 실제에 가까운 파라미터를 찾아가게 된다.
$$\begin{align}
\text{NLL} &= -\frac{1}{5} \bigg( 1 \cdot \log \frac{2}{7} + 4 \cdot \log \frac{5}{7} \bigg) \\
&\approx 0.520
\end{align}$$

> 예시(연속형)
- 낚시 지렁이들을 키우고 있는데, 어느정도 자란 녀석들의 길이를 측정한다고 해보자. 지렁이들의 길이를 측정하기 전에 지렁이 길이($G$)의 분포는 $G \sim \mathcal{N}(12, 3^2)$일것이라 예상한다. 실제로 지렁이 5마리의 키를 측정했더니 $(g_1=13, g_2=12, g_3=15, g_4=14, g_5=15)$이었다. 이때의 cross-entropy(=NLL)은?

>-  $$\begin{align} 
\text{NLL} &= -\frac{1}{5} \bigg( \log p(g_1=13 \mid G \sim \mathcal{N}(12, 3^2)) + \cdots + \log p(g_5=15 \mid G \sim \mathcal{N}(12, 3^2)) \bigg) \\
&= -\frac{1}{5} \bigg( \log(1/\sqrt{2 \cdot \pi \cdot 3^2}) - \frac{(13-12)^2}{2 \cdot 3^2} + \cdots + \log(1/\sqrt{2 \cdot \pi \cdot 3^2}) - \frac{(15-12)^2}{2 \cdot 3^2} \bigg) \\
&= -\frac{1}{5} \bigg(  5\log(1/\sqrt{2 \cdot \pi \cdot 3^2}) -\sum_{i=1}^5 \frac{(g_i - 12)^2}{2 \cdot 3^2}  \bigg)
\end{align}$$

> - 추정하려는 분포가 정규분포라면, NLL와 MSE는 단지 상수 만큼 차이가 있으므로 구현시에는 그냥 MSE를 사용하는 이유를 위에서 확인할 수 있다. 또한 NLL, cross-entropy, MSE를 최소화 하는 $G$의 파라미터를 찾는게 결국 Maximum likelihood estimation이다.

> - Reference
 - https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks?newreg=68e217d2659e4391914bf7ee19217138
 - https://taeoh-kim.github.io/blog/cross-entropy의-정확한-확률적-의미/

###### KL Divergence

- KL divergence = relative entropy

> $$\begin{align}
\mathbb{KL}(p \Vert q) &\triangleq \sum_{k=1}^K p_k \log \frac{p_k}{q_k} \\
&= \sum_k p_k \log p_k - \sum_k p_k \log q_k \\
&= -\mathbb{H}(p) + \mathbb{H}(p, q)
\end{align}$$
