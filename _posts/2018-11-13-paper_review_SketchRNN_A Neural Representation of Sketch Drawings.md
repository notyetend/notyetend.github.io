
SketchRNN이라는 이름으로 더 잘 알려진 논문인 'A Neural Representation of Sketch Drawings'을 리뷰하려 한다. bidirectional RNN과 VAE를 혼합해서 스케치를 생성하거나 못 그린 그림을 마져 그리게 하는게 흥미로운 논문이다.

# Abstract

- 그림, 그중에서도 펜으로 그리는 그림(stoke-based drawings)을 생성하는 sketch-rnn이란 모델을 소개한다. 
- 이 모델은조건부 스케치 생성 모델(conditional sketch generation model)과 무-조건부 스케치 생성 모형(unconditional sketch generation model)으로 나뉜다.
- 벡터 형식으로 된 스케치 데이터를 잘 학습시키기 위한 방법을 소개한다.

# 1. Introduction

- 요즘 GAN, Variational Inference(VI), Autoregressive(AR)과 같은 이미지를 생성하는 아래와 같은 모형이 붐이다. 그런데 이런 모형들은 픽셀 이미지를 생성하는 것들이다. 반면 이 논문에서는 펜 스트록 이미지(스케치)와 같은 벡터 이미지를 학습하고 생성하는 모형을 소개한다.
 - Figure 1

- 펜으로 그린 그림 데이터는 각 선을 (이전 점으로부터 움직인 xy값, 현재 점 이후 펜을 종이에서 뗄 것인지, 현재 점을 마지막으로 그림을 다 그렸는지)로 표현하며 이런 선들이 모여 하나의 그림 데이터가 된다.(3.1에서 자세히 설명)

- Contributions
 - 조건부 생성 모형(conditional generation model)과 무-조건부 생성 모형(unconditional generation model)
 - 벡터 형식의 데이터를 생성하는 RNN 베이스 모형(LSTM, HyperLSTM이 사용됨)
 - 벡터 이미지를 잘(robust) 학습시키기 위한 학습 과정을 제시
 - 조건부 생성 모형을 이용해 잠재공간(latent space)의 각 점에 해당하는 이미지 생성
 - 사용된 데이터셋 공개(https://github.com/googlecreativelab/quickdraw-dataset)

# 2. Related Work

화가를 따라하는 아래 알고리즘들은 생성 모형이라기보다 그림을 따라 그리려는 시도였음.
- 그림을 주면 로봇 팔을 컨트롤해서 그림을 그림 그리게 하는 방법
- 사진을 주면 강화학습을 이용해 붓의 움직임을 찾아내는 방법

픽셀로 구성된 이미지를 학습하는 신경망 기반의 생성 모형이 많이 개발됨.

벡터 이미지를 이용하는 접근법은 아래와 같은 것들이 있었음. 
- HMM을 이용해 선과 곡선을 생성하는 모형
- 그림의 각 점을 Mixture Density Network로 모델링하고 RNN으로 학습시키는 모형
- 한자를 조건부/무-조건부로 생성하는 모형

(후략)

# 3. Methodology

### 3.1 Dataset

- QuickDraw라는 데이터셋을 만듬
- pen stroke action을 표현하는 데이터 포멧을 정의함, $(\Delta x, \Delta y, p_1, p_2, p_3)$
 - $\Delta x$: 이전 점으로부터 $x$ 좌표 변화량
 - $\Delta y$: 이전 점으로부터 $y$ 좌표 변화량
 - $p_1$: 펜의 상태로서 펜이 지금 종이에 닿아 있는지를 표시 (1이면 닿아 있고, 다음 점과 연결되는 선이 그려질 것임.)
 - $p_2$: 펜이 현재점 이후 뜰 것임을 표시 (1이면 현재점 이후 좋이에 펜이 떨어질 것이므로 다음 점과 선이 그려지지 않을 것임)... ?? 그럼 $p_1$과 $p_2$중 하나만 1인가??
 - $p_3$: 그림 다 그렸음을 표시 (1이면 다 그렸다.)
 - 예를들어 (3, 2, 1, 0, 0) 이면 이전 좌표에서 $x$축으로 3만큼, $y$축으로 2만큼 움직였고, 현재 펜이 종이에 닿아 있고, 이후 다음 점과 선이 열결될 것이며, 아직 그림이 끝나지 않았음을 의미


```python
import numpy as np
with np.load('/Users/notyetend/Downloads/sketch-rnn-data/sketchrnn%2Fbird.npz', encoding='latin1') as a:
    train = a['train']
    print(train[:1][0][:10])
```

    [[ -21    7    0]
     [ -66    9    0]
     [ -30   11    0]
     [   5   -3    0]
     [  75  -11    0]
     [  72   -1    1]
     [-142   11    0]
     [  79   21    0]
     [  51    2    0]
     [  56   12    0]]


### 3.2 Sketch-RNN

###### Sequence-to-Sequence VAE

- 이 모델은 Sequence-to-Sequence Variational Autoencoder 이다.(참고 논문: S. R. Bowman, L. Vilnis, O. Vinyals, A. M. Dai, R. Józefowicz, and S. Bengio. Generating Sentences from a Continuous Space. CoRR, abs/1511.06349, 2015)

###### Encoder RNN and VAE

- Forward Encoder RNN(LSTM)은 순방향 스케치 데이터 $S$를 입력으로 받아 $h_{\rightarrow}$를 출력으로 내놓고, Backward Encoder RNN(LSTM)은 역방향 스케치 데이터 $S_{\text{reverse}}$를 입력으로 받아 $h_{\leftarrow}$를 출력으로 내놓는다. 두 출력을 병합한 출력을 $h = [ h_{\rightarrow}; h_{\leftarrow} ]$라 한다. ($h_{\rightarrow}$과 $h_{\leftarrow}$ 모두 크기가 $N_z$인 벡터이다.)

- 병합된 출력 $h$를 VAE의 중앙 히든 레이어의 입력으로 넣는데, 이 중앙 히든 레이어는 각각 크기가 $N_z$인 $\mu$와 $\sigma$로 구성된다. 이후 $z \sim p(z \mid x)$를 샘플링하는 과정은 VAE의 문법을 따른다.

###### Decoder RNN

- Decoder는 autoregressive RNN으로서 $z$가 주어졌을 때 스케치 데이터($S_1, \cdots, S_{N_{\text{max}}}$)를 생성하는 역할을 한다.
 - 첫 decoder RNN의 히든 입력은 $[h_0; c_0] = \tanh(W_z z + b_z)$이다. ($c_0$은 없을수도 있다.) Q) $c_0$는 뭐지?: cell state of LSTM
 - $i$번째 decoder RNN에는 $i-1$번째 decoder RNN-GMM-Softmax의 출력 $S_{i-1}$와 잠재 벡터 $z$을 병합한 $x_i = [S_{i-1}; z]$를 입력으로 넣는다. 
 - 첫번째 decoder RNN의 입력의 $S_0$는 $(0, 0, 1, 0, 0)$을 사용한다.
 - $i$번째 decoder RNN-GMM-Softmax의 출력 $S_i$는 $i$ 번째 그릴 점에 대한 확률 분포를 의미한다.

- 스케치를 구성하는 선을 $(\Delta x, \Delta y, p_1, p_2, p_3)$와 같이 표현한다.
 - $\Delta x$와 $\Delta y$에 대한 결합 확률 분포를 $M$개의 2변수 정규 분포(bivariate normal distribution)의 mixture로 표현한다.  이는 3번 수식에 해당하며, $M$개의 bivariate normal dist를 표현하기 위해 $5M$개의 파라미터가 필요하며 mixture weight $\Pi$를 표현하기 위해 $M$개의 파라미터가 필요하다.    
 // equation 3
 
 - $(p_1, p_2, p_3)$를 각각 $(q_1, q_2, q_3)$라는 범주형(categorical) 확률 변수로 표현하며 $q_1 + q_2 + q_3 = 1$이라는 제약조건이 있다.
 - 따라서 $(\Delta x, \Delta y, p_1, p_2, p_3)$ 한 쌍을 표현하기 위해서는 $5M + M + 3$개의 파라미터가 필요하다.
 
 

 
 - decoder RNN의 출력 $y_i$는 길이가 $6M + 3$인 벡터인데, 앞부분의 $6M$만큼은 $(\Delta x, \Delta y)$ 분포의 파라미터로 사용되고, 뒷부분 3만큼은 $(q_1, q_2, q_3)$를 표현하는데 사용된다. decoder RNN의 출력은 $\hat{y}$이고 그 형태는 아래와 같다.
 $$\big[ (\hat{\Pi}_1, \mu_x, \mu_y, \hat{\sigma}_x, \hat{\sigma}_y, \hat{\rho}_{xy})_1, \dots (\hat{\Pi}_1, \mu_x, \mu_y, \hat{\sigma}_x, \hat{\sigma}_y, \hat{\rho}_{xy})_M,  ~ (\hat{q}_1, \hat{q}_2, \hat{q}_3) \big] = y_i$$

 - decoder RNN은 $[S_{i-1}; z]$를 입력으로, $[h_{i-1}; c_{i-1}]$을 히든 입력으로 받아 히든 출력 $h_i$와 셀 상태 $c_i$를 만들고, 히든 출력은 다음 decoder RNN으로 보내고, $y_i = W_y h_i + b_y, ~ y_i \in \mathbb{R}^{6M + 3}$는 GMM-Softmax 레이어로 보낸다.

- $y_i$가 GMM-softmax의 입력으로 사용되기 전에 수식 6과 같이 표준편차로 사용되는 파라미터가 0보다 크게 하기 위한 장치가 들어간다. 또한 수식 (7)과 같이 logit $\hat{q}_k$를 확률 값 $q_k$로 변환하는 과정도 포함된다.
 - 또한 수식 (7)과 같이 mixture weight 벡터를 결정하기 위해 $\hat{\Pi}$를 softmax로 변환하여 categorical 확률 분포를 만든다.

###### Hard to train when to stop drawing

- 이 모델을 학습시킬 때 어려운 점은 3가지 펜의 움직임 $p_1, p_2, p_3$에 대한 확률이 매우 불균형하다는 것이다. $p_1$에 대한 확률은 $p_2$에 대한 확률보다 월등히 높으며, $p_3$ 이벤트는 그림을 그리는 동한 단 한번만 발생한다. 이런 상황에 대한 접근법으로 $p_1, p_2, p_3$ 각각의 loss에 서로 다른 가중치를 부여하는 것이다. 예를들어 (1, 10, 100)와 같은 가중치를 사용한다면, 그림을 그리는 동안 한번만 발생하여 전체 loss에 별 영향을 주지 못하는 $p_3$에 대한 loss가 적절히 보정된다.
 - 하지만 지금 다루려는 데이터는 다양한 종류의 이미지를 다뤄야하기 때문에 이 접근접이 적절하지 않다는걸 확인할 수 있었다.

- 대신 위 방법을 조금 변형했는데, 모든 스케치 데이터의 길이를 $N_\max$(training set 각 그림의 길이중 최대값)가 되도록 가공했다. 대부분 스케치 시퀀스의 길이가 $N_\max$보다 작기 때문에, 실제 $S$ 시퀀스 종료부터 $N_\max$까지 (0, 0, 0, 0, 1)로 체워 넣었다.

- 학습 후 스케치를 생성하는 과정은 우선 (첫 시작점을 생성하는 부분은 명확히 나와 있지 않음) 첫 시작점 $S^{'}_1$를 생성하고, 이를 인코더의 입력으로 넣어 GMM-softmax에서 두번째 점 $S^{'}_2$을 얻고, 이를 다시 네트워크의 입력으로 넣는 과정을 $p_3=1$일 때까지 혹은 $i=N_{\text{max}}$일때까지 반복한다.
- 스케치를 샘플링하는건 중간의 VAE 때문에 deterministic하지 않다. 이런 randomness를 파라미터로 조절할 수 있는데, 온도와 같은 역할을 하는 파라미터 $\tau$를 도입해서 이 값이 크면 randomness가 커지도록 할 수 있다. $\tau$는 0과 1 사이의 값을 사용한다. (이 파라미터를 조절했을 때 그림의 변화는 Figure3로 확인할 수 있으며, 파란색은 낮은 온도, 붉은 색은 높은 온도일때 생성된 그림이다.)


### 3.3 Unconditional Generation

- 3.2에서 설명한 구조는 latent vector $z$에 conditional한 출력을 내놓는다. 이 구조에서 앞단의 인코더를 없애고, 디코더 RNN만으로 학습시킬 수도 있는데, 이는 latent 변수가 없는 autoregressive model이라 할 수 있다. 이렇게 바꿨을 때 최초 hidden state과 cell state은 0값들을 사용한다. 그리고 각 time-step에 RNN의 입력은 (training time) $S_{i-1}$과 (inference time) $S^{'}_{i-1}$이다. 이렇게 unconditional하게 생성한 그림이 Figure3에 나온다.

- 이때의 loss function은? $w_{KL}$이 0인 경우에 해당

### 3.4 Training

- VAE에서 maximize하는 loss function은 아래와 같다. 즉 negative KL divergence와 MLE의 합인데, KL divergence를 최소화 하고, MLE(-reconstruction error)를 최대화 한다.(reconstruction error는 최소화)
$$\begin{align}
\mathcal{\tilde{L}}^B(\boldsymbol{\theta}, \boldsymbol{\phi}; \mathbf{x}^{(i)}) &= -  KL\big (q_\phi (\mathbf{z}^{(i, l)} \mid \mathbf{x}^{(i)}) \Vert p_\theta(\mathbf{z}^{(i, l)}) \big) + \frac{1}{L} \sum_{l=1}^L \log p_\boldsymbol{\theta}(\mathbf{x}^{(i)} \mid \mathbf{z}^{(i, l)}) \\
&=\frac{1}{2} \sum_{j=1}^J \bigg( 1 + \log( (\sigma_j^{(i)})^2 - (\mu_j^{(i)})^2) - (\sigma_j^{(i)})^2 \bigg) + \frac{1}{L} \sum_{l=1}^L \log p_\theta (\mathbf{x}^{(i)} \mid \mathbf{z}^{(i, l)} ) \\
~~~ \text{where} ~~ \mathbf{z}^{(i, l)} &= \boldsymbol{\mu}^{(i)} + \boldsymbol{\sigma}^{(l)} ~ \text{and} ~ \epsilon^{(l)} \sim \mathcal{N}(0, \mathbf{I}) 
\end{align}$$

- 이 논문에 최소화 하는 loss function은 VAE 논문에 등장하는 objective function, $\mathcal{\tilde{L}}^B(\boldsymbol{\theta}, \boldsymbol{\phi}; \mathbf{x}^{(i)})$의 부호를 바꾼 것이며          
KL divergence 항인 $K_{KL}$과       
reconstruction loss 항인 $L_R(=L_s + L_p)$을 합한 것이다.         
전체 loss를 최소화하므로 $K_{KL}$와 $L_R(=L_s + L_p)$ 모두 최소화 한다.

- $L_s$는 $\Delta x$와 $\Delta y$에 대한 loss인데, $N_S$이후(스케치가 끝나면) loss는 버린다. 반면 $L_p$는 $(p_1, p_2, p_3)$에대한 loss인데, $N_\max$까지 모든 loss를 사용한다. 이렇게 하면 $(p_1, p_2, p_3)$ 각각에 대한 loss에 weight을 달리하는 것 보다 강건하다.
- $L_{KL}$은 latent vector $z \sim p(z \mid x)$와 $z$의 prior $\mathcal{N}(0, 1)$의 차이를 의미한다.

- 전체 loss는 아래와 같이 $L_{KL}$에 가중치를 줄 수 있다. 즉 regularization 강도를 조절할 수 있다.
$$Loss = L_R + w_{KL} L_{KL}$$
 - $w_{KL}$ 값이 0에 가까워질수록 기본적인 autoencoder에 가까워진다.
 - unconditional generation에는 $L_{KL}$이 없다.
 

- Figure4(좌측): 점선은 standalone 모델로서 $L_{KL}$이 없는 모델이다. 그래서 encoder가 있는 모델 loss의 upper bound 쯤 된다. 

###### Mixture Density Networks
> 신경망 출력을 mixture of gaussian의 파라미터로 이용하는 아이디어로서 이 파라미터는 mixture weight와 각 gaussian의  파라미터(mu, sig, corr)를 말한다. 이런 네트워크는 cross-entropy(=negative log probability)를 최대화하는 방식으로 최적화(MLE) 한다.

###### Implementation of loss function    
 -  아래 loss function 구현과 관련해 'A. Graves. Generating sequences with recurrent neural networks. arXiv:1308.0850, 2013.'의 [4.1 Mixture Density Outputs]를 참고하면 좋다.
 - Mixture density network에 대한 일반적인 이해를 넓히려면 'C. Bishop. Mixture density networks. Technical report, 1994.'를 참고하면 좋다.


```python
def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr,
                 z_pen_logits, x1_data, x2_data, pen_data):
    """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
    # This represents the L_R only (i.e. does not include the KL loss term).

    result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2,
                           z_corr)
    epsilon = 1e-6
    # result1 is the loss wrt pen offset (L_s in equation 9 of
    # https://arxiv.org/pdf/1704.03477.pdf)
    result1 = tf.multiply(result0, z_pi)
    result1 = tf.reduce_sum(result1, 1, keep_dims=True)
    result1 = -tf.log(result1 + epsilon)  # avoid log(0)

    fs = 1.0 - pen_data[:, 2]  # use training data for this
    fs = tf.reshape(fs, [-1, 1])
    # Zero out loss terms beyond N_s, the last actual stroke
    result1 = tf.multiply(result1, fs)

    # result2: loss wrt pen state, (L_p in equation 9)
    result2 = tf.nn.softmax_cross_entropy_with_logits(
        labels=pen_data, logits=z_pen_logits)
    result2 = tf.reshape(result2, [-1, 1])
    if not self.hps.is_training:  # eval mode, mask eos columns
        result2 = tf.multiply(result2, fs)

    result = result1 + result2
    return result

def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850."""
    norm1 = tf.subtract(x1, mu1)
    norm2 = tf.subtract(x2, mu2)
    s1s2 = tf.multiply(s1, s2)
    # eq 25
    z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
         2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
    neg_rho = 1 - tf.square(rho)
    result = tf.exp(tf.div(-z, 2 * neg_rho))
    denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
    result = tf.div(result, denom)
    return result

```

(following the notation of 'A. Graves. Generating sequences with recurrent neural networks')
- args
 - z_pi: $\pi^1, \cdots, \pi^j, \cdots, \pi^M$, mixture weight for kth bivariate normal.
 - z_mu1: $[\mu_1^1, \cdots , \mu_1^j, \cdots, \mu_1^M]$
 - z_mu2: $[\mu_2^1, \cdots , \mu_2^j, \cdots, \mu_2^M]$
 - z_sigma1: $[\sigma_1^1, \cdots , \sigma_1^j, \cdots, \sigma_1^M]$
 - z_sigma2: $[\sigma_2^1, \cdots , \sigma_2^j, \cdots, \sigma_2^M]$
 - z_corr: $[\rho_{12}^1, \cdots , \rho_{12}^j, \cdots, \rho_{12}^M]$
 - z_pen_logits: $[ \hat{q}_1, \hat{q}_2, \hat{q}_3 ]$
 - x1_data: $x_1$
 - x2_data: $x_2$
 - pen_data: $[p_1, p_2, p_3]$

```Python
result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2,
                           z_corr)
```
>  'A. Graves. Generating sequences with recurrent neural networks'의 수식 (24)가 $M$개 있는... $ \big\{  \mathcal{N}(x_{t+1} \mid \mu_t^j, \sigma_t^j, \rho_t^j) \big\}_{j=1}^M $에 해당하며, $P(\text{data} \mid \text{hypothesis})$들이라 할 수 있다. 즉 네트워크가 학습한 gaussian에 대한 파라미터를 사용했을 때, 학습 데이터가 관측될 확률을 계산한다. (그런데 gaussian을 mix하는건 다음 단계에 등장한다.)

```Python
result1 = tf.multiply(result0, z_pi)
result1 = tf.reduce_sum(result1, 1, keep_dims=True)
```
> mixing. $$\text{Pr}(x_{t+1} \mid y_t) = \sum_{j=1}^M \pi_t^j \mathcal{N}(x_{t+1} \mid \mu_t^j, \sigma_t^j, \rho_t^j)$$

```Python
fs = 1.0 - pen_data[:, 2]  # use training data for this
fs = tf.reshape(fs, [-1, 1])
```
> 스케치 데이터의 실제 길이($N_s$)까지는 1이고 나머지 $N_{\text{max}}$까지는 0인 벡터(fs) 생성, 

```Python
# Zero out loss terms beyond N_s, the last actual stroke
result1 = tf.multiply(result1, fs)
```
> $N_s$ 이후 $L_s$ 값을 0으로 만듬.

```Python
# result2: loss wrt pen state, (L_p in equation 9)
result2 = tf.nn.softmax_cross_entropy_with_logits(
    labels=pen_data, logits=z_pen_logits)
result2 = tf.reshape(result2, [-1, 1])
```
> $q_k = \frac{\exp(\hat{q}_k)}{\sum_{j=1}^3\exp(\hat{q}_j)} ~ , k \in {1, 2, 3}$ 이므로 tf.nn.softmax_cross_entropy_with_logits를 이용해 $q_k$와 label의 cross-entropy를 구한다.
- cross-entropy: $$\begin{bmatrix}
-\big( p_1 \log q_1 + (1-p_1) \log(1-q_1) \big) \\
-\big( p_2 \log q_2 + (1-p_2) \log(1-q_2) \big) \\
-\big( p_3 \log q_3 + (1-p_3) \log(1-q_3) \big)
\end{bmatrix}$$

```Python
if not self.hps.is_training:  # eval mode, mask eos columns
    result2 = tf.multiply(result2, fs)
```
> inference할 때에는 $N_s$이후 $L_p$를 버린다. 반면 training할 때에는 $N_s$ 이후 $L_p$를 모두 사용한다. 즉 스케치가 끝나는 이벤트에 대한 loss를 크게 보이게 하는 것이다.

```Python
result = result1 + result2
return result
```
> $$L_R = L_s + L_p$$

# 4. Experiments

- sketch-rnn 모델에서는 RNN cell을 빌딩 블록으로 사용한다.
 - encoder에는 LSTM을
 - decoder에는 HyperLSTM을 사용

- 데이터는 cat, pig, face, firetruck, garden, owl, mosquito, yoga 등 특정 대상을 그린 그림들인데, cat과 pig를 섞어서 학습하는 것도 테스트 했다.

- $w_{KL}$를 조절하면서 $L_R$와 $L_{KL}$이 어떻게 달라지는지 관찰했다.
 - 예상대로 $w_{KL}$이 커지면 (regularization이 약해지므로) $L_{KL}$이 조금 증가하고 $L_R$이 비교적 크게 감소하는 경향이 있었다.
- standalone decoder 모델의 loss는 conditional model보다 항상 컸다.

### 4.1 Conditional Reconstruction

- conditional model에 스케치 데이터 $S$를 입력했을 때 다시 생성하내는 스케치 $S'$를 관찰했다. 이때  temperature $\tau$를 조절하면서 그 변화를 살펴봤다. 
 - (예상대로) temperature를 증가시킬수록 입력 이미지와 다른 이미지가 생성된다.

### 4.2 Latent Space Interpolation

- (고양이와 돼지로 학습된 하나의 모델에) 고양이 그림을 입력했을 때 latent space의 점과 돼지 그림을 입력했을 때 latent space의 점 사이 점들에 대한 출력 그림을 살펴본다. 그런데 $w_KL$이 클수록 실제 두 입력 이미지에 가까운 출력을 내놓는다.

### 4.3 Sketch Drawing Analogies

- 앞서 latent space interpolation이 유의미한 그림을 생성하는걸 보면, latent vector $z$에 그림의 특징이 함축적으로 담긴다는 알 수 있다. 그렇다면 latent vector만을 이용해 고양이 얼굴에 몸을 추가하는 식의 그림 생성(유추)도 가능하지 않을까? 이런 실험을 진행했는데, $L_{KL}$이 작은 모형에서 이런게 가능했고, 벡터 연산을 통해 이미지 특징간의 연산이 가능함을 확인했다. (Figure6 오른쪽에 고양이 얼굴에 돼지몸을 붙인 그림 생성)

### 4.4 Predicting Different Endings of Incomplete Sketches

- 미완성인 그림을 완성하는 걸 실험해 봤다. 모델은 (encoder 없는) standalone 모델을 사용했다.
 - 우선 미완성 그림을 decoder RNN에 입력으로 넣어 hidden state $h$를 우선 인코딩한 후, 이 hidden state을 (이전에 cat, pig 등 특정 그림으로 학습한) standalone 모형의 초기 상태로 넣어 나온 출력값으로 그림을 완성한다.

# 5. Applications and Future Work

- 이런걸 예술가의 창조 활동을 돕는데 사용할 수 있을 것 같다. 예를들어 그림의 시작 부분만 그리고 그 이후의 다양한 전개를 생성해본다거나(Ending incomplete sketches), 두개의 다른 그림을 입력하고 이 둘 사이에 존재할 수 있는 그림들을 생성해본다거나(Latent space interpoliation), 서로 다른 두개의 그림의 논리적 관계를 이용해 새로운 그림을 생성한다거나(drawing analogies), 어떤 패턴을 그리고 이와 유사하지만 다른 많은 패턴을 생성한다거나(Conditional reconstruction)하는 등으로 활용해 볼 수 있다.

- 돼지를 닮은 트럭, 고양이를 닮은 의자를 디자인하는 예, 의자와 고양이의 latent space를 interpolate해서 가능한 다양한 아이디어를 탐색(figure8).

- 그림을 연습하는 도구로도 사용 가능
- 이미지를 넣으면 스케치를 만들어주는 것도 가능할 것 같음.(future work)

# 6. Conclusion

- RNN으로 스케치 데이터를 학습하는 모형을 만들어봄.
- 미완성 그림을 완성하거나, latent space를 탐색해서 새로운 그림을 생성하거나, 입력된 그림과 비슷한 그림을 생성하거나, 그림을 속성을 조작하는 등의 시도를 해 봄
- prior의 강도를 $w_{KL}$로 조절할 수 있는데, 이 값에 따라 interpolated image의 퀄리티가 달라지더라.
- 사용된 데이터 세트를 공개함.

# Supplementary Materials

### 1. Dataset Details

- 스케치 라인을 단순화 하기 위해서 Ramer-Douglas-Peucker 알고리즘이 사용되었고, 이때 파라미터 $\epsilon$은 0.2를 사용했다.

### 2. Training Details

- $L_{KL}$항을 annealing 하는 것이 효과적이었다. 무슨 말이냐면.. 학습 초반에는 $L_{KL}$을 줄이는 것 보다 reconstruction error를 줄이는 것이 훨신 어렵기 때문에, 학습 초반에는 $L_R$부터 줄이도록 강제하기 위해 $L_{KL}$에 작은 weight을 주는 것이다. 이를 위해 $L_{KL}$에 아래 항을 곱한다.
$$\eta_{step} = 1 - (1 - \eta_{min})R^{step}$$
$\eta_{min}$은 0이나 0.1이고, $R$은 1보다 작고 1에 가까운 수를 사용하므로 $step$이 증가할 수록 $\eta_{step}$값은 0에서 시작해서 1에 가까워지게 된다. 즉 초반에는 $L_{KL}$을 거의 줄이지 않는 것이다.

- $z$가 $\mathcal{N}(0, 1)$에 가까워지면 단지 $\mathcal{N}(0, 1)$에서 샘플을 뽑아서 스케치를 생성해볼 수 있게 된다. 이런 스케치 생성 실험을 $L_{KL}$값 이 다른 여러 모형으로 테스트 해 봤는데, $L_{KL}$이 작아질수록 명확한 스케치를 얻을 수 있었다. 하지만 $L_{KL}$이 0.3아래로 내려가면 스케치의 품질에 별 차이가 없었다. 그 말은 모형 학습시 $L_{KL}$에 대한 최소값을 지정하면 일정값 이하로 $L_{KL}$를 줄일 필요가 없어지고 대신 $L_{R}$에 집중할 수 있게 된다. $L_{KL}$에 대한 최소값 $KL_{min}$은 대략 0.1에서 0.5 사이의 값을 사용하며 'D. Ha. Recurrent Net Dreams Up Fake Chinese Characters in Vector Format with TensorFlow, 2015.'에서도 비슷한 기법을 사용한다.

### 3. Model Configuration

- 인코더에는 RNN 512개, 디코더에는 RNN 2048개를 사용
- $M=20$
- $N_z=127$
- Layer Normalization을 사용함
- 학습시 recurrent dropout을 사용함(keep prob = 90%)
- batch size = 100
- adam optimizer (learning rate = 0.0001)
- gradient clipping of 0.1
- $KL_{min}=0.2$
- R = 0.99999
- data augmentation: $\Delta x$와 $\Delta y$에 랜덤한 값(0.9 1.1 uniform)을 곱해서 데이터 생성함.
- 특별한 경우가 아니면 $w_{KL}$ = 1.0

### 4. Model Limitations

- single-class dataset(cat, pig 등 한가지 종류 스케치로 구성된 데이터)에는 대략 300개 샘플까지는 잘 학습할 수 있었다. 하지만 그 이상이 되면 학습하기가 어려워졌다. 이런 이유로 Ramer-Douglas-Peucker 알고리즘으로 스트록을 단순화해서 대략 200개 정도가 되게 만들었다. 물론 최대한 그림의 특징이 훼손되지 않도록 노력했다. 

- 인어나 랍스터 같은 복잡한 그림을 학습시켰을 때 $L_R$이 (다른 단순한 그림을 학습시킬 때에 비해) 상대적으로 많이 줄지 않았다. 이런 복잡한 그림을 학습시킨 후 그림을 생성하면 실제 그림보다 부드럽고 둥글둥글한 그림을 생성하는 경향이 있었다. 이런 현상은 VAE가 입력 데이터에 노이즈를 섞는 효과를 갖기 때문에 발생하는 것 같은데, 상황에 따라서는 이 효과가 꼭 나쁘다기보다는 좀 더 심미적인 그림이라고도 볼 수 있다.

- sketch-RNN이 여러 클레스의 그림(cat+pig, crab+face+pig+rabbit)을 동시에 학습하는데는 그다지 효과적이지는 않았다. 75개 클레스를 동시에 학습하는걸 테스트 해 봤는데 Figure4 좌측과 같이 엉터리였다. 반면 4개 정도 클레스를 학습했을 때는 Figure4 오른쪽과 같이 썩 나쁘지는 않았다.

### 5. Multi-Sketch Drawing Interpolation

- 두개의 이미지를 $z$에 투영하고 interpolation하는걸 앞서 4.2에서 봤었는데, 4개의 입력 이미지를 interpolation하는것도 실험해봤다. 

### 6. Which Loss Controls Image Coherency?

- 퀄이 좋은 그림을 생성하는데 있어서 $L_R$과 $L_{KL}$중 어느 것이 더 중요할까? 얼핏 생각하기에는 $L_R$이 입력과 출력 사이의 차이를 줄이려는 reconstruction error이므로 입력과 같은 이미지를 생성하는데 중요할 것 같지만, $L_R$만 작은 모델이 퀄 좋은 그림을 생성하지는 않았다.

- 예를들어 smile face 그림을 학습시켰을 때, $L_R$만 작은 모델은 얼굴의 외곽 동그라미가 입력과 가까운 경우 $L_R$이 매우 작아지지만 그림의 중요한 특징이라 할 수 있는 눈과 웃는 입은 잘 표현하지 못했다. 

- 다양한 $w_{KL}$ 값에 생성 이미지가 Figure9에 나와 있는데, $w_{KL}$이 1인 경우 돼지의 코를 잘 표현하지 못했고, 가제의 다리를 잘 표현하지 못했다. 반면 $w_{KL}$이 0.25인 경우 형체가 많이 어그러진 그림이 나왔다. 즉 $w_{KL}$을 잘 조절해서 $L_R$과 $L_{KL}$ 사이의 trade-off가 필요하다.

- interpolation 결과를 보면 $L_{KL}$이 작을 수록 interpolation으로 생성한 그림이 보다 (보기에) 유의미(more coherent) 했고, 이는 $L_{KL}$이 작을 수록 $z$ 벡터가 보다 유의미한 정보를 함축한다는 것을 의미한다.

### 참고자료

- paper: https://arxiv.org/abs/1704.03477
- drawing test: https://magenta.tensorflow.org/assets/sketch_rnn_demo/index.html
- blogs: 
 - https://magenta.tensorflow.org/sketch-rnn-demo
 - https://ai.googleblog.com/2017/04/teaching-machines-to-draw.html
 - https://magenta.tensorflow.org/sketch_rnn
 - https://experiments.withgoogle.com/sketch-rnn-demo
- github repo: https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/README.md
