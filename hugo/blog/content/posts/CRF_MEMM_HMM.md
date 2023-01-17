---
title: "CRF_MEMM_HMM"
date: 2022-10-08T11:55:22+08:00
#draft: true
---

# 预备知识

在介绍HMM、MEMM、CRF之前先介绍一些前置的知识点

## Generative vs Discriminative models

生成模型 vs 判别式模型，首先需要理解这两个模型的区别，对于后续模型介绍理解会比较重要。同时，介绍这些模型的时候，也会涉及遇到一些简单的概率知识点，核心是贝叶斯公式。我们先给出文中会使用的一些符号定义。

$$
x \text{:  输入数，可以是多维数据;} \quad y \text{:  输出数据，也可以是多维数据}

$$

> ***Generative model***：核心是需要对$p(x \mid y)$ 建模，或者是对 $p(x)$建模，最终的目的是实现 $p(x,y)$的建模
> 
> ***Discriminative model***：通常是直接建模$p(y \mid x)$，不需要考虑$p(x)$

看上面的定义，有时候在实际应用中可能会存在困惑，因为二者的区别不是很明显。例如对一个生成模型$p_g(\theta)$

$$
p_g(y,x;\theta) = p_g(y;\theta) p_g(x \mid y;\theta) \\
p_g(y,x;\theta) = p_g(x;\theta) p_g(y \mid x;\theta)
$$

生成模型$p_g(y,x;\theta)$使用上面的形式的时候，看上去像生成模型，但是采用下面公式（通常是在inference的时候采用）看上去像判别式模型。其实核心还是在于我们在用判别式的时候并不考虑$p(x;\theta)$（下文中会介绍）。真正在推理的时候，我们正确使用方式是$p_g(x;\theta) = \sum_y p_g(y,x;\theta)$， $p_g(y \mid x;\theta) = {p_g(y,x;\theta) \over p_g(x;\theta)} $，也就是说虽然生成式模型可以应用于推断$p(y \mid x)$，但是底层使用的模型还是生成式模型$p_g(y,x;\theta)$

对于判别式模型，我们直接建模求解：$p(y \mid x; \hat\theta)$，而$p(x;\theta)$我们依然是未知的。

大部分情况下，我们可以断定$\theta \ne \hat\theta$。为什么很多情况下 判别式模型表现由于生成式？从上面的公式我们可以看出，生成式模型要同时满足 $p(x;\theta)$和$p(y \mid x;\theta)$的分布，而判别式模型只考虑$p(y \mid x; \theta)$，二者恰恰是真实应用场景里关注的问题所在。

> ***Generative model parameters $\theta$ are used in both the input distribution and conditional distribution, a good set of parameters must represent both well,  potentionally at the cost of tradding off accuracy on $p(y \mid x)$, the distribution we care about, for accuracy on $p(x)$, which we care less obout. On the other hand, the added degree of freedom brings about an increasing risk of overfitting the training data, and generalizing worse on unseen data.***

## Hammersley-Clifford定理证明

### 马尔可夫随机场（MRF）

> **定义一：一个无向图模型$G$称之为马尔可夫随机场（MRF），如果满足两个顶点被观测顶点分割情况下独立。也就是说对图中任意顶点$X_i$，以下条件属性成立**
> 
> $$
> \begin{equation}
P(X_i|X_{G \backslash i}) = P(X_i|X_{N_i}) 
\end{equation}
> $$
> 
> **$X_{G \backslash i}$代表除了$X_i$之外的所有顶点，$X_{N_i}$代表$i$的所有邻居顶点，即所有与$X_i$相连的顶点。**
> 
> An undirected graphical model $G$  is called a *Markov Random Field (MRF)* if two nodes are conditionally independent whenever they are separated by evidence nodes. In other words, for any node $X_i$ in the graph, the equation (1) holds, where $X_{G \backslash i}$ denotes all the nodes except $X_i$, and $X_{N_i}$ denotes the neighborhood of $X_i$ - all the nodes that are connected to $X_i$.

> 

上面两个定义直接看公式十分生涩，下面给两个例子来说明。

```mermaid
flowchart LR
x1((x1))
x2((x2))
x3((x3))
x4((x4))
x5((x5))

x1 --> x2 --> x3 --> x5

x2 --> x4 --> x5
```

上面这张图是有向图，贝叶斯网络/HMM/卡尔曼滤波器都属于这类有向图网络。上面图的联合概率：

$P(x_1,\dots,x_n) = P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_2) \cdot P(x_4 \mid x_2) \cdot P(x_5 \mid x_3,x_4)$

上面公式的分解很容易理解，不展开介绍。下面重点要介绍无向图概率模型（MRF）

```mermaid
flowchart LR
x1((x1))
x2((x2))
x3((x3))
x4((x4))
x1 --- x3 --- x2
x1 --- x4 --- x2
```

#### 全局MRF

<img title="" src="https://upload-images.jianshu.io/upload_images/11248965-f89b11e7430eff95.png?imageMogr2/auto-orient/strip|imageView2/2/w/756/format/webp" alt="" data-align="center" width="476">

假设节点集合A、B是在无向图中被节点集合C分开的任意节点集合，全局马尔可夫定义 ：给定基于节点集合C的随机变量$Y_C$的条件性，$Y_A和Y_B条件独立$。

$$
P(Y_A,Y_B \mid Y_C) = P(Y_A \mid Y_C) P(Y_B \mid Y_C)
$$

从全局马尔可夫可以推导出局部马尔可夫和成对马尔可夫

#### 局部马尔可夫（Markov Banket）

又叫做马尔可夫毯，也就是前面给出的定义，一个无向图中一个节点给定它的邻居节点的情况下，独立于其它节点。下面给出一个具体例子：

<img title="" src="https://upload-images.jianshu.io/upload_images/11248965-632725354eeb4c99.png?imageMogr2/auto-orient/strip|imageView2/2/w/910/format/webp" alt="" data-align="center" width="383">

设无向图中任意一节点$v$, $\mathcal{W}$是与节点$v$有边相连的所有节点，$\mathcal{O}$是$u, \mathcal{W}$之外的其它节点集合，局部马尔可夫性是指：给定$Y_{\mathcal W}$的情况下，$Y_v$与$Y_{\mathcal O}$独立。$P(Y_v,Y_{\mathcal O}) \mid Y_{\mathcal W}) = P(Y_v \mid Y_{\mathcal W}) P(Y_{\mathcal O} \mid Y_{\mathcal W})$

这个公式也可以改写为：$P(Y_v \mid Y_{\mathcal O}, Y_{\mathcal W}) = P(Y_v \mid P_{\mathcal W})$

#### 成对马尔可夫

假设无向图$G$中任意两个没有边连接的节点$u,v$，其它所有节点为$\mathcal O$，成对马尔可夫：给定$Y_{\mathcal O}$条件下，$Y_u, Y_v$条件独立。$P(Y_u, Y_v \mid Y_{\mathcal O}) = P(Y_u \mid Y_{\mathcal O}) P(Y_v \mid Y_{\mathcal O})$

上面介绍了成对、局部和全局马尔可夫性，使用大白话来说就是每一个节点的分布只和有边相连的节点有关系。**其实最大团的概念就是从成对马尔可夫衍生过来，只需要把上面的节点理解为节点集合**。

### Gibbs分布

> 定义二：在无向图模型G上的一个概率分布P(X)称之为吉布斯分布，如果它能够被因子分解为定义在**团（clique）** 上的正函数的乘积，这些团覆盖了G的所有顶点和边，即
> 
> $$
> \begin{equation}
\begin{aligned}
P(X) &= \frac{1}{Z} \prod\limits_{c \in C_G} \phi_c(X_c) \\\\
Z &= \sum\limits_x \prod\limits_{c \in C_G} \phi_c(X_c)
\end{aligned}
\end{equation}
> $$
> 
> C_G是G上所有（最大）团的集合，c代表一个具体的`clique`（实际上是一个小的集合），Z是归一化常量，保证满足概率定义。
> 
> A probability distribution P(X) on an undirected graphical model G is called a Gibbs distribution if it can be factorized into positive functions defined on cliques that cover all the nodes and edges of G. That is equation (2), where $G_C$ is a set of all (maximal) cliques in G and $Z = \sum_x \prod_{c \in C_G} \phi_c(X_c) $is the normalization constant.
> 
> Hammersley Clifford clas 认为，MRF和Gibbs分布是一致的，也就是说：
> 
> 1. Gibbs分布一定满足由`node separation`导致的条件独立性。
> 
> 2. MRF的概率分布一定可以表示为最大团集合上的非负函数乘积形式。
> 
> We formally define the notion of whether a factor graph "describes" a given distribution or not. Let $N(a)$ be the neighbors of the factor with index $a$, i.e., a set of variable indices. Then:
> 
> $$
> \text{Definition 2.1. A distribution } p(y) 
\text{ factorizes according to 
 a factor graph } \\
 G \text{ if there exists a set of local function } \Psi_a \text{ such that } p \text{ can be written as} \\
p(y) = Z^{-1} \prod\limits_{a \in F} \Psi_a(y_{N(a)})
> $$
> 
> 下面的无向图的概率计算：$p(y_1,y_2,y_3) = \Psi_1(y_1,y_2) \Psi_2(y_2,y_3) \Psi_3(y_1,y_3)$，图中的f1/f2/f3对应$\Psi_1/\Psi_2/\Psi_3$
> 
> **最大团**：若C是无向图G的一个团（即C中任何两个节点存在G中的边相连），并且不能再加入跟着一个G的节点使其成为一个更大的团，则称此C为G的一个最大团

```mermaid
graph LR
y1((y1))
y2((y2))
y3((y3))
y1 ---|f1| y2 ---|f2| y3 ---|f3| y1
```

马尔科夫网络(markov network) vs 因子分解图(factor graph)。从无向图角度分析，下面三个图是一个图，都是三个节点之间两两连接。如果从马尔科夫网络分析，它的概率分解其实是和因子分解图2等价的，即$p(y_1,y_2,y_3) \sim f(y_1,y_2,y_3)$，但是因子分解图2 $p(y_1,y_2,y_3) \sim f(y_1,y_2) g(y_2,y_3) h(y_1,y_3)$。所以从因子分解图角度看，马尔科夫网络的概率计算是有歧义的。

```mermaid
graph TD
subgraph 因子分解图2
    f4(( ))
    f5(( ))
    f6(( ))
    d[ ]
    f4 --- d
    d --- f5
    d --- f6

end

subgraph 因子分解图1
    f1(( ))
    f2(( ))
    f3(( ))
    f1 ---| f | f2
    f2 ---| f | f3
    f3 ---| f | f1
end


subgraph 马尔科夫网络
    n1((  ))
    n2((  ))
    n3((  ))
    n1 --- n2
    n2 --- n3
    n3 --- n1
end
```

Hammersley Clifford证明了上面两个定义是等价的，下面搬运证明过程

## 反向证明（吉布斯分布—> MRF）

设$D_i = N_i \cup {X_i}$是包含$X_i$邻居顶点和$X_i$本身的集合。从上面等式(1)的右边开始。另外我们引入一些符号定义：$D_i = N_i \cup \{X_i\}$是包含$X_i$和它邻居定点的集合，无向图$G$的最大团为$G_C$，并且通过判断定点$X_i$是否包含在`clique`可以将集合$G_C$分成两个集合：$C_i=c \in G_C: X_i \in c$和$R_i = c \in G_C: X_i \notin G_C$，且$C_i \cup R_i = G_C$。下面是证明过程：

$$
\begin{equation}
\begin{aligned}
P(X_i \mid X_{N_i}) &= \frac{P(X_i,X_{N_i})}{P(X_{N_i})} \\\\
&= \frac{\sum\limits_{G\backslash D_i} P(X_i,X_{N_i},X_{G\backslash D_i})}
   {\sum\limits_{x_i}\sum\limits_{G\backslash D_i}P(X_i,X_{N_i},X_{G\backslash D_i}) }\\\\
&= \frac{\sum\limits_{G\backslash D_i} \prod\limits_{c \in G_C} \phi_c(X_c)} 
    {\sum\limits_{x_i}\sum\limits_{G\backslash D_i} \prod\limits_{c \in G_C} \phi_c(X_c)} \\\\
&= \frac{\sum\limits_{G\backslash D_i} \prod\limits_{c \in C_i}\phi_c(X_c) \prod\limits_{c \in R_i}\phi_c(X_c)}
  {\sum\limits_{x_i}\sum\limits_{G\backslash D_i} \prod\limits_{c \in C_i}\phi_c(X_c) \prod\limits_{c \in R_i}\phi_c(X_c)} \\\\
&= \frac{\prod\limits_{c \in C_i}\phi_c(X_c) \left( \sum\limits_{G\backslash D_i} \prod\limits_{c \in R_i}\phi_c(X_c)  \right)}
  {\left(\sum\limits_{x_i} \prod\limits_{c \in C_i}\phi_c(X_c) \right)  \left(\sum\limits_{G\backslash D_i} \prod\limits_{c \in R_i}\phi_c(X_c) \right) }\\\\
&= \frac{\prod\limits_{c \in C_i}\phi_c(X_c)}{\sum\limits_{x_i} \prod\limits_{c \in C_i}\phi_c(X_c)} \times 
  \frac{\prod\limits_{c \in R_i}\phi_c(X_c)}{\prod\limits_{c \in R_i}\phi_c(X_c)} \\\\
&= \frac{{1 \over Z} \prod\limits_{c \in G_C} \phi_c(X_c)}
  {{1 \over Z} \sum\limits_{x_i}\prod\limits_{c \in G_C} \phi_c(X_c)} \\\\
&= \frac{P(X)}{P(X_{G\backslash i})} = P(X_i \mid X_{G\backslash i})
\end{aligned}
\end{equation}

$$

最终公式(3)的结论和公式(1)相同。上面的证明是严格推导，下面给出一个比较有体感的特殊例子证明。我们这次从前面的全局马尔可夫来证明下图中公式成立。

从上面的全局马尔可夫图可以得出：$P(x_A,x_B,x_c) = {1 \over Z} \phi(x_A, x_c) \phi(x_B,x_C)$，我们要证明$P(x_A,x_B \mid x_C) = P(x_A \mid x_C) P(x_B \mid x_C)$，下面给出证明：

$$
\begin{equation}
\begin{aligned}
P(x_A,x_B \mid x_C) &= \frac{P(x_A,x_B,x_C)}{P(x_C)} \\\\
&= \frac{P(x_A,x_B,x_C)}{\sum_{x_A}\sum_{x_B}P(x_A,x_B,x_C)} \\\\
&= \frac{\phi_{AC}(x_A,x_C) \phi_{BC}(x_B,x_C)}{\sum_{x_A}\sum_{x_B} \phi_{AC}(x_A,x_C) \phi_{BC}(x_B,x_C)} \\\\
&= \frac{\phi_{AC}(x_A,x_C)}{\sum_{x_A}\phi_{AC}(x_A,x_C)} \times
   \frac{\phi_{BC}(x_B,x_C)}{\sum_{x_B}\phi_{BC}(x_B,x_C)} \\\\
&= \frac{P(x_A,x_C)}{P(x_C)} \times \frac{P(x_B,x_C)}{P(x_C)} \\\\
&= P(x_A \mid x_C) \cdot P(x_B \mid x_C)
\end{aligned}
\end{equation}
$$

## 正向证明（MRF—>吉布斯分布）

没看明白，待续

# CRF

> **条件随机场定义**：设$X$与$Y$是随机变量，$P(Y \mid X)$是在给定$X$的条件下$Y$的条件概率分布。若随机变量$Y$构成一个由无向图$G=(V, E)$表示的**马尔可夫随机场**，即：
> 
> $P(Y_v \mid X, Y_w; w \ne v) = P(Y_v \mid X, Y_w; w \sim v)$
> 
> 对任意顶点$v$成立，则称**条件概率分布**$P(Y \mid X)$ 为**随机条件场**。 
> 
> 其中：
> 
>   $w \sim v$表示在图$G=(V,E)$ 与定点$v$有边连接的所有定点$w$;
> 
>   $w \ne v$ 表示定点$v$ 以外的所有定点；
> 
> $Y_v,Y_w$为定点$v$与$w$对应的随机变量

常见的条件随机场图结构如下图：

![](https://raw.githubusercontent.com/huangpeng1126/huangpeng1126.github.io/master/images/crf_hmm.png)

## 链式随机条件场

在实际应用中，一般设定$X$和$Y$有相同的图结构，以线性链条件随机场应用最为广泛。

其中有两种主要的线性条件随机场的图结构如下：

<img title="" src="https://pic2.zhimg.com/v2-3e284501e3cae972ec8d3549e3820fa1_r.jpg" alt="" data-align="center" width="451">

注意上图左边那个图，最大团是相邻三个顶点形成的团，而右边的图最大团是相邻两个顶点形成的团。

> **线性条件随机场定义**：设$X=(X_1,X_2,\dots,X_n), Y=(Y_1,Y_2,\cdots,Y_n)$均为线性表示的随机变量序列，若在给定随机变量序列$X$的条件下，随机变量序列$Y$的条件概率分布$P(Y \mid X)$构成条件随机场，即满足马尔科夫性：
> 
> $P(Y_i \mid X, Y_1,Y_2,\cdots,Y_{i-1},Y_{i+1},\cdots,Y_n)=P(Y_i \mid X, Y_{i-1})$
> 则称$P(Y \mid X)$为线性条件随机场。

下图是线性随机场的图结构：

<img title="" src="https://pic1.zhimg.com/80/v2-cebd08fab5f567dff1ca16e2d204984c_1440w.webp" alt="" data-align="center" width="495">

## 线性条件随机场公式

> **Linear-chaiin CRFs Definition**: Let $X, Y$ be random vectors, $\theta = \{\theta_k\} \in \mathcal{R}^K$ be a parameter vector, and $\mathcal{F}=\{f_k(y, y, x_t)\}$ be a set of real-valued functions, Then *a linear-chain conditional random field* is a distribution $p(y \mid x)$ that takes the form:
> 
> $$
> \begin{equation}
\begin{aligned}
p(y \mid x) &= {1 \over Z(x)} \prod_{t=1}^T exp \left[ \sum_{k=1}^K
\theta_k f_k(y_t,y_{t-1},x_t)  \right] \\\\
&= \frac{exp\left(\sum\limits_{t=1}^T\sum\limits_{k=1}^K \theta_k f_k(y_t,y_{t-1},x_t) \right)}{Z(x)}   \\\\
Z(x) &= \sum_y exp\left[\sum\limits_{t=1}^T\sum\limits_{k=1}^K \theta_k f_k(y_t,y_{t-1},x_t)
\theta_kf_k(y_t,y_{t-1},x_t) \right]
\end{aligned}
\end{equation}
> $$
> 
> where $Z(x)$ is an input-depedent normalizatioin function.

上面的公式定义中使用的是`log-linear`形式的函数，在通用CRFs中使用的是通用函数定义：$\Psi_t(y_t,y_{t-1},x_t \rightarrow \theta_k f_k(y_t,y_{t-1},x_t)$

CRF的具体例子和参数估计方式，可以参考[知乎CRF文档](https://zhuanlan.zhihu.com/p/483820319)

## CRF在词性标注的应用

目标函数正如公式(5)所定义，而我们的目标是最大化上面的线性条件随机场示意图中的隐状态序列的得分（即$p(y\mid x)$，因此我们需要计算每个隐状态序列$y$对应的得分。根据公式定义，我们在应用CRF做词性标注的时候，需要首先完成特征函数$f_k$的设计，然后通过大量数据来学习参数集合$\theta_k$。接下来我们将介绍如何确定参数$\theta_k$。

通过公式(5)我们已经知道了条件概率的函数定义，可以使用MLE方法来求解最优参数。

$$
\begin{equation}
\begin{aligned}
\mathcal{L}(\theta) &= \prod_{i=1}^N p(y_i \mid x_i) \\\\
&= \prod_{i=1}^N \frac{exp\left(\sum\limits_{t=1}^T\sum\limits_{k=1}^K \theta_k f_k(y_t,y_{t-1},x_t) \right)}{Z(x)} \\\\
ln(\mathcal{L}(\theta)) &= \sum_{i=1}^N\left(\sum\limits_{t=1}^T\sum\limits_{k=1}^K \theta_k f_k(y_t,y_{t-1},x_t)-ln(Z(x)\right)

\end{aligned}
\end{equation}
$$

继续，通过计算偏导数求极值（如果使用Pytorch或TensorFlow等支持自动求导的框架，直接使用$\mathcal{Loss}$函数即可）：

$$
{\partial (ln(\mathcal{L}(\theta))) \over {\partial \theta_k}}
 = \sum_{i=1}^N\left(\sum_{t=1}^T f_k(y_t,y_{t-1},x_t) - {1 \over Z(x)} {\partial Z(x) \over \partial \theta_k} \right)
$$

对$\partial Z(x) \over \partial \theta_k$单独求导：

$$
\begin{aligned}
{\partial Z(x) \over \partial \theta_k} &= \sum_y exp \left( \sum\limits_{t=1}^T\sum\limits_{k=1}^K \theta_k f_k(y_t,y_{t-1},x_t)
\theta_kf_k(y_t,y_{t-1},x_t) \right) * \sum_{t=1}^T f_k(y_t,y_{t-1},x_t)  \\\\
{1 \over Z(x)} {\partial Z(x) \over \partial \theta_k} &= \sum_y \left[
\frac{exp \left( \sum\limits_{t=1}^T\sum\limits_{k=1}^K \theta_k f_k(y_t,y_{t-1},x_t)
\theta_kf_k(y_t,y_{t-1},x_t)\right)}{Z(x)}*\sum_{t=1}^T f_k(y_t,y_{t-1},x_t)\right]\\\\
&= \sum_y \left[ p(y\mid x) * \sum_{t=1}^T f_k(y_t,y_{t-1},x_t) \right]
\end{aligned}
$$

所以，最终对似然函数的求导结果为

$$
{\partial (ln(\mathcal{L}(\theta))) \over {\partial \theta_k}}
= \sum_{i=1}^N\left[\sum_{t=1}^T f_k(y_t,y_{t-1},x_t) - \sum_y \left( p(y\mid x) * \sum_{t=1}^T f_k(y_t,y_{t-1},x_t)\right) \right]
$$

上面公式中，$\sum_{t=1}^T f_k(y_t,y_{t-1},x_t)$ 只需要遍历一遍样本就可以求出，但是后半部分求解十分复杂，因为假设$x_{1:n}, y_{1:m}$，那么$y$的求解空间是$m^n$。这个概率求解我们可以通过HMM里的`向前-向后算法求解`。
