---
title: "Xgboost_lightgbm_catboost分析"
date: 2022-06-28T11:33:19+08:00
draft: true
---

## xgboost

### 模型定义

> 什么是XGBoost：[官方文档]([Introduction to Boosted Trees &mdash; xgboost 2.0.0-dev documentation](https://xgboost.readthedocs.io/en/latest/tutorials/model.html#))
> 
> XGBoost 是基于决策树的集成机器学习算法，它以梯度提升（Gradient Boost）为框架。在非结构数据（图像、文本等）的预测问题中，人工神经网络的表现要优于其他算法或框架。但在处理中小型结构数据或表格数据时，现在普遍认为基于决策树的算法是最好的。下图列出了近年来基于树的算法的演变过程：
> 
> ![](https://pic2.zhimg.com/80/v2-c108a582cd03dae298c5c51305498cf5_1440w.jpg)
> 
> XGBoost 算法最初是华盛顿大学的一个研究项目。[陈天奇](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3MzI4MjgzMw%3D%3D%26mid%3D2650760514%26idx%3D2%26sn%3Df9e03c2e4aead6098e30111493f49d28%26chksm%3D871aa13cb06d282a227d056740803c69c4c978d5bdb87fc1b30f1d1ed57c0c01cf8ec99657dd%26token%3D1906738629%26lang%3Dzh_CN)和 Carlos Guestrin 在 SIGKDD 2016 大会上发表的论文《XGBoost: A Scalable Tree Boosting System》在整个机器学习领域引起轰动。自发表以来，该算法不仅多次赢得 Kaggle 竞赛，还应用在多个前沿工业应用中，并推动其发展。许多数据科学家合作参与了 XGBoost 开源项目，GitHub 上的这一项目（[https://github.com/dmlc/xgboost/](https://link.zhihu.com/?target=https%3A//github.com/dmlc/xgboost/)）约有 350 个贡献者，以及 3600 多条提交。和其他算法相比，XGBoost 算法的不同之处有以下几点：
> 
> - 应用范围广泛：该算法可以解决回归、分类、排序以及用户自定义的预测问题
> 
> - 语言：支持包括 C++、Python、R、Java、Scala 和 Julia 在内的几乎所有主流编程语言
> 
> - 云集成：支持 AWS、Azure 和 Yarn 集群，也可以很好地配合 Flink、 Spark 等其他生态系统

xgboost目标函数：我们知道XGBoost是由`k`个基模型组成的一个加法运算：

$$
\hat{y_i}^{(t)} = \sum_{k=1}^t f_k(x_i) = \hat{y_i}^{(t-1)} + f_t(x_i)
$$

xgboost的模型基本上可以用上面的公式来表达：

- xgboost是一个加法模型，由总共`k`颗数模型来表达。具体表现为训练过程中一直向前迭代，首先训练第一棵树，假设100分，达到了80分（实际的代码实现过程中一般都会降权，例如降权因子为0.1，实际得分就为8分）。然后依据第一棵树的表现开始训练第二棵树（在第一棵树结果的基础上开始训练）

- NOTES：$\hat{y}^{(t)}$是第`t`次迭代后样本`i`的预测结果；$f_t(x_i)$是第`t`棵树的模型预测结果；$\hat{y}^{(t-1)}$是第`t-1`棵树的预测结果

### 目标函数 - 损失函数

$$
Obj = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{j=1}^t \Omega(f_j)
$$

其中$l(y_i,\hat{y}_i)$是我们模型的损失函数，$\hat{y}^i$是整个模型在第`i`个样本上的预测值，$y_i$是第`i`个样本的真实值。NOTES：$\sum_{j=1}^t \Omega(f_j)$是全部`t`棵树的复杂度求和，在实际实现过程中一般用`L1`或`L2`正则表达式

从这个目标函数我们需要掌握的细节：① 模型是加性函数，向前迭代；② 前后部分是两个维度的计算问题。

**两个累加的变量是不同的**

- 一个是`i`，前面分式代表的是训练样本数量，也就是对每个样本我们都会计算一个损失，这个损失是总共`t`棵树的预测值之和与样本真实值之间的差值计算

- 另一个是累加变量`j`，代表的是**树**的数量，也就是我们对每棵树的复杂度进行累加计算

我们继续往前推导，$\hat{y}_i^t = \hat{y}_i^{(t-1)} + f_t(x_i)$，其中$\hat{y}_i^{(t-1)}$是第`t-1`步的模型给出的预测值，是已知常数，$f_t(x_i)$使我们当前需要加入（训练）的新模型的预测值，此时目标函数可以进一步表示为：

$$
\begin{equation*}
\begin{split}
Obj^t &= \sum_{i=1}^n l(y_i, \hat{y}_i^t) + \sum_{i=1}^t \Omega(f_i) \\
      &= \sum_{i=1}^n l(y_i, \hat{y}_i^{t-1} + f_t(x_i)) + \sum_i^t \Omega(f_i)
\end{split}
\end{equation*} 
$$

此时要优化目标函数，等价于要求解$f_t(x_i)$。

> 泰勒公式是将一个在$x=x_0$楚具有`n`阶导数的函数$f(x)$利用关于$x-x_0$ 的 `n`次多项式来逼近函数的方法，若函数$f(x)$在包含$x_0$的某个闭区间`[a,b]`上具有 `n` 阶导数，且在开区间$(a,b)$ 上具有`n+1`阶导数，则对闭区间 `[a,b]`上任意一点$x$有$f(x)=\sum_{i=0}^n \frac{f^{(i)}}{i!}(x-x_0)^i + R_n(x)$，其中多项式称为函数在$x_0$处的泰勒展开式，$R_n(x)$是泰勒公式的余项并且是$(x-x_0)^n$的高阶无穷小。

根据泰勒公式我们可以把函数$f(x+\Delta{x})$在点$x$处进行泰勒的二阶展开，可以得到下面的等式：

$$
f(x+\Delta{x}) \approx f(x) + f^{'}(x)\Delta{x} + \frac{1}{2}f^{''}\Delta{x}^2
$$

> **关注点：**
> 
> 上面的公式中，$\Delta{x}$对应的是第`t`颗树的模型$f_t(x)$，**$\Delta{x}$对应的是一颗树模型，是一个树模型，是一个树模型，重要的事情讲三遍，是一颗树模型，而不是一个具体的标量数值。** 而$x$对应的是$\hat{y}^{t-1}$，所以最后相应的损失函数应该是$l(y_i,\hat{y}_i^{(t-1)}+f_t(x_i)$

我们把$\hat{y}^{t-1}$看做是$x$，$f_t(x_i)$看做是$\Delta{x}$，因此可以将目标函数写为：

$$
Obj^{(t)} = \sum _{i=1} ^n \left [ l(y_i, \hat{y}_i^{t-1}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right ] + \sum_{i=1}^t \Omega(f_i)
$$

其中$g_i$为损失函数的一阶求导，$h_i$为损失函数的二阶求导，**注意这里的被求导函数是$\hat{y}_i^{t-1}$**。

$$
\begin{equation*}
\begin{split}
g_i &= \partial_{\hat{y}^{t-1}} l(y_i, \hat y^{t-1}) \\
h_i &= \partial^2_{\hat{y}^{t-1}} l(y_i, \hat y^{t-1}) \\
\end{split}
\end{equation*}
$$

我们以平方损失函数举例：

$$
\sum_{i=1}^n(y_i - (\hat{y}_i^{t-1}+f_t(x_i)))^2
$$

则可以写出：

$$
\begin{equation*}
\begin{split}
g_i &= \partial_{\hat{y}^{t-1}} (\hat{y}^{t-1} - y_i)^2 = 2(\hat{y}^{t-1} - y_i)  \\
h_i &= \partial^2_{\hat{y}^{t-1}}  (\hat{y}^{t-1} - y_i)^2  = 2\\
\end{split}
\end{equation*}
$$

由于在第`t`步时$\hat{y}^{t-1}$已经是一个已知的值所以$l(y_i, \hat{y}_i^{t-1})$是一个常数，其对函数的优化不会影响最终的结果，所以目标函数可以简写成：

$$
Obj^{(t)} \approx \sum _{i=1} ^n \left [ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right ] + \sum_{i=1}^t \Omega(f_i)
$$

所以我们只需要求出每一步损失函数的一阶导和二阶导的值（由于前一步的$\hat{y}^{t-1}$是已知的，所以这两个值是常数），然后最优化目标函数，就可以得到每一步的$f(x)$，最后根据加法模型得到一个整体模型。

### 目标函数 - 正则化

机器学习中，训练过程中降低损失函数是为了提升训练精度，具体就是使得我们的模型经过训练和训练样本的拟合度更高，从机器学习角度描述是为了降低模型的`偏差`(bias)。但是为了模型效果在实际业务上表现能够和训练时保持一致的高精度，我们还需要通过其它方式来达成，这个过程被称为降低模型的`方差`(variance)，而最常用的技术方案是通过正则化来实现(regulization)。通俗的讲正则化是通过降低模型的复杂度来实现降低模型方差。为了后续的描述和推导方便，我们首先引入一些符号定义：

$f_t(x) = w_{q(x)}$，$x$为某一个样本，等式中$q(x)$代表该样本在哪个叶子结点，而$w_q$则代表了样本所在叶子结点的取值$w$，所以$w_{q(x)}$就代表了每个样本$x$的具体取值（即预测值）。

xgboost的基模型使用的是分类回归决策树（CART），而树模型的复杂度可以由叶子结点$T$表示，基于一个朴素的观点：叶子少的树比叶子多的树更加简单，也就表示叶子结点少的树模型在位置数据上的表现更加稳定（这是一个很朴素但是正确的结论，现在基本上所有类型的模型在正则化方面都是基于这个朴素假设），假设叶子节点总数为$T$；另外，更加稳定的树模型，它的叶子结点的权重$w$也不能过高，所以目标函数的正则化计算项可以重写成：

$$
\Omega{(f_t)} = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_i^2
$$

即决策树模型最终的复杂度有生产的所有决策树的叶子节点数量和所有叶子节点权重所组成的向量的$L_2$范式共同决定，如下图所示：

![](https://pic1.zhimg.com/v2-e0ab9287990a6098e4cdbc5a8cff4150_r.jpg)

上图描述了基于决策树的XGBoost的正则式计算方式。

我们设$I_j = \{i | q_{x_i}=j\}$为决策树`q`的第`j`个叶子节点的样本集合，然后可以重写目标函数为：

$$
\begin{equation*}
\begin{split}
Obj^{(t)} &\approx \sum _{i=1} ^n \left [ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right ] + \sum_{i=1}^t \Omega(f_i) \\
 &= \sum_{i=1}^n \left[g_i w_{q_{(x_i)}} + \frac{1}{2} h_i  w^2_{q_{(x_i)}}   \right] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2 \\
 &= \sum_{j=1}^T \left[(\sum_{i \in I_j} g_i) w_j + \frac{1}{2} (\lambda + \sum_{i \in I_j}h_i) w_j^2 \right] + \gamma T \\
&, w_j \in R^T(叶子节点权重空间) \\
&, q: R^d -> \{1,2,\cdots,T\}（树结构，将样本映射到叶子节点）
\end{split}
\end{equation*}
$$

> NOTES：第二步到第三步的推导不太直观，可以从概念上去理解会更加容易。第二步的操作是先遍历所有样本，然后计算每个样本的损失函数，但是注意观察，损失函数的计算最终都是在**有效**的叶子节点上。这里的`有效`是指针对一个具体样本，它最终是落在每棵决策树的具体一个节点上，最终生效的叶子节点其实是模型中决策树的数量。这里遍历所有的节点数量`T`，其实针对具体一个样本很多叶子节点是无效的，但是针对样本集基本上每个叶子节点都存在若干样本。所以，三步的逻辑是我们从遍历样本转换成先遍历叶子节点，然后遍历每个叶子节点上的样本，实现了步骤二相同的计算逻辑

为了简化公式，我们定义：$G_j = \sum_{i \in I_j} g_i$，$H_j = \sum_{i \in I_j} h_i$，则目标函数简化为：

$$
Obj^{(t)} = \sum_{j=1}^T \left[ G_j w_j + \frac{1}{2} (\lambda + H_j)w_j^2 \right] + \gamma T
$$

进一步观察，我们可以看到$G_j, H_j$都是前面`t-1`步骤已经计算出来的，所以只有最后一棵树的叶子节点$w_j$是未知数，我们要计算目标函数的最小值。因此，针对固定的决策树结构（即固定结构的$q$）求解$w_j$一阶求导，并令其为0，则可以求解出叶子节点`j`对应的权值：

$$
w^{\star}_j = - \frac {\sum_{i \in I_j}g_i} {\sum_{i \in I_j}h_i  + \lambda} = - \frac {G_j} {H_j+\lambda} 
$$

所以，目标函数可以简化为：

$$
Obj^{t} = -\frac{1}{2} \times \sum\limits_{j=1}^T\frac{ G_j^2}{H_j+\lambda} + \gamma T 
$$

下面给出一个具体的图示来展示计算：

![](https://raw.githubusercontent.com/dmlc/web-data/master/xgboost/model/struct_score.png)

> 计算过程：
> 
>   ① 计算每个样本的一阶导数和二阶导数
>   ② 针对每个节点所包含的样本求和得到$G_j$和$H_j$
> 
>   ③ 遍历决策树节点可以得到目标函数

到目前为止，我们主要在介绍如何通过样本来更新模型的权重（即叶子节点的权重值）。但是XGBoost的模型不仅仅是叶子节点权重，还包括`Tree Structure`。是的，XGBoost的模型包括：`Tree Structure + Leaf Weight`两部分组成。前文一直在介绍如何更新叶子权重，下面将介绍前面一部分，即如何构建CART树。

### CART树构建

> Classification and Regression Tree (CART) for short is a term introduced by [Leo Breiman](https://en.wikipedia.org/wiki/Leo_Breiman) to refer to [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree_learning) algorithms that can be used for classification or regression predictive modeling problems. Classically, this algorithm is referred to as “decision trees”, but on some platforms like R they are referred to by the more modern term CART. The CART algorithm provides a foundation for important algorithms like bagged decision trees, random forest and boosted decision trees.

In a word，CART就是决策树的说法，它通常是一颗二叉树，每一个节点代表了对于一个特征维度的二维划分，而叶子节点包含了一个具体的数值`y`，被作为一个输入样本的预测值。

假设我们有一个数据集只有两个特征：高度（单位为厘米）、长度（单位是千米），希望通过这两个维度的数字来预测一个输入$x={(x_1,x_2)}$所属的性别`F / M`，下图是一个简单的示例：

![](https://machinelearningmastery.com/wp-content/uploads/2016/02/Example-Decision-Tree.png)

从上面的描述可以看出，一颗决策树实际上是一个对输入空间的划分，我们可以把每一个输入的变量当做是一个`p维空间`里的一个维度，而决策树把空间划分成一个多维举矩阵的空间。

<img title="" src="https://scikit-learn.org/stable/_images/iris.svg" alt="" width="498" data-align="center">

常见决策树有 `ID3`、`C4.5/5.0`、`CART`，其中`ID3`和`C4.5`生成的决策树是多叉树，只能处理分类而不能处理回归。而CART是分类回归树，既可以分类也可以回归。

ID3使用`信息增益`做为特征选择器来分割节点，C4.5使用的是`信息增益率`选择特征（避免信息增益指标偏向于多值属性），CART树选择的是`Gini Index`基尼系数选择特征，基尼系数代表了模型的不纯度，**基尼系数越小，不纯度越低，特征越好**。这和信息增益（率）相反。

#### Gini Index

数据集`D`的纯度可以用基尼值来衡量：

$$
Gini(D) = \sum\limits_{i=1}^n p(x_i)(1-p(x_i)) = 1 - \sum_{i=1}^n p(x_i)^2
$$

其中，数据集`D`可以看成是CART树上具体一个节点上的数据集，通常是全部训练集的一个子集（可以简单看做是训练过程中输入到这个节点的数据集），$p(x_i)$是分类$x_i$出现的概率，`n`是分类的数量，Gini(D)反映了从数据集D中随机抽取两个样本，其类别标记不一致的概率。因此，**Gini(D)越小，则[数据集D的纯度越高**。

对于样本D，个数为|D|，根据特征A 是否取某一可能值$\alpha$，把样本D分成两部分 $D_1,D_2$，所以CART树建立的是一颗二叉树：

$$
D_1=(x,y) \in D | A(x) = \alpha, D_2 = D - D_1
$$

在属性A的条件下，样本D的基尼系数定义为：

$$
GiniIndex(D | A \le \alpha) = \frac{|D_1|}{|D|} Gini(D_1) + \frac{|D_2|}{|D|} Gini(D_2) 
$$

#### 最优切分点划分算法

##### 贪心算法

贪心算法在XGBoost里通过设置参数`tree_method=exact`来选择。

> 1. 从深度为0的树开始，第一个叶子节点开始（包含所有样本），对每个特征计算最佳分裂值，并获取分裂收益。计算逻辑是按照**每个特征首先按照特征值做升序排列**，然后线性扫描的方式来计算并确定最佳分裂点，并记录该分裂的收益值（计算公式下面内容介绍）
> 
> 2. 步骤`1`中选取收益最大的分裂特征和分裂点进行分裂，生成两个子节点，并未新节点关联对应的样本集
> 
> 3. 回到步骤`1`，递归执行知道满足特定条件（最大树深度、最大子节点数量、无法达到分裂条件等）
> 
> 4. 将训练得到的CART树加入到模型中：$\hat{y}^{(t)} = \hat{y}^{(t-1)} + f_t(x)$
>    
>    ① 通常在实际应用中我们为了防止过拟合： $\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta f_t(x)$
>    ② $\eta$通常被称为学习步长或学习率，通常设置为0.1
>    
>    ③ 这意味着我们在训练的时候是朝着最优化目标求解模型参数（即求解$f_t(x)$的时候不使用参数$\eta$），但是在预测的时候使用（同时意味着在训练$f_{t+1}(x)$的时候超参数$\eta$会发挥作用）

介绍分裂收益计算公式前，我们先引入一些符号定义。假设$I_L$和$I_R$是节点分裂后左节点和右节点的样本集，则分裂前的样本集合为$I=I_L \cup I_R$，那么分裂收益计算公式如下：

$$
\begin{equation*}
\begin{split}
Gain_{split} &= \frac{1}{2} \left[\frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L}h_i + \lambda} + 
    \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R}h_i + \lambda} 
    - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I}h_i + \lambda} \right] 
    - \gamma  \\
            &= \frac{1}{2} \left[   
                    \frac{G_L^2}{H_L+\lambda}
                  + \frac{G_R^2}{H_R+\lambda}
                  - \frac{(G_L+G_R)^2}{(H_L+H_R+\lambda}  
                 \right] - \gamma
\end{split}
\end{equation*}
$$

我们可以发现对于所有的分裂点$\alpha$，我们只要做一遍从左到右的扫描就可以枚举出所有分割的梯度和$G_L$和 $G_R$。然后用上面的公式计算每个分割方案的分数就可以了。

#### 近似算法

贪心算法在求解每个特征的最佳分裂点的时候会遍历这个特征所有的取值，并且在遍历前会排序，在数据量大的情况下是xgboost最耗时的地方。为了提升训练效率，xgboost内部实现了近似算法，通过设置参数`tree_method=approx`

针对每个特征，`exact`算法会遍历每个特征值，十分耗时。近似算法会针对每个特征的取值范围划分`分位点`，每个分位点代表了不同特征值包含训练样本的数量占比。下面给出`appox`算法的伪代码

> // Approximate Algorithm for Split Finding in XGBoost
> for k=1 to m do
>     Propose $S_k = \{s_{k1}, s_{k2}, ..., s_{km}\}$ by percentiles on feature k
>     Proposal can be done per tree (global), or per split (local)
> end
> 
> for k=1 to m do
>     $G_{kv} =  \sum_{j \in {j|s_{k,v} \ge x_{j,k} \ge s_{k,v-1} g_j} }$
>     $H_{kv} =  \sum_{j \in {j|s_{k,v} \ge x_{j,k} \ge s_{k,v-1} h_j} }$
> end
> 
> Follow the same steps as in previous section to find max score only among proposed splits

- **第一个`for`循环**：对特征 k 根据该特征分布的分位数找到切割点的候选集合 $S_k = {s_{k1}, s_{k2}, ..., s_{km}}$

- **针对第二个`for`循环**：针对每个特征的候选集合，将样本映射到由该特征对应的候选点集构成的分桶区间中，即$s_{k,v} \ge x_{j,k} \ge s_{k,v-1}$，对每个桶统计$G,H$值，最后在这些统计量上寻找最佳分裂点

下面是`approx`算法的一个示例：

<img title="" src="https://pic2.zhimg.com/80/v2-5d1dd1673419599094bf44dd4b533ba9_1440w.jpg" alt="" data-align="inline">

$$
Gain = max \{ \\
  Gain, \\
  \frac{G_1^2}{H_1+\lambda} + \frac{G_{23}^2}{H_{23}+\lambda} - \frac{G_{123}^2}{H_{123}+\lambda} - \lambda, \\
  \frac{G_{12}^2}{H_{12}+\lambda} + \frac{G_{3}^2}{H_{3}+\lambda} - \frac{G_{123}^2}{H_{123}+\lambda} - \lambda
\}
$$

根据样本特征进行排序，然后基于分位数进行划分，并统计三个桶内的$G,H$值，最终求解节点划分的增益。

****加权分位数缩略图****

事实上， XGBoost 不是简单地按照样本个数进行分位，而是以二阶导数值 $h_i$作为样本的权重进行划分，如下：

![](https://pic4.zhimg.com/80/v2-5f16246289eaa2a3ae72f971db198457_1440w.jpg)

为什么要用$h_i$进行样本加权？这是因为通过对$Obj$函数进行变形，可以将$h_i$提取到外部看成是权重。举一个例子：

假设我们分配到一个叶子节点只有一个样本，那么权重最优值计算公式如下：

$$
w^\star_j = (\frac{1}{h_j+\lambda}) (-g_j)
$$

其中$-g_j$就是反向梯度，而$1/(h_j+\lambda)$就是一个学习率。权重的最佳值就是负的梯度乘以一个权重系数，该系数类似于随机梯度下降中的学习率。观察这个权重系数，我们发现，h_j越大，这个系数越小，也就是学习率越小。h_j越大代表什么意思呢？代表在该点附近梯度变化非常剧烈，可能只要一点点的改变，梯度就从10000变到了1，所以，此时，我们在使用反向梯度更新时步子就要小而又小，也就是权重系数要更小。这里是一个直观的解释，完整的公式推导如下：

$$
\begin{equation*}
\begin{split}
Ojb^{(t)} &\approx \sum_{i=1}^n \left[g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \sum_{j=1}^t\Omega(f_i) \\
    &= \sum_{i=1}^n\left[g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) + \frac{1}{2} \frac{g_i^2}{h_i} \right] + \Omega(f_t) + C \\
    &= \sum_{i=1}^2\frac{1}{2}h_i \left[f_t(x_i)-(-\frac{g_i}{h_i}) \right]^2 + \Omega(f_t) + C

\end{split}
\end{equation*}
$$

其中$\frac{1}{2}\frac{g_i^2}{h_i}$与$C$都是常数，同时我们可以看到$h_i$就是平方损失函数中样本的权重。

## LightGBM

> 先给一个概念性的总结来给LightGBM vs XGBoost定性：
> 
> - Hitogram
> 
> - GOSS算法
> 
> - EFB算法
> 
> - Voting算法
> 
> - Leaf-wise树生长
> 
> - Categorical feature内部自动处理
> 
> 通过上述三个主要算法优化策略，LightGBM对比XGBoost最大的优点是 **快** 

xgboost最大的问题是计算效率，从上面对xgboost的介绍也可以看到最大的计算资源消耗在分裂点的计算过程。分裂点计算涉及到：① 对每个feature在所有的样本上的排序，如果样本特征空间很大，以及数据集也很大，那么这个排序需要耗费大量的资源：$N_f \times Ns$。其中$N_f$是特征空间维度，$N_s$是样本数量。LightGBM的提出也正是针对xgboost存在的问题。先给出LightGBM的改进点：

- 基于Histogram的决策树算法（xgboost也实现了）

- **单边梯度采样（Gradient-based One-Side Sampling, GOSS）** 使用GOSS可以减少大量只具有小梯度的数据实例，这样在计算信息增益的时候只利用剩下的具有高梯度的数据就可以了，相比XGBoost遍历所有特征值节省了不少时间和空间上的开销

- **互斥特征捆绑 (Exclusive Feature Bundling, EFB)** 使用EFB可以将许多互斥的特征绑定为一个特征，这样达到了降维的目的

- **Leaf-wise决策树生长策略** 大多数GBDT工具使用低效的按层生长 (level-wise) 的决策树生长策略，因为它不加区分的对待同一层的叶子，带来了很多没必要的开销。实际上很多叶子的分裂增益较低，没必要进行搜索和分裂。LightGBM使用了带有深度限制的按叶子生长 (leaf-wise) 算法

- **直接支持类别特征** 

### 直方图策略

Histogram algorithm并不是一个很新颖的创新，在统计学被大量使用。这里也只是给一个简单的介绍。直方图算法的基本思想是：先把连续的浮点特征值离散化成 $k$个整数，同时构造一个宽度为 $k$ 的直方图。在遍历数据的时候，根据离散化后的值作为索引在直方图中累积统计量，当遍历一次数据后，直方图累积了需要的统计量，然后根据直方图的离散值，遍历寻找最优的分割点。特征离散化有很多优点：存储方便、运算更快、鲁棒性强、模型更加稳定等，我们在这里突出最显著的两个优点：

![](https://www.researchgate.net/publication/346577317/figure/fig3/AS:1001743705993216@1615845719667/Histogram-algorithm.png)

![](https://www.researchgate.net/publication/350848994/figure/fig2/AS:1019742676582401@1620137008154/Histogram-algorithm-of-LightGBM.ppm)

-     **内存占用小**： 直方图算法不仅不需要额外存储预排序的结果，而且可以只保存特征离散化后的值，而这个值一般用$8$位整型存储就足够了，内存消耗可以降低为原来的 $8$ 。也就是说XGBoost需要用$32$位的浮点数去存储特征值，并用$32$位的整形去存储索引，而 LightGBM只需要用$8$ 位去存储直方图，内存相当于减少为 $\frac{1}{8}$；

![](https://pic4.zhimg.com/v2-3064f201bc8545f851c7ccf47921c0e7_r.jpg)

- **计算代价更小**：预排序算法XGBoost每遍历一个特征值就需要计算一次分裂的增益，而直方图算法LightGBM只需要计算 $k$次（ $k$可以认为是常数），直接将时间复杂度从$O(\#data \times \#feature) $ 降低到$ O(k \times \#feature) $，而我们知道$\#data >> k$。

由于特征被离散化后，找到的并不是很精确的分割点，所以会对结果产生影响。但是试验显示离散化的分割点对最终的精确度影响微乎其微，甚至有时候效果更加好。其原因是决策树在这里是一个弱模型（weak learner），分割点不精准对最终模型的影响不大，而且更加粗略的分割点也能一定程度上起到正则化的效果，可以有效防止过拟合。另外，即使单棵决策树的效果不好，但是在梯度提升(Gradient Boosting)的框架下影响不大，这一切都是采用Histogram策略的基础。

LightGBM的另外一个优化是对Histogram做**差加速**。`一个叶子的直方图可以由它的父亲节点的直方图与它兄弟的直方图做差得到，在速度上可以提升一倍`。通常构造直方图时，需要遍历该叶子上的所有数据，但直方图做差仅需遍历直方图的k个桶。在实际构建树的过程中，LightGBM还可以先计算直方图小的叶子节点，然后利用直方图做差来获得直方图大的叶子节点，这样就可以用非常微小的代价得到它兄弟叶子的直方图。

![](https://pic4.zhimg.com/v2-b51f2764c13ca0a7b4cb41849a367a87_b.jpg)

### 带深度限制的Leaf-Wise算法

$$
L(y,\hat{y}) = ln(1+e^{-y\hat{y}}), y \in \{-1, 1\}) \\
g = L^{'} = \frac {-y} {1+e^{y\hat{y}}} \\
h = g^{'} = \left(  \frac{-y}{1+e^{y\hat{y}}}\right)^{`} = 
$$

## CatBoost
