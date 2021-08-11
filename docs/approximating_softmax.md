# Approximating Softmax

[TOC]

在进入具体的方法描述前，先对问题给出一些定义。假设有一个语料库，包含一个语句序列：$w_1,w_2,\ldots,w_T$。这些单词来源于一个预先定义的词表$V$，它的大小使用$|V|$描述。我们在下面的讨论中会引入语言模型，其中有上下文的概念，使用$c$表示（$|c|=n$），即上下文的窗口大小为$n$。每一个单词的 ‘**word embedding**’表示为$v_w$，它的维度设定为$d$。则在上下文范围内计算单词概率公式：
$$
p(w|c) = \frac {exp(h^T\cdot v_w)}{\sum_{w_i \in V exp(h^T \cdot v_{w_i})}}
$$
其实上面的公式就是一个softmax计算。可以看到，如果词表很大的情况下，计算$h^T \cdot v_{w_i}$需要耗费很多的计算资源。例如词表大小为3万，则需要3万次矩阵计算。下面内容会介绍各种不同的方案，解决上述计算量过大的问题

参考文献：[Strategies for Training Large Vocabulary Neural Language Models](https://arxiv.org/pdf/1512.04906.pdf)，里面包含了多种方案的效果对比

## Softmax-based Approaches

### Hierarchical Softmax



### Differentiated Softmax

### CNN-Softmax



## Sampling-based Approaches

### Importance Sampling

### Target Sampling

### Noise Contrastive Sampling

### Negative Sampling

### Self-Normalization

### Infrequent Normalization

### Other Approaches

## Which Approach to choose?

## Conclusion

