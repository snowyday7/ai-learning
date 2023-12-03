# 十一 Transformer

谷歌2017年文章《All you need is attention》提出Transformer模型，文章链接：http://arxiv.org/abs/1706.03762 。下面是几个基于Transformer的主要的模型

## 1 Bert
[《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](https://arxiv.org/abs/1810.04805)

- 整个模型可以划分为`embedding`、`transformer`、`output`三部分
  1. embedding部分由word embedding、position embedding、token type embedding三个模型组成，三个embedding相加形成最终的embedding输入。
  2. transformer部分使用的是标准的Transformer模型encorder部分。
  3. output部分由具体的任务决定。对于token级别的任务，可以使用最后一层Transformer层的输出；对于sentence级别的任务，可以使用最后一层Transformer层的第一位输出，即[CLS]对应的输出。


## 2 GPT

- [GPT:《Improving Language Understanding by Generative Pre-Training》](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)
- [GPT2:《Language Models are Unsupervised Multitask Learners》](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

GPT为生成式模型。如果说BERT使用了Transformer模型中的encoder部分，那GPT就相当于使用了Transformer模型中的deconder部分。

## 3 Transformer XL

[《Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context》](http://arxiv.org/abs/1901.02860)

Transformer XL是Transformer模型的扩展，主要在encoder部分引入了位置编码，并且引入了动态的序列长度。

相比于传统的transformer模型，主要有以下两点修改：
1. 每个transformer节点除了使用本帧上层节点的数据外，还使用了上一帧上层节点的数据；
2. 在做position embedding的时候，使用相对位置编码代替绝对位置编码。

## 4 ALBERT

《ALBERT: A Lite BERT for Self-supervised Learning of Language Representations》

ALBERT（A Lite BERT）即轻量级的BERT，轻量级主要体现在减少传统BERT的参数数量，提高模型的泛化能力。相比于传统BERT有以下三点区别：

1. 降低embedding层的维度，在embedding层与初始隐藏层之间增加一个全连接层，将低纬的embedding提高至高维的隐藏层纬度，相当于对原embedding层通过因式分解降维；
2. 在transformer层之间进行参数共享；
3. 使用SOP（sentence order prediction）代替NSP（next sentence prediction）对模型进行训练，ALBERT参数规模比BERT比传统BERT小18倍，但性能却超越了传统BERT。
