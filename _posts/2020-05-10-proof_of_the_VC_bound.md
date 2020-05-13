---
layout: post
title: 机器学习基石笔记第二章：VC generalization bound的证明
date: 2020-05-10
author: 唐涵
categories: machine learning, VC theory
mathjax: true
---

林老师在机器学习基石课程的lecture 6中谈到了为什么机器可以学习（Why can machines learn）。老师以二分类问题（binary classification）为例，利用VC理论得到了泛化上界（VC genearlization bound），从而从理论上保证了学习的可行性。关于VC泛化上界的正确性，老师采用了比较直观的方式解释其合理性，我听了之后很受启发，之后也阅读了课本*Learning from data*附录中VC泛化上界的严格证明。虽然阅读的过程比较痛苦，但好在定理证明的逻辑相当清晰，坚持啃完之后受益匪浅（至少再长的证明都有勇气看下去了，哈哈）。

**定理1（VC generalization bound）**

对任意的 $$\delta \gt 0$$，不等式


$$
E_{\mathrm{out}}(g) \leq E_{\mathrm{in}}(g)+\sqrt{\frac{8}{N} \ln \frac{4 m_{\mathcal{H}}(2 N)}{\delta}}
$$



以$$\ge 1 - \delta$$的概率成立。

VC泛化上界可以由下面的VC不等式（VC inequality）推得：

**定理2（VC inequality）**


$$
\mathbb{P}\left[\sup_{h\in \mathcal{H}} \vert E_{\text{in}}(h) - E_{\text{out}}(h) \vert \gt \epsilon \right] \le 4 m_{\mathcal{H}}(2N)e^{-\frac{1}{8}\epsilon^2N}.
$$



这里有两点要注意：

1. 因为我们最后选择的 $$g$$ 也是 $$\mathcal{H}$$ 中的一员，所以事件 $$\{\vert E_{\text{in}}(g) - E_{\text{out}}(g) \gt \epsilon \vert \}$$ 被包含在事件 $$\{\sup_{h\in \mathcal{H}} \vert E_{\text{in}}(h) - E_{\text{out}}(h) \vert \gt \epsilon \}$$ 中，因此前者的概率 $$\mathbb{P} \left[ \vert E_{\text{in}}(g) - E_{\text{out}}(g) \gt \epsilon \vert\right]$$ 也会被 $$4 m_{\mathcal{H}}(2N)e^{-\frac{1}{8}\epsilon^2N}$$ 压制。
2. 这里的随机性来源于数据集$$\mathcal{D}$$的不确定性，它是iid的从 $$P_{XY}$$ 中产生的 $$N$$ 个数据 $$\{(\mathbf{x}_i, y_i)\}_{i=1}^N$$，因此概率是对随机产生的大小为$$N$$的数据集$$\mathcal{D}$$取的。

最后，只需要令 $$\delta$$ 等于 $$4 m_{\mathcal{H}}(2N)e^{-\frac{1}{8}\epsilon^2N}$$，反解出$$\epsilon$$就可以得到VC泛化上界。因此，我们将重心放在证明VC不等式。接下来的证明来自课本的附录，我会补充一些细节以及自己的理解。

首先，由于我们不知道$$(X, Y)$$的具体分布$$P_{XY}$$，所以也就无法计算直接计算$$E_{\text{out}}$$。其次，正如前面提到的注意2，数据集$$\mathcal{D}$$的不确定性是另一大麻烦。因此，证明的核心即为寻找 $$E_{\text{in}} - E_{\text{out}}$$ 的一个可以计算的替代品，并且将问题先限制在某个固定的数据集上，然后再想办法进一步推广。

## $$E_{\text{in}} - E_{\text{out}}$$ 的替代品： $$E_{\text{in}} - E_{\text{in}}^\prime$$

我们引入第二个数据集$$\mathcal{D}^\prime$$，称为ghost数据集。它是独立于$$\mathcal{D}$$，从$$P_{XY}$$中产生的新的$$N$$个数据。Ghost数据集只是理论分析的一个工具，实际上我们并没有真的再次抽取$$N$$个数据。我们希望用ghost数据集上的$$E_{\text{in}}^\prime$$（in sample error ） 替代$$E_{\text{out}}$$（out of sample error ），也就是找到$$\mathbb{P}(\vert E_{\text{in}} - E_{\text{in}}^\prime \vert \text{很大})$$ 的上界。这就有了接下来的引理1（对应课本Lemma A.2），它告诉我们这样的替换是可行的。

**引理 1**


$$
\left(1-2 e^{-\frac{1}{2} \epsilon^{2} N}\right) \mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{out}}(h)\right|>\epsilon\right] \leq \mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2}\right],
$$



其中，RHS的概率取$$\mathcal{D}$$和$$\mathcal{D}^\prime$$的联合概率。

**Proof.** 
首先，我们可以假设 $$\mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{out}}(h)\right|>\epsilon\right] \gt 0$$。利用概率的性质以及条件概率的定义，可以得到


$$
\begin{aligned}
& \mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2}\right] \\
\geq & \mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2} \bigcap \sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{out}}(h)\right|>\epsilon\right] \\
=& \mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{out}}(h)\right|>\epsilon\right] \times \\
& \mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2}\left| \; \sup _{h \in \mathcal{H}} \vert E_{\mathrm{in}}(h)-E_{\mathrm{out}}(h) \vert \gt \epsilon \right. \right].
\end{aligned}
$$


现在考虑最后一项


$$
\mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2} \left| \; \sup _{h \in \mathcal{H}} \vert E_{\mathrm{in}}(h)-E_{\mathrm{out}}(h) \vert \gt \epsilon \right. \right].
$$


这个条件概率基于事件


$$
B = \{\mathcal{D}: \sup _{h \in \mathcal{H}} \vert E_{\mathrm{in}}(h)-E_{\mathrm{out}}(h) \vert \gt \epsilon  \}.
$$



现在我们随机选取$$B$$中的一个数据集$$\mathcal{D}$$固定住，并将此记为事件$$B_\mathcal{D}$$，那么在这个数据集$$\mathcal{D}$$上一定满足 $$\sup _{h \in \mathcal{H}} \vert E_{\mathrm{in}}(h)-E_{\mathrm{out}}(h) \vert \gt \epsilon$$。因此，一定存在一个hypothesis $$h^* \in \mathcal{H}$$ 使得 


$$
\vert E_{\mathrm{in}}(h^*)-E_{\mathrm{out}}(h^*) \vert \gt \epsilon.
$$



由于$$h^*$$只依赖于$$\mathcal{D}$$而与ghost数据集$$\mathcal{D}^\prime$$的选取无关，所以$$h^*$$相对ghost数据集$$\mathcal{D}^\prime$$是固定的。由Hoeffding不等式可以得到


$$
\begin{array}{l}
\mathbb{P}\left[\left|E_{\text {in }}^{\prime}\left(h^{*}\right)-E_{\text {out }}\left(h^{*}\right)\right| \leq \frac{\epsilon}{2}\biggm| B_{\mathcal{D}}, B \right] \geq 1-2 e^{-\frac{1}{2} \epsilon^{2} N}.
\end{array}
$$



注意这里的概率是对ghost数据集$$\mathcal{D}^\prime$$取的，所以才要求$$h^*$$必须不随$$\mathcal{D}^\prime$$变化而变化。

现在最后一项
$$
\mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2} \biggm| B \right]
$$
可以写成：


$$
\sum_{\mathcal{D} \in B} \mathbb{P}\left[B_\mathcal{D} \left| B \right.\right] \mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2} \biggm| B_{\mathcal{D}}, B \right].
$$



而对相乘的第二项，又有下列不等式成立


$$
\begin{align}
\mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2} \biggm| B_{\mathcal{D}}, B \right] 
&\ge  \mathbb{P} \left[ \vert E_{\mathrm{in}}(h^*)-E_{\mathrm{in}}^{\prime}(h^*) \vert \gt \frac{\epsilon}{2} \biggm| B_{\mathcal{D}}, B \right] \\
& \ge \mathbb{P} \left[ \vert E_{\mathrm{in}}^{\prime}(h^*)-E_{\mathrm{out}}(h^*) \vert \le \frac{\epsilon}{2} \biggm|  B_{\mathcal{D}}, B \right] \\
& \geq 1-2 e^{-\frac{1}{2} \epsilon^{2} N}.  \label{eq: 1} \tag{1}
\end{align}
$$


第一个不等式成立是因为事件
$$
\{\vert E_{\mathrm{in}}(h^*)-E_{\mathrm{in}}^{\prime}(h^*) \vert \gt \frac{\epsilon}{2} \}
$$
可以推出事件
$$
\{\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2}\}
$$
，第二个不等式成立则是因为事件
$$
\{\vert E_{\mathrm{in}}(h^*)-E_{\mathrm{out}}(h^*) \vert \gt \epsilon\}
$$
 和事件
$$
\{\vert E_{\mathrm{in}}^{\prime}(h^*)-E_{\mathrm{out}}(h^*) \vert \le \frac{\epsilon}{2}\}
$$
 一起（二者取交）可以推出事件
$$
\{\vert E_{\mathrm{in}}(h^*)-E_{\mathrm{in}}^{\prime}(h^*) \vert \gt \frac{\epsilon}{2}\}
$$
，而第三个不等式则是由Hoeffding不等式得到的。

因为对事件$$B$$中任意的一个数据集$$\mathcal{D}$$，上面的不等式$\eqref{eq: 1}$都成立，所以


$$
\mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2} \biggm| B \right] \geq 1-2 e^{-\frac{1}{2} \epsilon^{2} N}.
$$



从而，我们证明了


$$
\left(1-2 e^{-\frac{1}{2} \epsilon^{2} N}\right) \mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{out}}(h)\right|>\epsilon\right] \leq \mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2}\right].
$$



引理1中的不等式还包含一项$$\left(1-2 e^{-\frac{1}{2} \epsilon^{2} N}\right)$$，为了去除它，我们可以假设$$e^{-\frac{1}{2} \epsilon^{2} N} \lt \frac{1}{4}$$，不然的话定理2（VC不等式）中的RHS $$4 m_{\mathcal{H}}(2N)e^{-\frac{1}{8}\epsilon^2N}$$ 会比1大（因为 $$m_{\mathcal{H}}(2N) \ge 1$$，并且 $$e^{-\frac{1}{8}\epsilon^2N} \gt e^{-\frac{1}{2}\epsilon^2N} \gt \frac{1}{4}$$）。于是就得到了


$$
\mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{out}}(h)\right|>\epsilon\right] \leq 2 \mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2}\right].
$$



## 压制 $$E_{\text{in}} - E_{\text{in}}^\prime$$

找到 $$E_{\text{in}} - E_{\text{out}}$$ 的替代品 $$E_{\text{in}} - E_{\text{in}}^\prime$$ 之后，接下来就要想办法找到后者的一个上界。首先，注意到概率$$\mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2}\right]$$ 是对数据集 $$\mathcal{D}$$ 和数据集 $$\mathcal{D}^\prime$$ 的联合分布取的，它们是各自独立地从 $$P_{XY}$$ 中随机产生的 $$N$$ 个数据点的集合，因此产生 $$\mathcal{D}$$ 和 $$\mathcal{D}^\prime$$ 的过程可以等价地视为以下两步：
1. 从 $$P_{XY}$$ 中随机产生 $$2N$$ 个数据，记为数据集 $$\mathcal{S}$$；
2. 从 $$\mathcal{S}$$ 中不放回地随机抽取 $$N$$ 个数据作为数据集 $$\mathcal{D}$$，剩下的则作为数据集 $$\mathcal{D}^\prime$$。

于是，由全概率公式可以得到


$$
\begin{aligned}
& \mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2}\right] \\
=& \sum_{S} \mathbb{P}[S] \times \mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2} \biggm| S\right] \\
\leq & \sup _{S} \mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\varepsilon}{2} \biggm| S\right].
\end{aligned}
$$


现在，我们已经把问题限制在了一个大小为$$2N$$的数据集 $$\mathcal{S}$$ 上，而 $$\mathcal{H}$$ 在这$$2N$$个点上所能产生的dichotomy的数量$$\mathcal{H}(\mathcal{S})$$是有限的，不会超过 $$m_{\mathcal{H}}(2N)$$。

又因为一个dichotomy对应到一个 
$$
\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|,
$$
所以不需要考虑$$\mathcal{H}$$中所有的hypothesis，只要取dichotomy的总数（设为 $$M$$）个作为类代表就可以了。记$$h_1, \cdots, h_M$$为产生这$$M$$个dichotomy的代表，则


$$
\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|=\sup _{h \in\left\{h_{1}, \ldots, h_{M}\right\}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|.
$$



于是，利用union bound就可以得到


$$
\begin{aligned}
& \mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2} \biggm| S\right] \\
=& \mathbb{P}\left[\sup _{h \in\left\{h_{1}, \ldots, h_{M}\right\}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2} \biggm| S\right] \\
\leq & \sum_{m=1}^{M} \mathbb{P}\left[\left|E_{\mathrm{in}}\left(h_{m}\right)-E_{\mathrm{in}}^{\prime}\left(h_{m}\right)\right|>\frac{\epsilon}{2} \biggm| S\right]  \\
\leq & M \times \sup _{h \in \mathcal{H}} \mathbb{P}\left[\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2} \biggm| S\right],
\end{aligned}
$$


这里也用到了事件 $$\{最大值 \gt 某数\}$$ 等价于事件 $$\{至少有一个值 \gt 某数\}$$这个性质（这个说法不是特别严谨，但主要精神是这样）。

如果再用上条件 $$M \le m_{\mathcal{H}}(2N)$$，并对$$\mathcal{S}$$取$$\sup$$就可以得到下面的引理2（课本Lemma A.3）：



**引理2**


$$
\mathbb{P}\left[\sup _{h \in \mathcal{H}}\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2}\right] \le m_{\mathcal{H}}(2N) \times \sup_{\mathcal{S}} \sup _{h \in \mathcal{H}} \mathbb{P}\left[\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2} \biggm| S\right],
$$



这里LHS的概率取$$\mathcal{D}$$和$$\mathcal{D}^\prime$$的联合概率，RHS的概率取将$$\mathcal{S}$$随机划分为$$\mathcal{D}$$和$$\mathcal{D}^\prime$$的概率分布。



引理2将$$\sup$$从概率里面提到了外面，这样就可以固定住一个$$h$$，寻找该$$h$$下
$$
\mathbb{P}\left[\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2} \biggm| S\right]
$$
的上界，这就有了接下来的引理3（课本Lemma A.4）。

**引理3**


$$
\mathbb{P}\left[\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>\frac{\epsilon}{2} \biggm| S\right] \le 2 e^{-\frac{1}{8}\epsilon^2N}，
$$


其中，概率取自$$\mathcal{S}$$随机划分为$$\mathcal{D}$$和$$\mathcal{D}^\prime$$的概率分布。



为了证明引理3，课本又引入了Lemma A.5（Hoeffding）：

**引理4**（Hoeffding）

令集合 $$\mathcal{A} = \{a_1, \cdots, a_{2N}\}$$，其中 $$a_n \in [0, 1]$$，并令$$\mu = \frac{1}{2N}\sum_{n=1}^{2N}a_n$$为这$$2N$$个数的均值。若 $$\mathcal{D} = \{z_1, \cdots, z_N\}$$ 为从 $$\mathcal{A}$$ 中进行不放回随机抽样所得到的大小为$$N$$的样本，则


$$
\mathbb{P}\left[\left|\frac{1}{N} \sum_{n=1}^{N} z_{n}-\mu\right|>\epsilon\right] \leq 2 e^{-2 \epsilon^{2} N}.
$$



利用引理4可以证明引理3:

$$\mathcal{S}$$ 对应 $$\mathcal{A}$$，可以令$$a_n = 1$$，若$$h(\mathbf{x}_n) \neq y_n$$，否则$$a_n=0$$，于是$$\{a_n\}$$就是$$h$$在$$\mathcal{S}$$上所犯错误（error）的集合。接下来把$$\mathcal{S}$$随机划分为$$\mathcal{D}$$和$$\mathcal{D}^\prime$$，得到两个in sample error


$$
E_{\mathrm{in}}(h)=\frac{1}{N} \sum_{a_{n} \in \mathcal{D}} a_{n}\\
E_{\mathrm{in}}^{\prime}(h)=\frac{1}{N} \sum_{a_{n}^{\prime} \in \mathcal{D}^{\prime}} a_{n}^{\prime}.
$$



此时引理4中的$$\mu$$为


$$
\mu=\frac{1}{2 N} \sum_{n=1}^{2 N} a_{n}=\frac{E_{\mathrm{in}}(h)+E_{\mathrm{in}}^{\prime}(h)}{2}.
$$



于是，


$$
\vert E_{\text{in}} - \mu \vert \gt t \iff \vert E_{\text{in}} - \frac{1}{2}(E_{\text{in}} + E_{\text{in}}^\prime) \vert \gt t \iff \vert E_{\text{in}} - E_{\text{in}}^\prime \vert \gt 2t.
$$



根据引理5，


$$
\mathbb{P}\left[\left|E_{\mathrm{in}}(h)-E_{\mathrm{in}}^{\prime}(h)\right|>2 t\right] \leq 2 e^{-2 t^{2} N}.
$$



取$$t=\frac{\epsilon}{4}$$即可证得引理3。

由引理1、2、3就可以证明定理2（VC不等式）。



**参考文献**

Abu-Mostafa, Y. S., Magdon-Ismail, M., & Lin, H. (2012). *Learning from data: a short course.* [United States]: AMLBook.com.

