---
layout: post
title: 正则化解惑
date: 2020-01-05
author: 唐涵
categories: 机器学习
mathjax: true
---

本文记录了我在学习正则化模型（主要是Ridge Regression）时的一些困惑和思考。困惑主要有三点：

1. 正则化模型中为什么需要把特征x中心化（centering）或者标准化（standardization）？有什么好处？参数估计结果有哪些差异？
2. 为什么对y一般只需要中心化而不需要标准化？
3. 为什么不对bias项进行惩罚？

下面以Ridge Regression为例尝试对以上三个问题进行解释。基本设定如下：假设 $\mathbf{x}_i \in \mathbb{R}^p$ , $$y_i \in \mathbb{R}$$, $$i=1, \cdots, n$$ 为一组训练样本，我们希望找到$$w_0 \in \mathbb{R}$$ 和 $$\mathbf{w} \in \mathbb{R}^p$$ 使得 ridge regression 的目标函数


$$
\begin{eqnarray}
f(w_0, \mathbf{w}) &=& \text{MSE}(w_0, \mathbf{w}) + \lambda \left\lVert \mathbf{w}\right\rVert^2 \\
&=& \frac{1}{n} \sum \limits_{i=1}^{n} (y_i - w_0 - \mathbf{w}^T \mathbf{x}_i) ^ 2 + \lambda \left\lVert \mathbf{w}\right\rVert^2
\end{eqnarray}
$$


最小，其中 $$\lambda$$ 为事先给定的正则化系数。

## 困惑一：为什么需要把特征x中心化或标准化？

对变量 $$x$$ 中心化是指去掉它的均值，而标准化则还要除上它的标准差。

**中心化**

为什么要进行中心化？其实在看 *The Elements of Statistical Learning* 的 Ridge Regression (P63) 之前我对中心化也并不了解，平常听得和用得更多的是标准化。但是，看完书并做了对应的课后习题（Exercise 3.5）后我发现中心化不仅在**理论推导**上很有用，并且提供了**估计正则化模型参数的一般方法**。通常，为了把上面的函数 $$f$$ 写成矩阵形式，我们会人工加入一列 $$(1, 1, \cdots, 1)^T$$作为 bias 项所对应的列，从而得到 $$n \times (p+1)$$ 的设计矩阵 $$X$$，然后再把 $$w_0$$ 加入到 $$\mathbf{w}$$ 中得到 $$\mathbf{w}' \in \mathbb{R}^{p+1}$$。这样可以把 $$f$$ 写成矩阵形式：


$$
\begin{eqnarray}
f(w_0, \mathbf{w}) = f^{*}(\mathbf{w'}, \mathbf{w}) = \frac{1}{n} \left\lVert \mathbf{y} - X\mathbf{w}' \right\rVert^2 + \lambda \left\lVert \mathbf{w}\right\rVert^2.
\end{eqnarray}
$$


可以看到，由于我们没有将 bias 项加入到正则化项中（不加的原因在困惑三中说明），使得目标函数的矩阵形式是比较复杂的（既有 $$\mathbf{w}$$ 又包含 $$\mathbf{w}'$$），不能通过简单的对 $$\mathbf{w}'$$ 求导来求最优解。但是，将特征 $$x$$ 中心化之后，我们将看到参数 $$w_0$$ 和 $$\mathbf{w}$$ 的最优解都会有一个**显示表达式**。

另外，如果我们对 $$f(w_0, \mathbf{w}) = \displaystyle{\frac{1}{n} \sum \limits_{i=1}^{n} (y_i - w_0 - \mathbf{w}^T \mathbf{x}_i) ^ 2 + \lambda \left\lVert \mathbf{w}\right\rVert^2}$$ 关于 $$w_0$$ 求偏导并令其为0，则有：


$$
\begin{eqnarray}
-\frac{2}{n} \sum \limits_{i=1}^{n} (&y_i& - w_0 - \mathbf{w}^T \mathbf{x}_i) = 0 \\
\implies w_0 &=& \frac{1}{n} \sum \limits_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i) \\
\implies w_0 &=& \overline{y} - \mathbf{w}^T \overline{\mathbf{x}}
\end{eqnarray}
$$


其中，$$\overline{y} \in \mathbb{R}$$ 为 $$y$$ 的均值，$$\overline{\mathbf{x}} \in \mathbb{R}^p$$ 为特征的均值向量。可以看到，如果 $$\overline{\mathbf{x}}$$ 为0，则 $$w_0$$ 可以简单的用 $$y$$ 的均值 $$\overline{y}$$ 进行估计，**而将 $$x$$ 中心化以后再求它的均值向量 $$\overline{\mathbf{x}}$$ 刚好会得到0！** 这也是对特征 $$x$$ 进行中心化的一个原因：bias 项 $$w_0$$ 可以用 $$y$$ 的均值 $$\overline{y}$$ 估计！

下面推导将特征 $$x$$ 中心化以后参数 $$w_0$$ 和 $$\mathbf{w}$$ 的最优解。

首先，将目标函数 $$f(w_0, \mathbf{w})$$ 作恒等变换转换成新的函数 $$g(\overset{\sim}{w_0}, \mathbf{w})$$：


$$
\begin{eqnarray}
f(w_0, \mathbf{w}) &=& \frac{1}{n} \sum \limits_{i=1}^{n} \left(y_i - w_0 - \mathbf{w}^T \overline{\mathbf{x}} - \mathbf{w}^T(\mathbf{x}_i - \overline{\mathbf{x}})\right)^2 + \lambda \left\lVert \mathbf{w}\right\rVert^2 \\
&=& \frac{1}{n} \sum \limits_{i=1}^{n} (y_i - \overset{\sim}{w_0} - \mathbf{w}^T \overset{\sim}{\mathbf{x}_i}) + \lambda \left\lVert \mathbf{w}\right\rVert^2  \\
&\triangleq& g(\overset{\sim}{w_0}, \mathbf{w})
\end{eqnarray}
$$


其中，$$\overset{\sim}{w_0} = w_0 + \mathbf{w}^T \overline{\mathbf{x}}$$，$$\overset{\sim}{\mathbf{x}_i} = \mathbf{x}_i - \overline{\mathbf{x}}$$。于是，最小化函数 $$f$$ 等价于最小化函数 $$g$$。

其次，因为中心化之后的均值向量变为 $$\displaystyle{\overline{\overset{\sim}{\mathbf{x}}}} = 0$$ ，根据之前的说明，$$g$$ 关于 $$\overset{\sim}{w_0}$$ 求导得到 $$\overset{\sim}{w_0}$$ 的估计为 $$\overline{y}$$，代入 $$g$$ 中得到：


$$
g(\mathbf{w}) = \frac{1}{n} \sum \limits_{i=1}^{n} \left(\overset{\sim}{y_i} - \mathbf{w}^T \overset{\sim}{\mathbf{x}_i} \right)^2 + \lambda \left\lVert \mathbf{w}\right\rVert^2
$$



其中，$$\overset{\sim}{y_i} = y_i - \overline{y}$$。

若令


$$
\overset{\sim}{X} =
\begin{bmatrix}
\overset{\sim}{\mathbf{x}_1}^T \\
\vdots \\
\overset{\sim}{\mathbf{x}_n}^T
\end{bmatrix},
\quad
\overset{\sim}{\mathbf{y}} =
\begin{bmatrix}
\overset{\sim}{y_1} \\
\vdots \\
\overset{\sim}{y_n}
\end{bmatrix}
$$



则可以将 $$g$$ 写成矩阵形式：


$$
g(\mathbf{w})= \frac{1}{n} \left\lVert \overset{\sim}{\mathbf{y}} - \overset{\sim}{X}\mathbf{w} \right\rVert^2 + \lambda \left\lVert \mathbf{w}\right\rVert^2.
$$



上式类似 Ridge Regression 在教科书上通常定义的目标函数。对 $$\mathbf{w}$$ 求偏导并令结果为0可以得到：


$$
\mathbf{w} = \left(\overset{\sim}{X}^T \overset{\sim}{X} + n \lambda I \right)^{-1} \overset{\sim}{X}^T \overset{\sim}{\mathbf{y}}.
$$



同时，原来的 bias 项 $$w_0 = \overset{\sim}{w_0} - \mathbf{w}^T \overline{\mathbf{x}} = \overline{y} - \mathbf{w}^T \overline{\mathbf{x}}$$。

因此，在训练正则化模型时我们可以首先把特征 $$x$$ 和标签 $$y$$ 中心化（相当于把坐标原点移动到均值点，这样在拟合模型时就不需要截距项了），然后忽略 bias 项，解一个最优化问题得出权重$$\mathbf{w}$$，最后再根据权重$$\mathbf{w}$$ 、标签均值$$\overline{y}$$和特征均值 $$\overline{\mathbf{x}}$$ 计算 bias 项 $$w_0$$。

**标准化**

为什么要进行标准化？这就要回到正则化的目的是什么。从统计的角度看，正则化是为了进行**变量选择**：加入正则化项后，一些对被解释变量 $$y$$ 影响较小（系数较小）的变量 $$x$$ 就被剔除掉了，只保留系数较大的变量。而为了保证变量之间**比较的公平性**，所有变量都必须是同一尺度。如果一个变量是以万为单位，而另一个变量只有可怜的零点几，则第二个变量在正则化模型中几乎不可能被选出来（它要对 $$y$$ 产生影响的话必须乘上一个很大的系数但是这样反过来会极大地增加惩罚项）。因此需要对所有变量进行标准化处理。接下来我们想知道将 $$x$$ 标准化后参数的估计有哪些变化。

将变量 $$x$$ 标准化，我们要最小化目标函数 $$h(w_0^{*}, \mathbf{w}^*)$$:


$$
\begin{eqnarray}
h(w_0^{*}, \mathbf{w}^{*})
&=& \frac{1}{n} \sum \limits_{i=1}^{n} \left(y_i - w_0^* - \sum \limits_{j=1}^{p} \frac{x_{ij} - \overline{x}_j}{\sigma_j} w_j^* \right)^2 + \lambda \lVert \mathbf{w}^* \rVert  \\
&=& \frac{1}{n} \sum \limits_{i=1}^{n} \left(y_i - w_0^* - \mathbf{w}^{*T}\Sigma^{-1}(\mathbf{x}_i - \overline{\mathbf{x}})\right)^2 + \lambda \lVert \mathbf{w}^* \rVert
\end{eqnarray}
$$


其中，$$\Sigma$$ 为 $$p \times p$$ 对角矩阵 $$\text{Diag}(\sigma_1, \cdots, \sigma_p)$$，$$\sigma_i$$ 为第 $$i$$ 个变量的标准差。若令 $$\mathbf{x}_i^* = \Sigma^{-1} (\mathbf{x}_i - \overline{\mathbf{x}})$$，则上式可进一步化简为：


$$
h(w_0^{*}, \mathbf{w}^*) = \frac{1}{n} \sum \limits_{i=1}^{n} \left(y_i - w_0^* - \mathbf{w}^{*T} \mathbf{x}_i^* \right)^2 + \lambda \lVert \mathbf{w}^* \rVert.
$$



对比函数 $$g(\overset{\sim}{w_0}, \mathbf{w})$$ 的表达式，可以发现它们的结构是类似的，因此可以得到 $$w_0^*$$ 的估计为 $$\overline{y}$$。如果我们再令 $$n \times p$$ 矩阵 $$X^*$$ 为


$$
X^* =
\begin{bmatrix}
{\mathbf{x}_1^{*T}} \\
\vdots \\
{\mathbf{x}_n^{*T}}
\end{bmatrix}，
$$



那么仿照求解 $$\mathbf{w}$$ 的过程就可以得到 $$\mathbf{w}^*$$ 的估计为 $$\left(X^{*T} X^* + n \lambda I \right)^{-1} X^{*T} \overset{\sim}{\mathbf{y}}$$，即


$$
\begin{eqnarray}
w_0^* &=& \overline{y}  \\
\mathbf{w}^* &=& \left(X^{*T} X^* + n \lambda I \right)^{-1} X^{*T} \overset{\sim}{\mathbf{y}}.
\end{eqnarray}
$$


又因为 $$X^* = \overset{\sim}{X} \Sigma^{-1}$$，所以


$$
\begin{eqnarray}
\mathbf{w}^*
&=& \left(X^{*T} X^{*} + n \lambda I \right)^{-1} X^{*
T} \overset{\sim}{\mathbf{y}} \\
&=& (\Sigma^{-1} \overset{\sim}{X}^T \overset{\sim}{X} \Sigma^{-1} + n \lambda I)^{-1} \Sigma^{-1} \overset{\sim}{X}^T  \overset{\sim}{\mathbf{y}}.
\end{eqnarray}
$$


对比 $$\mathbf{w}$$ 的表达式
$$
\mathbf{w} = \left(\overset{\sim}{X}^T \overset{\sim}{X} + n \lambda I \right)^{-1} \overset{\sim}{X}^T \overset{\sim}{\mathbf{y}}，
$$



可以发现虽然它们的表达式不同，但 $$\mathbf{w}^*$$ 也只是多了将 $$x$$ 中心化后再除上标准差这一步。

在实际建模中，如果特征之间的尺度差异不大，则一般只需要进行中心化（特征、标签都要），分两步求得权重 $$\mathbf{w}$$ 和 bias 项 $$w_0$$；而如果特征之间的尺度相差很大，则需要把特征标准化，标签中心化，也分两步求得权重 $$\mathbf{w}^*$$ 和 bias 项 $$w_0^*$$。注意，这里我们得到的是关于 $$y$$ 的两个不同的估计式，也就是两个不同的模型！

## 困惑二：为什么对y一般只需要中心化而不需要标准化？

前面已经说明了将标签 $$y$$ 中心化后可以配合特征的中心化分两步求解各参数的估计。至于不需要对 $$y$$ 标准化的原因，目前为止我的想法是：中心化是为了进行**公平比较**，而这里我们并没有拿 $$y$$ 和其它变量去比，因此也就没有标准化的必要。

## 困惑三：为什么不对bias项进行惩罚？

因为进行正则化的目的是为了减少参数个数，也就是让我们的损失函数更加平滑（直线相比曲线更平滑），而 bias 项对函数的平滑程度没有影响，它只是进行上下平移！因此也就没有惩罚的必要了。
