---
layout: post
title: Numerically Stable Softmax
description: Note on different ML topic
tags: mlnote
date: 2026-09-22
featured: false
giscus_comments: false
---

- The softmax function returns the probability of a multinoulli distribution given a real-valued vector.
- Given a vector of logits $\textbf{x} \in R^{K}$, it returns the probability $p_i$ over $K$ outcomes, where $\sum_{i}^{k} p_i = 1$.

  $$
  \text{softmax}(x_i) = \frac{\text{exp}(x_i)}{\sum_{j}^{K} \text{exp} (x_j)}
  $$

- Key properties of $\text{exp}$, also written as, $e^x$:
  - Always positive and $exp(0) = 1$.
  - Always increases- negatives are close to 0, positives are above 1.
  - Rate of increase is *exponential* ; it is it's own derivative.
- The $\text{exp}$ is necessary because $x_i$ could be negative, which will return negative $p_i$ (invalid).
- ⚠️The equation is numerically unstable to implement, it faces both numerical overflow and underflow.
  - Overflow is when a large value rounds up as infinity. In this case, when $x_i$ are very large, it will overflow.
  - Underflow is when a value close to 0 rounds down as 0. In this case it happens when $x_i$ are very negative.
  - Define $\mathbf{z} = \mathbf{x} - \text{max}_i x_i$ and calculate $\text{softmax}(\mathbf{z})$
  - This addresses the both problem.
    - The exponents $e^{z_i}$ are now in $(0,1]$.
    - Overflow: the largest $z_i$ is now 0.
    - Underflow: at least one denominator is 1 (because of the largest $z_i$ is 0). Hence, division by zero is avoided. **Still possible to underflow though because of the numerator when values are further spread, but negligible as small-to-none probability**

## Log Softmax
- It is also implemented similarly to avoid underflow and overflow.

  $$
  \log \text{softmax} \left(x_i\right) = \text{log}\frac{\text{exp}(x_i)}{\sum_{j}^{K} \text{exp} (x_j)}
  $$

- The derivation from the original equation is,
  - $\log \text{softmax} \left(x_i\right) =\text{log}\ e^{x_i} - \text{log} \sum_{j} e^{x_j}$
  - $\log \text{softmax} \left(x_i\right) =x^{i} - \text{log} \sum_{j} e^{x_j}$
  - Same as before, if we take out the max, $m = \max_j x_j$:
  
    $$
    x_i - \log\left(\sum_j e^{x_j}\right)
    =
    x_i - \log\left(e^m \sum_j e^{x_j-m}\right).
    $$
  
  - $\log \text{softmax} \left(x_i\right) =x^{i} - \log e^m - \log \sum_{j} e^{x_j - m}$
  - This is the final numerically stable form: $\log \text{softmax} \left(x_i\right) =x^{i} - m\ - \text{log} \sum_{j} e^{x_j - m}$