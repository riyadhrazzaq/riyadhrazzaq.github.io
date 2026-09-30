---
layout: post
title: Cross Entropy (and numerical stability)
description: Note on different ML topic
tags: mlnote
date: 2026-09-30
featured: true
giscus_comments: false
---

# Intuition

- Maximizing the likelihood is equivalent to minimizing the negative of the likelihood.
- If we have a possible outcome of a Bernoulli event $y$ and the probability $q \in [0, 1]$, the **likelihood** is the product over the $N$ independent events:

  $$
  \prod_{i=1}^{N} q_i^{y_i} (1-q_i)^{1-y_i}
  $$

- Intuitively, the cross entropy loss measures how confident the model was and punishes accordingly.
  - If the event happened, $y=1$, and the model predicted $q = 0.9$, it was confident and correct.
  - For $y=0$ and the model predicted $q = 0.9$, it was confident and incorrect.

# The cross entropy

$$
L = -\sum_c^C p_c \log q_c \tag{1}
$$

- $C$ is the number of classes.
- $p$ is the reference distribution, $q$ is the hypothesis distribution.
- Optimization techniques work with minimization problems — therefore, we negate the likelihood.
- Only where $p_c = 1$ contributes to the loss. This removes the effect of $C-1$ logits from the sum, but their effect stays in the gradient.
- Since $p$ does not depend on the model parameters, and when $p_c = 0$ the multiplication is 0 as well, minimizing $(1)$ becomes minimizing the *negative log likelihood*:

  $$
  L = - \log q_y
  $$

  only where $p_y = 1$.

- For a sequence $T$, average them:

  $$
  L = - \frac{1}{T} \sum_{t}^T \log q_{t, p_t=1}
  $$

- For a batch of $B$:

  $$
  L = - \frac{1}{B} \frac{1}{T} \sum_{b}^B \sum_{t}^T \log q_{t, p_t = 1}^b
  $$

  - This is when the sequence length is constant within a batch.
  - Simply normalizing by the total sequence length works when it is not constant:

  $$
  L = -\frac{1}{\sum_{b}^{B} T_b } \sum_{b}^{B} \sum_{t}^{T_b} \log q_{t, p_t = 1}^b
  $$

# Specific cases

**Binary cross entropy**

$$
L = -\big[\, p \log q + (1 - p)\log(1 - q) \,\big]
$$

- $p \in \{0,1\}$
- $q \in P(\text{class} = 1)$

**Multi-label multi-class**

$$
L = - \frac{1}{C} \sum_{c=1}^{C} \big[\, p_c \log {q_c} + (1 - p_c)\log(1 - q_c) \,\big]
$$

- $q$ is an independent sigmoid for each class.

# From logits

- We get probabilities from logits by taking the softmax. Hence the order of the expression becomes: logits → softmax → log → sum.
  - But it is not numerically stable to implement this way.
- Like the [numerically stable softmax]({% post_url 2026-09-22-activation-func %}), a numerically stable log-softmax is obtained by subtracting the maximum logit.
- The log-softmax of a logit $z_t$ where $p_t = 1$ is

  $$
  z_t - m - \log \sum_c^C e^{z_c - m}
  $$

  - $C$ is the number of classes.
  - $m = \max_c z_c$.
  - The sum iterates over all the logits $\mathbf{z}$.

- The negative log likelihood comes to

  $$
  L = \log \sum_c^C e^{z_c - m} - (z_t - m) \tag{2}
  $$

  - The batch and token averaging need to be applied here as well.

# Shape discussion

- Logits $Z = (B, L, V)$ for batch, sequence length and vocab size in the case of an LLM.
- Reference $Y = (B, L)$ where $Y_i \in \{1, \dots, V\}$ and one-hot.
- The loss is calculated for each $Z_{i,j}$ and averaged.
  - $m$ is the maximum value of the logits over the category dim. Hence $m$ has shape $(B, L, 1)$.
  - $z_i$ is $(B, L, 1)$.
  - $\log \sum \dots$ is $(B, L, 1)$.
- Taking the average across the entire sequence $L$: $(B, 1)$.
- Taking the average across the entire batch $B$: $(1, 1)$.

# Implementation note

- Padding should be handled by ignoring those logits. Those logits should not contribute anywhere in $(2)$.
