---
layout: post
title: Cross Entropy (and numerical stability)
description: Note on different ML topic
tags: mlnote
date: 2026-09-30
featured: true
giscus_comments: false
---

# Notation

Everything below — prose, equations and code — uses one set of symbols.

- $N$ is the batch size, $C$ the number of classes (the vocab size for an LLM), $T$ the sequence length.
- $n$, $c$, $t$ are the corresponding indices.
- $L$ is always the loss, never a length.
- $Z$ is the logits tensor, shaped $(N, C, T)$ — class in dim 1, the layout `torch.nn.functional.cross_entropy` expects. A single logit is $z_c$, or $z_{n,c,t}$ when the other indices matter.
- $Y$ is the target tensor, shaped $(N, T)$.
- $p$ is the reference distribution, $q$ the hypothesis distribution, $y$ the correct class index.

# Intuition

- Maximizing the likelihood is equivalent to minimizing the negative of the likelihood.
- If we have a possible outcome of a Bernoulli event $y$ and the probability $q \in [0, 1]$, the **likelihood** is the product over the $N$ independent events in the batch:

  $$
  \prod_{n=1}^{N} q_n^{y_n} (1-q_n)^{1-y_n}
  $$

- Intuitively, the cross entropy loss measures how confident the model was and punishes accordingly.
  - If the event happened, $y=1$, and the model predicted $q = 0.9$, it was confident and correct.
  - For $y=0$ and the model predicted $q = 0.9$, it was confident and incorrect.

# The cross entropy

$$
L = -\sum_{c=1}^{C} p_c \log q_c \tag{1}
$$

- $C$ is the number of classes.
- $p$ is the reference distribution, $q$ is the hypothesis distribution.
- Optimization techniques work with minimization problems — therefore, we negate the likelihood.
- Only where $p_c = 1$ contributes to the loss. This removes the effect of $C-1$ logits from the sum, but their effect stays in the gradient (see $(2)$ below).
- Since $p$ does not depend on the model parameters, and when $p_c = 0$ the multiplication is 0 as well, minimizing $(1)$ becomes minimizing the *negative log likelihood*:

  $$
  L = - \log q_y
  $$

  only where $p_y = 1$.

- For a sequence of length $T$, average them:

  $$
  L = - \frac{1}{T} \sum_{t=1}^{T} \log q_{t, y}
  $$

- For a batch of $N$:

  $$
  L = - \frac{1}{N} \frac{1}{T} \sum_{n=1}^{N} \sum_{t=1}^{T} \log q_{n, t, y}
  $$

  - This is when the sequence length is constant within a batch.
  - Simply normalizing by the total sequence length works when it is not constant:

  $$
  L = -\frac{1}{\sum_{n=1}^{N} T_n } \sum_{n=1}^{N} \sum_{t=1}^{T_n} \log q_{n, t, y}
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
- The log-softmax of the correct-class logit $z_y$ — that is, the class $y$ where $p_y = 1$ — is

  $$
  z_y - m - \log \sum_{c=1}^{C} e^{z_c - m}
  $$

  - $C$ is the number of classes.
  - $m = \max_c z_c$.
  - The sum iterates over all the logits $\mathbf{z}$ of that position.

- The negative log likelihood comes to

  $$
  L = \log \sum_{c=1}^{C} e^{z_c - m} - (z_y - m) \tag{2}
  $$

  - Indexing the correct class by $y$ and not by $t$ keeps $t$ free to mean the sequence position, which is what the `dim` arguments in the code assume.
  - The batch and token averaging need to be applied here as well.

# Label smoothing

- Minimizing the cross entropy is maximizing the log-likelihood of the correct label if the labels are one-hot (only one correct label among $C$ classes) {% cite szegedy_rethinking_2016 --file refs %}
  - This may result in over-fitting if the model tries to assign full probability to the correct label.
  - Consequently, it leads to less generalization and adaptability within the model.
- Instead of a one-hot label distribution, the **reference** distribution is modified so that the incorrect labels also have a tiny probability:

  $$
  p^{\prime}_c = (1 - \epsilon)\,\delta_{c,y} + \epsilon\, u(c)
  $$

  - $p^{\prime}_c$ is the new label distribution after smoothing.
  - $\delta_{c,y}$ is 1 at the correct class and 0 elsewhere.
  - $\epsilon$ is the smoothing factor.
  - $u(c)$ is a fixed distribution, usually uniform, i.e. all classes have $1/C$ probability.
  - The paper writes this distribution $q^{\prime}(k \mid x)$. Here $q$ is reserved for the model's output, so the smoothed *label* distribution is $p^{\prime}$ and the class index is $c$.

- The loss isn't simply the negative log-likelihood anymore; it looks a bit like the multi-label multi-class case.
- Implementation wise:
  - `nll` ← the negative log-likelihood as in $(2)$ for the correct class's logit $z_y$.
  - `smooth` ← the negative log-likelihood as in $(2)$ for all $C$ logits.
  - Take the `mean` of `smooth` (this comes from $u(c)$ being uniform. We're also supposed to take the mean of `nll`, but because it is one scalar, it is ignored).
  - $L = (1 - \epsilon) \times \text{nll} + \epsilon \times \text{smooth}$

# Shape discussion

- Logits $Z$ is $(N, C, T)$ — batch, classes, sequence. **The class axis is dim 1**, so every reduction below is along `dim=1`.
  - LLM logits usually come out as $(N, T, C)$ with the vocab last. Transpose with `.transpose(1, 2)` before using them here; `F.cross_entropy` wants the same layout.
- Reference $Y$ is $(N, T)$ where $Y_{n,t} \in \{0, \dots, C-1\}$ is a zero-based **class index**, not a one-hot row. The one-hot row is the conceptual $p$ of $(1)$; storing the index is just the compact form of it, and it is what `gather` consumes.
- The loss is computed per position $(n, t)$, then averaged. Along the way:
  - $m = \max_c z_c$ is $(N, 1, T)$.
  - $z = Z - m$ is $(N, C, T)$.
  - $\log \sum_{c} e^{z_c}$ is $(N, 1, T)$.
  - $z_y$, gathered at the target index, is $(N, 1, T)$.
  - the per-token loss of $(2)$ is $(N, T)$.
- Taking the average across the sequence: $(N,)$.
- Taking the average across the batch: a scalar.

# Implementation

Padding is handled by ignoring those positions: where $Y_{n,t}$ equals `ignore_index`, the position contributes nothing to $(2)$ and nothing to the denominator of the average.

```python
def cross_entropy(input: Float[Tensor, "N C T"], target: Int[Tensor, "N T"], ignore_index=-100, reduction='mean', label_smoothing=0.0) -> Float[Tensor, "..."]:
    """
    N = batch
    C = classes
    T = sequence
    """
    # mask is true where the target is valid
    # (N, T)
    mask = target != ignore_index
    # so that `gather()` works
    safe_target = target.masked_fill(~mask, 0)  # (N, T)

    # shift the maximum value from each logit
    # for numerical stability
    # m, reduced over the class axis
    shift = torch.amax(input, dim=1, keepdim=True)  # (N, 1, T)

    # shifted stable logits
    z = input - shift

    # calculate the logsumexp term
    exp = torch.exp(z)  # (N, C, T)
    sumexp = exp.sum(dim=1, keepdim=True)  # (N, 1, T)
    logsumexp = torch.log(sumexp)  # (N, 1, T)

    # z_y, the correct class's logit
    zy = torch.gather(input=z, dim=1, index=safe_target.unsqueeze(1))  # (N, 1, T)
    # eq. (2), per position
    nll = (logsumexp - zy).squeeze(1)  # (N, T)

    if label_smoothing > 0.0:
        # eq. (2) for every class, then the mean from the uniform u(c)
        smooth = (logsumexp - z).mean(dim=1, keepdim=False)  # (N, C, T).mean(1) => (N, T)
        nll = (1 - label_smoothing) * nll + (label_smoothing * smooth)  # (N, T)

    # ignores index
    out = nll * mask

    if reduction == 'mean':
        out = out.sum() / mask.sum()
    elif reduction == 'sum':
        out = out.sum()
    elif reduction != 'none':
        raise Exception('valid reductions are mean, sum, none')

    return out
```

### References

<div class="publications">
{% bibliography --cited --file refs %}
</div>
