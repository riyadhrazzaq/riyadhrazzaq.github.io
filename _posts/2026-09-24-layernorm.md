---
layout: post
title: Layer Norm
description: Note on different ML topic
tags: mlnote
date: 2026-09-24
featured: true
giscus_comments: false
---

# Why Normalize a Neuron?

- Normalization is a way to minimize training time.
- Pre-activation is also known as summed input.
- "A feed-forward neural network is a non-linear mapping from a input pattern $x$ to an output vector $y$" {% cite ba_layer_2016 --file refs %}
- **Covariate shift**. "One of the challenges of deep learning is that the *gradients* with respect to the *weights* in one layer are highly dependent on the outputs of the neurons in the previous layer especially if these outputs change in a highly correlated way." {% cite ba_layer_2016 --file refs %}
- **Batch normalization** {% cite ioffe_batch_2015 --file refs %} rescales the summed inputs (aka, logits) to the activation function according to their variances under the distribution of the data.
  - This reduces such undesirable "covariate shift".
- "However, the summed inputs to the recurrent neurons in a recurrent neural network (RNN) often vary with the length of the sequence so applying batch normalization to RNNs appears to require different statistics for different time-steps." {% cite ba_layer_2016 --file refs %}

## Layer Normalization

- Changes in the output of one layer will tend to cause highly correlated changes in the summed inputs to the next layer, especially with ReLU units whose outputs can change by a lot.
- This suggests the "covariate shift" problem can be reduced by fixing the mean and the variance of the summed inputs within each layer {% cite ba_layer_2016 --file refs %}.

  $$
  \mu^l = \frac{1}{H} \sum_{i=1}^{H} a_i^l \qquad \sigma^l = \sqrt{\frac{1}{H} \sum_{i=1}^{H} \left( a_i^l - \mu^l \right)^2}
  $$

- The normalization statistics mean $\mu$ and standard deviation $\sigma$ for a layer $l$ are computed within the layer itself, from its own neurons. Only $g$ and $b$ are learned.
- $a^l = w_{i}^{l} h^l$ is the logits, $H$ is the number of neurons in a layer.
- The update rule is,

  $$
  \frac{g^l}{\sigma^l} \times (a^l - \mu^l) + b^l
  $$

  where $g$ is the gain parameter to rescale the normalized summed input if the network wants to, and $b$ is the bias.

## The shape of the tensors

- $N = \text{Batch Size}$
- $H = \text{Hidden size after the } X \cdot W^T$
- Hence, the summed input is $(N, H)$
- For layer $l$,
  - $\mu$ is $(N, 1)$
  - $\sigma$ is $(N, 1)$
  - shift is $\frac{(N, H) - (N, 1)}{(N, 1)} = (N, H)$
  - gain multiplies and bias adds, both $(1, H)$ broadcast over $(N, H)$: $(1, H) \times (N, H) + (1, H) = (N, H)$
- **For each sample, we have different statistics.**

## Implementation

- Implementation challenges,
  - The denominator of the shift requires to be always positive.
  - The square-root blows up during backpropagation at origin.
  - Solve: add an element to avoid being 0, $\epsilon$, usually $10^{-5}$, to the denominator before the square-root (aka, the variance).
- In `torch`, the module takes the `input` and a declared `normalized_shape`. The input is reduced from the left and normalizes the dims which match from the right. For example, if the input is $(a,b,c,d)$ and the `normalized_shape` is $(c,d)$, the statistics will be of shape $(a,b)$ and it normalizes over the **first matching dims from the right**, $(c,d)$.

```python
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | Sequence[int], eps: float = 1e-5,
                 elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape: tuple[int, ...] = tuple(normalized_shape)
        self.eps: float = eps
        # Reduce over the trailing dims that normalized_shape names: (-n, ..., -1).
        self.dims: tuple[int, ...] = tuple(range(-len(self.normalized_shape), 0))

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    @override
    @typed
    def forward(self, input: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        # unbiased=False is the /N variance LayerNorm uses, matching F.layer_norm.
        var, mean = torch.var_mean(input, dim=self.dims, unbiased=False, keepdim=True)
        out = (input - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            out = out * self.weight + self.bias
        return out

    def extra_repr(self) -> str:
        return (f"{self.normalized_shape}, eps={self.eps}, "
                f"elementwise_affine={self.weight is not None}")
```

### References
<div class="publications">
{% bibliography --cited --file refs %}
</div>
