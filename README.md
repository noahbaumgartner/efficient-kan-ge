# An Efficient Implementation of Kolmogorov-Arnold Network

This repository contains an efficient implementation of Kolmogorov-Arnold Network (KAN).
The original implementation of KAN is available [here](https://github.com/KindXiaoming/pykan).

The performance issue of the original implementation is mostly because it needs to expand all intermediate variables to perform the different activation functions.
For a layer with `in_features` input and `out_features` output, the original implementation needs to expand the input to a tensor with shape `(batch_size, out_features, in_features)` to perform the activation functions.
However, all activation functions are linear combination of a fixed set of basis functions which are B-splines; given that, we can reformulate the computation as activate the input with different basis functions and then combine them linearly.
This reformulation can significantly reduce the memory cost and make the computation a straightforward matrix multiplication, and works with both forward and backward pass naturally.

The problem is in the **sparsification** which is claimed to be critical to KAN's interpretability.
The authors proposed a L1 regularization defined on the input samples, which requires non-linear operations on the `(batch_size, out_features, in_features)` tensor, and is thus not compatible with the reformulation.
I instead replace the L1 regularization with a L1 regularization on the weights, which is more common in neural networks and is compatible with the reformulation.
The author's implementation indeed include this kind of regularization alongside the one described in the paper as well, so I think it might help.
More experiments are needed to verify this; but at least the original approach is infeasible if efficiency is wanted.

Another difference is that, beside the learnable activation functions (B-splines), the original implementation also includes a learnable scale on each activation function.
I provided an option `enable_standalone_scale_spline` that defaults to `True` to include this feature; disable it will make the model more efficient, but potentially hurts results.
It needs more experiments.

2024-05-04 Update: @xiaol hinted that the constant initialization of `base_weight` parameters can be a problem on MNIST.
For now I've changed both the `base_weight` and `spline_scaler` matrices to be initialized with `kaiming_uniform_`, following `nn.Linear`'s initialization.
It seems to work much much better on MNIST (~20% to ~97%), but I'm not sure if it's a good idea in general.

## Grid extension

`KANLinear`/`KAN` already support `update_grid`, which redistributes the existing grid points to better match the input distribution, but keeps `grid_size` fixed.

This fork adds `extend_grid(x, new_grid_size, margin=0.01)`, which implements the "grid extension" technique from the original KAN paper: it changes `grid_size` itself (to a finer or coarser grid) while refitting the spline coefficients (via least squares, like `curve2coeff`) so the function represented by the layer is preserved. This lets you start training on a coarse, cheap grid and later refine it to a finer grid without losing what was learned, e.g.:

```python
model = KAN([2, 4, 1], grid_size=5)
# ... train for a while ...
model.extend_grid(x_sample, new_grid_size=10)
# ... continue training with a finer grid ...
```

`x_sample` should be a representative batch of inputs to the model (`(batch_size, in_features)`); it's used both to evaluate the current spline function and to fit the new one, and is propagated layer by layer so every layer is refit against the inputs it actually sees.
