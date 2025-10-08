# generalized_orders_of_magnitude

Reference implementation of [generalized orders of magnitude](https://openreview.net/pdf?id=SUuzb0SOGu) (GOOMs), for PyTorch, enabling you to operate on real numbers _far_ beyond the limits of conventional floating-point formats. You no longer need to scale, clip, or stabilize values with cumbersome techniques to keep magnitudes within those limits.

Toy example:

```python
import torch
import generalized_orders_of_magnitude as goom
import torch_parallel_scan as tps  # https://github.com/glassroom/torch_parallel_scan

DEVICE = 'cuda'  # change as needed

goom.config.keep_logs_finite = True          # log(0) will return a finite floor
goom.config.cast_all_logs_to_complex = True  # all GOOMs will be cast to a complex dtype
goom.config.float_dtype = torch.float32      # dtype of real and imaginary components

mats = torch.randn(256, 1024, 1024, device=DEVICE)  # chain of square matrices

# Multiply all matrices in the chain with a parallel scan over float tensors:
prod = tps.reduce_scan(mats, torch.matmul, dim=-3)
print('Computes over float tensors?', prod.isfinite().all().item())          # failure!

# Multiply the same matrices with a parallel scan over complex-typed GOOMs:
log_prod = tps.reduce_scan(goom.log(mats), goom.log_matmul_exp, dim=-3)
print('Computes over complex GOOMs?', log_prod.isfinite().all().item())      # success!
```

GOOMs generalize the concept of "order of magnitude" to incorporate complex numbers that exponentiate to real ones. As with ordinary orders of magnitude, GOOMs are more stable than the real numbers to which they would exponentiate, enabling _effortless scaling and parallelization of high-dynamic-range computations_.

We provide two implementations, Complex64 and Complex128 GOOMs, with _more than 10^37 and 10^307 decimal digits of normal dynamic range on each side of the decimal point_, respectively. For comparison, Float32 and Float64 provide 38 and 308 decimal digits of normal dynamic range on each side, respectively.

Comparing Complex64 GOOMs to Float32 and Complex128 GOOMs to Float64 on CUDA devices, over common representable magnitudes, precision is competitive (the same or within a fraction of the least significant decimal digit), while execution time and memory use typically double (with some variation).

For a detailed comparison to Float32 and Float64 on CUDA devices, see [here](#comparison-to-float32-and-float64-on-cuda-devices).


## Installing

```
pip install git+https://github.com/glassroom/generalized_orders_of_magnitude
```

Alternatively, you can download a single file to your project directory: [generalized_orders_of_magnitude.py](generalized_orders_of_magnitude/generalized_orders_of_magnitude.py).

The only dependency is a recent version of PyTorch.


## Using

Import the library with:

```python
import generalized_orders_of_magnitude as goom
```


### Mapping Real Tensors to Complex-Typed GOOMs and Back

`goom.log()` maps real tensors to complex-typed GOOMs, and `goom.exp()` maps them back to real tensors:

```python
import torch
import generalized_orders_of_magnitude as goom

DEVICE = 'cuda'  # change as needed
goom.config.float_dtype = torch.float32

# Create a float-typed real tensor:
x = torch.randn(5, 3, device=DEVICE)
print('x:\n{}\n'.format(x))

# Map it to a complex-typed GOOM tensor:
log_x = goom.log(x)
print('log_x:\n{}\n'.format(log_x))

# Map it back to a float-typed real tensor:
print('exp(log_x):\n{}\n'.format(goom.exp(log_x)))
```


### Matrix Multiplication over Complex-Typed GOOMs

`goom.log_matmul_exp()` computes the equivalent of a real-valued matrix multiplication over complex-typed GOOMs. For example, the following snippet of code executes the same matrix multiplication over real numbers and over complex-typed GOOMs:

```python
import torch
import generalized_orders_of_magnitude as goom

DEVICE = 'cuda'  # change as needed
goom.config.float_dtype = torch.float32

x = torch.randn(5, 4, device=DEVICE)
y = torch.randn(4, 3, device=DEVICE)
z = torch.matmul(x, y)
print('z:\n{}\n'.format(z))

log_x = goom.log(x)
log_y = goom.log(y)
log_z = goom.log_matmul_exp(log_x, log_y)
print('exp(log_z):\n{}\n'.format(goom.exp(log_z)))
```


### Chains of Matrix Products over Complex-Typed GOOMs via Parallel Scans

You can apply `goom.log_matmul_exp()` via a parallel scan to compute chains of matrix products. Here is a toy example
(note: to run the code below, you must first install [`torch_parallel_scan`](https://github.com/glassroom/torch_parallel_scan/)):

```python
import torch
import generalized_orders_of_magnitude as goom
import torch_parallel_scan as tps  # you must install

DEVICE = 'cuda'  # change as needed
goom.config.float_dtype = torch.float32

# A chain of matrix products:
n, d = (5, 4)
x = torch.randn(n, d, d, device=DEVICE) / (d ** 0.5)
y = tps.reduce_scan(x, torch.matmul, dim=0)
print('y:\n{}\n'.format(y))

# The same chain, executed over GOOMs:
log_x = goom.log(x)
log_y = tps.reduce_scan(log_x, goom.log_matmul_exp, dim=0)
print('exp(log_y):\n{}\n'.format(goom.exp(log_y)))
```


### Other Functions over GOOMs:

We have implemented a variety of functions over complex-typed GOOMs. All function are defined in [generalized_orders_of_magnitude.py](generalized_orders_of_magnitude/generalized_orders_of_magnitude.py). To see a list of them, run the following on a Python command line:

```python
import generalized_orders_of_magnitude as goom
print('List of implemented functions:', *[
    name for name in dir(goom)
    if (not name.startswith('_')) and
    (name not in ['dataclass', 'math', 'torch', 'Config', 'config'])
], sep='\n')
```

To see the docstring and source code of any implemented function, type its name followed by "??" and `Enter` on a Python command line, as usual.


## Configuration Options

Our library has three configuration options, set to sensible defaults:

* `goom.config.keep_logs_finite` (boolean): If True, `goom.log()` always returns finite values. The finite value returned for any input element numerically equal to zero will numerically exponentiate to zero in the specified float dtype. If False, `goom.log()` returns `float("-inf")` values for inputs numerically equal to zero. Default: True. 

* `goom.config.cast_all_logs_to_complex` (boolean): If True, `goom.log()` always returns complex-typed tensors. If False, `goom.log()` returns float tensors whenever all real input elements are equal to or greater than zero. Setting this option to False can improve performance and reduce memory use when working with real values that are always non-negative, such as measures and probabilities. Default: True.

* `goom.config.float_dtype` (torch.dtype): Float dtype of real and imaginary components of complex GOOMs, and of real GOOMs. Default: torch.float32, _i.e._, complex-typed GOOMs are represented by default as torch.complex64 tensors with torch.float32 real and imaginary components. For greater precision, set `goom.config.float_dtype = torch.float64`. Note: We have tested this configuration option only with torch.float32 and torch.float64.

To view the current configuration, use:

```python
print(goom.config)
```


## Using as a Component of PyTorch Models

You can use all functions provided by our library as components of PyTorch models, trainable via stochastic gradient descent (SGD) with conventional tools and techniques, _without hassle_. All functions are parallelized, broadcastable over an arbitrary numbers of preceding indices, and compatible with backpropagation of gradients. We have taken special care to handle the singularity at the logarithm of zero gracefully, for use in a broad range of applications, including deep learning.

It's even possible to implement models that operate entirely over GOOMs, end-to-end, but in most cases we would not recommend it, due to the increased compute and memory cost. Instead, we would normally recommend implementing over GOOMs only those computations for which the dynamic range of Float32 and Float64 proves insufficient. Before mapping GOOMs to floats, via `goom.exp()`, you must scale the GOOMs to values that are representable as floats. For convenience, we provide two functions for scaling GOOMs: `goom.scale()` and `goom.scaled_exp()`. See their respective docstrings for usage.


## Replicating Published Results

In our paper, we present the results of three representative experiments: (1) compounding up to one million real matrix products _far_ beyond standard float limits; (2) estimating spectra of Lyapunov exponents in parallel _orders-of-magnitude faster than with previous methods_, using a novel selective-resetting method to prevent state colinearity; and (3) training deep recurrent neural networks that capture long-range dependencies over _non-diagonal recurrent states, computed in parallel via a prefix scan, without requiring any form of stabilization_:


### Chains of Matrix Products that Compound Magnitudes Far Beyond Float Limits

This repository provides a Python script that attempts to compute chains of up to 1M products of real matrices, each with elements independently sampled from a normal distribution, over torch.float32, torch.float64, and complex64 GOOMs, for matrix sizes ranging from `8 x 8` to `1024 x 1024`, on CUDA devices. For each data type, for each matrix size, the script will attempt to compute the entire chain 30 times. To run the script, clone _and install_ this repository, install `torch`, `numpy`, `pandas`, `matplotlib`, and `tqdm`, and execute from the command line:

```
python replicate_experiment_one_from_paper.py
```

The script will create two files: `'longest_chains.pt'`, a PyTorch file containing data for all runs, which you can load from a Python shell or notebook with `torch.load('longest_chains.pt')`, and `'fig_longest_chains.png'`, an image file containing a summary plot. WARNING: The script will take a LONG time to execute, because all product chains finish successfully with GOOMs.


### Parallel Estimation of the Spectrum of Lyapunov Exponents

The code for estimating spectra of Lyapunov exponents in parallel, via a prefix scan over GOOMs, incorporating our selective-resetting method, is at:

[https://github.com/glassroom/parallel_lyapunov_exponents](https://github.com/glassroom/parallel_lyapunov_exponents)


### Deep RNNs that Apply Non-Diagonal SSMs in Parallel without Stabilization

The code implementing deep recurrent neural networks that capture long-range dependencies via non-diagonal recurrences over GOOMs, without requiring any form of stabilization, is at:

[https://github.com/glassroom/goom_ssm_rnn](https://github.com/glassroom/goom_ssm_rnn)


## Selective Resetting

In our paper, we formulate a method for selectively resetting interim states at any step in a linear recurrence, as we compute all states in the linear recurrence in parallel via a prefix scan. We apply this method as a component of our parallel algorithm for estimating the spectrum of Lyapunov exponents, over GOOMs. If you are interested in understanding how our selective-resetting method works, we recommend taking a look at [https://github.com/glassroom/selective_resetting/](https://github.com/glassroom/selective_resetting/), an implementation of selective resetting over floats instead of complex-typed GOOMs. We also recommend reading Appendix C of our paper, which explains the intuition behind selective resetting informally, with step-by-step examples.


## Comparison to Float32 and Float64 on CUDA Devices


### Dynamic Range Compared to Float32 and Float64

| Representation  | Bits | Smallest Normal Magnitude      | Largest Normal Magnitude     |
| --------------- | ---- | ------------------------------ | ---------------------------- |
| Float32         | 32   | 10^-38                         | 10^38                        |
| Float64         | 64   | 10^-308                        | 10^308                       |
| Complex64 GOOM  | 64   | 10^(-10^37.64) ≈ exp(-10^38)   | 10^(10^37.64) ≈ exp(10^38)   |
| Complex128 GOOM | 128  | 10^(-10^307.64) ≈ exp(-10^308) | 10^(10^307.64) ≈ exp(10^308) |

Note: Our implementation of GOOMs is meant to be _complementary_ to conventional numerical formats, _not a replacement_ for them. We recommend using it only when their dynamic range falls short.


### Precision, Execution Time, and Memory Use Compared to Float32 and Float64

We provide a Python script for comparing the precision, execution time, and memory use of Complex64 GOOMs to Float32 and Complex128 GOOMs to Float64 on CUDA devices. To run the script, clone _and install_ this repository, install `torch`, `numpy`, `pandas`, and `matplotlib`, and execute from the command line:

```
python compare_gooms_to_floats.py
```

We find that for both Complex64 GOOMs vs. Float32 and Complex128 GOOMs vs. Float64, precision over common representable magnitudes is competitive (the same or within a fraction of the least significant decimal digit), while execution time and memory use typically double (with some variation).

Note: The comparisons are valid only for this implementation, not for GOOMs in general.


## Relationship to Floating-Point Formats

In our paper, we define GOOMs as a set of mathematical objects, and show that floating-point numbers are a special case of GOOMs. All conventional and extended floating-point formats, including Float32 and Float64, are GOOMs that represent imaginary components with a single bit.

Our implementation is a special case too, but one that represents real and imaginary components with either Float32 or Float64 numbers, which, as we just mentioned, are themselves special cases of GOOMs, in effect forming an "edifice of GOOMs." We have implemented this "edifice of GOOMs" by extending PyTorch's complex data types.

Defining and naming GOOMs enables us to talk and reason about all possible special cases, including floating-point numbers, in the abstract. From a practical standpoint, GOOMs are complementary to existing numerical formats, providing a mechanism that can leverage them, enabling you to operate over a far greater dynamic range of real numbers than previously possible.


## Limitations


### Limitations of Our Initial Implementation of Log-MatMul-Exp

Our initial implementation of `goom.log_matmul_exp` (LMME) is sub-optimal, both in terms of precision and performance. Ideally, what we want is an implementation of LMME that delegates the bulk of parallel computation to a highly optimized kernel that executes and aggregates results over _tiled sub-tensors of complex dtype_. Unfortunately, PyTorch and its ecosystem, including intermediate compilers like Triton, currently provide no support for developing _complex-typed kernels_ (as of mid-2025). As we discuss in our paper, we considered implementing LMME so it computes all outer sums in parallel, then applies log-sum-exp, but decided not to do so, because it requires $\mathcal{O}(ndm)$ space for two matrices of size $n \times d$ and $d \times m$, respectively. We also considered applying log-sum-exp to the elementwise addition of each pair of vectors independently of the other pairs, with a vector-mapping operator like `torch.vmap`, but decided not to do so, because it runs into memory-bandwidth constraints on hardware accelerators like Nvidia GPUs, which are better suited for parallelizing computational kernels that execute and aggregate results over _tiled_ sub-tensors.

As a compromise, our initial implementation of LMME delegates the bulk of parallel computation to PyTorch's existing, highly optimized, low-level implementation of the dot-product over float tensors, limiting precision to that of scalar products in the specified floating-point format. In practice, we find that our initial implementation works well in diverse experiments, incurring execution times that are approximately twice as long as the underlying float matrix product on recent Nvidia GPUs. In our view, this is a reasonable initial tradeoff for applications that must be able to handle a greater dynamic range than is possible with torch.float32 and torch.float64.

The following snippet of code implements LMME with the two naive approaches discussed above (log-sum-exp of outer sums, vmapped vector operations), in case you want to experiment with them:

```python
import torch
import generalized_orders_of_magnitude as goom

DEVICE = 'cuda'  # change as needed

def naive_lmme_via_lse_of_outer_sums(log_x1, log_x2):
    """
    Naively implements log-matmul-exp as a log-sum-exp of outer sums, which
    is more precise than our initial implementation, but consumes O(ndm)
    space, where log_x1 has shape n x d and log_x2 has shape d x m.
    """
    outer_sums = log_x1.unsqueeze(-1) + log_x2.unsqueeze(-3)
    return goom.log_sum_exp(outer_sums, dim=-2)

def naive_lmme_via_vmapped_vector_ops(log_x1, log_x2):
    """
    Naively implements log-matmul-exp by processing each (row vec, col vec)
    pair independently, in parallel, with greater precision than our initial
    implementation, but less scalability due to memory-bandwidth issues.
    """
    _vve = lambda row_vec, col_vec: goom.exp(row_vec + col_vec).sum()  # vec, vec -> scalar
    _mve = torch.vmap(_vve, in_dims=(0, None), out_dims=0)             # mat, vec -> vec
    _mme = torch.vmap(_mve, in_dims=(None, 1), out_dims=1)             # mat, mat -> mat
    c1, c2 = (log_x1.real.detach().max(), log_x2.real.detach().max())  # scaling constants
    return goom.log(_mme(log_x1 - c1, log_x2 - c2)) + c1 + c2
    
log_x = goom.log(torch.randn(4, 3, device=DEVICE))
log_y = goom.log(torch.randn(3, 2, device=DEVICE))

print('All methods compute the same result:\n')
print(goom.log_matmul_exp(log_x, log_y), '\n')
print(naive_lmme_via_lse_of_outer_sums(log_x, log_y), '\n')
print(naive_lmme_via_vmapped_vector_ops(log_x, log_y), '\n')
```


#### Alternate Implementation of Log-MatMul-Exp

For applications that truly require more precision, we provide `goom.alternate_log_matmul_exp`, an alternate implementation of LMME that applies vmapped vector operations, broadcasting over any preceding dimensions. This alternate implementation is more precise but also _much slower_ than `goom.log_matmul_exp`, especially for larger input tensors. In our experiments, we have found it unnecessary to use `goom.alternate_log_matmul_exp`. Please see its docstring for usage details.


### Other Limitations

The current implementaton of `goom.log` is incompatible with `torch.vmap`, because the latter cannot operate over flow control code, such as the `if` statement evaluated inside `goom.log` to determine whether to cast all logarithms to complex tensors. The PyTorch team is working on [a solution to the flow-control issue](https://docs.pytorch.org/docs/stable//generated/torch.cond.html), but as of mid-2025 it is still a prototype, not recommended for use in applications. As a workaround, you can edit the code that implements `goom.log` to remove the `if` statement and always return a complex-typed tensor.


## Citing

```
@article{
heinsen2025generalized,
title={Generalized Orders of Magnitude for Scalable, Parallel, High-Dynamic-Range Computation},
author={Franz A. Heinsen and Leo Kozachkov},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=SUuzb0SOGu},
note={}
}
```


## Notes

The work here originated with casual conversations over email between us, the authors, in which we wondered if it might be possible to find a succinct expression for computing non-diagonal linear recurrences in parallel, by mapping them to the complex plane. Our casual conversations gradually evolved into the development of generalized orders of magnitude, along with an algorithm for estimating Lyapunov exponents in parallel, and a novel method for selectively resetting interim states in a parallel prefix scan.

We hope others find our work and our code useful.
