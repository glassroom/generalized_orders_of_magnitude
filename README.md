# generalized_orders_of_magnitude

Reference implementation of generalized orders of magnitude (GOOMs), for PyTorch, enabling you to operate on real numbers outside the bounds representable by torch.float32 and torch.float64. Toy example:

```python
import torch
import generalized_orders_of_magnitude as goom

DEVICE = 'cuda'                                                    # change as needed
goom.config.float_dtype = torch.float64                            # real and imag dtype
mm, lmme = (torch.matmul, goom.log_matmul_exp)                     # for legibility

x = torch.randn(3, 3, dtype=torch.float64, device=DEVICE) * 1e128  # large magnitudes
y = torch.linalg.inv(x)                                            # inverts mult by x
z = mm(mm(mm(mm(x, x), x), y), y)                                  # z should equal x
print('Computes over float64?', torch.allclose(x, z))              # computation fails!

log_x = goom.log(x)                                                # map x to a GOOM
log_y = goom.log(y)                                                # map y to a GOOM
log_z = lmme(lmme(lmme(lmme(log_x, log_x), log_x), log_y), log_y)  # z should equal x
print('Computes over GOOMs?', torch.allclose(x, goom.exp(log_z)))  # computation succeeds!
```

For important limitations of this initial implementation, both in terms of precision and performance, see [here](#limitations). For instructions to replicate published results, see [here](#replicating-published-results). For an implementation of selective resetting, see [here](#selective-resetting).


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

### Mapping Real Tensors to GOOMs and Back

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

### Matrix Multiplication over GOOMs

The following snippet of code executes the same matrix multiplication over real numbers and over GOOMs:

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

### Chains of Matrix Products over GOOMs 

Note: To be able to run the code below, you must first install [`torch_parallel_scan`](https://github.com/glassroom/torch_parallel_scan/).

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

We have implemented a variety of functions over GOOMs. All function are defined in [generalized_orders_of_magnitude.py](generalized_orders_of_magnitude/generalized_orders_of_magnitude.py). To see a list of them, run the following on a Python command line:

```python
import generalized_orders_of_magnitude as goom
print('List of implemented functions:', *[
    name for name in dir(goom)
    if (not name.startswith('_')) and
    (name not in ['dataclass', 'math', 'torch', 'Config', 'config'])
], sep='\n')
```

To see the docstring and source code of any implemented function, type its name followed by "??" and <Enter> on a Python command line, as usual.


## Configuration Options

Our library has three configuration options, set to sensible defaults:

* `goom.config.keep_logs_finite` (boolean): If True, `goom.log()` always returns finite values. The finite value returned for any input element numerically equal to zero will numerically exponentiate to zero in the specified float dtype. If False, `goom.log()` returns `float("-inf")` values for inputs numerically equal to zero. Default: True. 

* `goom.config.cast_all_logs_to_complex` (boolean): If True, `goom.log()` always returns complex-typed tensors. If False, `goom.log()` returns float tensors whenever all real input elements are equal to or greater than zero. Setting this option to False can improve performance and reduce memory use when working with real values that are always non-negative, such as measures and probabilities. Default: True.

* `goom.config.float_dtype` (torch.dtype): Float dtype of real and imaginary components of complex GOOMs, and of real GOOMs. Default: torch.float32, _i.e._, complex-typed GOOMs are represented by default as torch.complex64 tensors with torch.float32 real and imaginary components. For greater precision, set `goom.config.float_dtype = torch.float64`. Note: We have tested this configuration option only with torch.float32 and torch.float64.

To view the current configuration, use:

```python
print(goom.config)
```


## Replicating Published Results

In our paper, we present the results of three representative experiments: (1) compounding up to one million real matrix products beyond standard float limits; (2) estimating spectra of Lyapunov exponents in parallel, using a novel selective-resetting method to prevent state colinearity; and (3) training deep recurrent neural networks that maintain long-range dependencies without numerical degradation, allowing recurrent state elements to fluctuate freely over time steps:

### Chains of Matrix Products that Compound Magnitudes beyond Float Limits

The code below will attempt to compute chains of up to 1M products of real matrices, each with elements independently sampled from a normal distribution, over torch.float32, torch.float64, and complex64 GOOMs (_i.e._, with torch.float32 real and imaginary components). For every matrix size, for each data type, the code will attempt to compute the entire chain 30 times. WARNING: The code below will take a LONG time to execute, because all product chains finish successfully with GOOMs.

```python
import torch
import generalized_orders_of_magnitude as goom
from tqdm import tqdm  # you must install from https://github.com/tqdm/tqdm

goom.config.keep_logs_finite = True
goom.config.float_dtype = torch.float32

DEVICE = 'cuda'
n_runs = 30                                    # number of runs per matrix size
d_list = [8, 16, 32, 64, 128, 256, 512, 1024]  # square matrix sizes
n_steps = 1_000_000                            # maximum chain length

longest_chains = []
for dtype in [torch.float32, torch.float64]:
    for run_number in range(n_runs):
        for d in d_list:
            state = torch.randn(d, d, dtype=dtype, device=DEVICE)
            for t in tqdm(range(n_steps), desc=f'Chain over {dtype}, run {run_number}, matrix size {d}'):
                update = torch.randn(d, d, dtype=dtype, device=DEVICE)
                state = torch.matmul(state, update)
                if not state.isfinite().all().item():
                    break
            longest_chains.append({
                'method': 'MatMul_over_R', 'dtype_name': str(dtype),
                'run_number': run_number, 'd': d, 'n_completed': t + 1,
            })

for run_number in range(n_runs):
    for d in d_list:
        log_state = goom.log(torch.randn(d, d, dtype=torch.float32, device=DEVICE))
        for t in tqdm(range(n_steps), desc=f'Chain over Complex64 GOOMs, run {run_number}, matrix size {d}'):
            log_update = goom.log(torch.randn(d, d, dtype=torch.float32, device=DEVICE))
            log_state = goom.log_matmul_exp(log_state, log_update)
            if not log_state.isfinite().all().item():
                break
        longest_chains.append({
            'method': 'LogMatMulExp_over_GOOMs', 'dtype_name': 'torch.complex64',
            'run_number': run_number, 'd': d, 'n_completed': t + 1,
        })

torch.save(longest_chains, 'longest_chains.pt')  # load with torch.load('longest_chains.pt')
print(*longest_chains, sep='\n')
```

### Parallel Estimation of the Spectrum of Lyapunov Exponents over GOOMs

The code for estimating spectra of Lyapunov exponents in parallel, via a prefix scan over GOOMs, incorporating our selective-resetting method, is at:

[https://github.com/glassroom/parallel_lyapunov_exponents](https://github.com/glassroom/parallel_lyapunov_exponents)


### Deep RNN Modeling Sequences with Non-Diagonal SSMs over GOOMs

The code implementing deep recurrent neural networks that capture long-range dependencies over GOOMs, allowing recurrent state elements to fluctuate freely over time steps, is at:

[https://github.com/glassroom/goom_ssm_rnn](https://github.com/glassroom/goom_ssm_rnn)


## Limitations

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

Our library provides a broadcastable implementation of LMME as a composition of vmapped vector operations, as `goom.alternate_log_matmul_exp`. It is _much slower_ than the current default implementation of LMME, especially for larger matrices, but offers greater precision. In practice, we have found it unnecessary to use it, because the current default implementation, despite being sub-optimal, has proven to work well in diverse experiments. Please see our source code for details.


## Selective Resetting

In our paper, we formulate a method for selectively resetting interim states at any step in a linear recurrence, as we compute all states in the linear recurrence in parallel via a prefix scan. We apply this method as a component of our parallel algorithm for estimating the spectrum of Lyapunov exponents, over GOOMs. If you are interested in understanding how our selective-resetting method works, we recommend taking a look at [https://github.com/glassroom/selective_resetting/](https://github.com/glassroom/selective_resetting/), an implementation of selective resetting over real numbers instead of GOOMs. We also recommend reading Appendix C of our paper, which explains the intuition behind selective resetting informally, with step-by-step examples.


## Citing

TODO: Update citation.

```
@misc{heinsenkozachkov2025gooms,
    title={
        Generalized Orders of Magnitude for
        Scalable, Parallel, High-Dynamic-Range Computation},
    author={Franz A. Heinsen, Leo Kozachkov},
    year={2025},
}
```

## Notes

The work here originated with casual conversations over email between us, the authors, in which we wondered if it might be possible to find a succinct expression for computing non-diagonal linear recurrences in parallel, by mapping them to the complex plane. Our casual conversations gradually evolved into the development of generalized orders of magnitude, along with an algorithm for estimating Lyapunov exponents in parallel, and a novel method for selectively resetting interim states in a parallel prefix scan.

We hope others find our work and our code useful.

