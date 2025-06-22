# generalized_orders_of_magnitude

Reference implementation of generalized orders of magnitude (GOOMs), for PyTorch, enabling you to operate on real numbers outside the bounds representable by torch.float32 and torch.float64. Toy example:

```python
import torch
import generalized_orders_of_magnitude as goom

DEVICE = 'cuda'                                                    # change as needed
goom.config.float_dtype = torch.float64                            # real and imag dtype
mm, lmme = (torch.matmul, goom.log_matmul_exp)                     # for easier legibility

x = torch.randn(3, 3, dtype=torch.float64, device=DEVICE) * 1e128  # large magnitudes
y = torch.linalg.inv(x)                                            # small magnitudes
z = mm(mm(mm(mm(x, x), x), y), y)                                  # z should equal x
print('Computes over float64?', torch.allclose(x, z))              # computation fails!

log_x = goom.log(x)                                                # map x to a GOOM
log_y = goom.log(y)                                                # map y to a GOOM
log_z = lmme(lmme(lmme(lmme(log_x, log_x), log_x), log_y), log_y)  # z should equal x
print('Computes over GOOMs?', torch.allclose(x, goom.exp(log_z)))  # computation succeeds!
```


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

All implemented functions are defined in [generalized_orders_of_magnitude.py](generalized_orders_of_magnitude/generalized_orders_of_magnitude.py). To see a list of them, execute the following on a Python command line:

```python
import generalized_orders_of_magnitude as goom
print('List of implemented functions:', *[
    name for name in dir(goom)
    if (not name.startswith('_')) and
    (name not in ['dataclass', 'math', 'torch', 'Config', 'config'])
], sep='\n')
```

To see the docstring and implementation of any function, type its name followed by "??" on a Python command line, as usual.


## Configuration Options

Our library has three configuration options, set to sensible defaults. They are:

* `goom.config.keep_logs_finite` (boolean): If True, `goom.log()` always returns finite values. The finite value returned for any input element numerically equal to zero will numerically exponentiate to zero in the specified float dtype. If False, `goom.log()` returns `float("-inf")` values for inputs numerically equal to zero. Default: True. 

* `goom.config.cast_all_logs_to_complex` (boolean): If True, `goom.log()` always returns complex-typed tensors. If False, `goom.log()` returns float tensors whenever all real input elements are equal to or greater than zero. Setting this option to False can improve performance and reduce memory use when working with real values that are always non-negative, such as measures and probabilities. Default: True.

* `goom.config.float_dtype` (torch.dtype): Float dtype of real and imaginary components of complex GOOMs, and of real GOOMs. Default: `torch.float32`, _i.e._, complex-typed GOOMs are represented by default as `torch.complex64` tensors with `torch.float32` real and imaginary components. For greater precision, set `goom.config.float_dtype = torch.float64`. Note: We have only tested this configuration option only with `torch.float32` and `torch.float64`.


## Replicating Published Results

In our paper, we perform three representative experiments: (1) compounding up to one million real matrix products beyond standard floating-point limits; (2) estimating spectra of Lyapunov exponents in parallel, using a novel selective-resetting method to prevent state colinearity; and (3) training deep recurrent neural networks that maintain long-range dependencies without numerical degradation, despite allowing recurrent state elements to fluctuate freely over time steps. To replicate our experiments, follow the instructions below.

### 1. Chains of Matrix Products that Compound Magnitudes toward Infinity

The code below will attempt to compute chains of up to 1M products of real random matrices, each with elements independently sampled from a normal distribution, over torch.float32, torch.float64, and complex64 GOOMs (_i.e._, with torch.float32 real and imaginary components). For every matrix size, for each data type, the code will attempt to compute the entire chain 30 times. WARNING: If you run the code below on a CPU, it will take a LONG time, because all product chains finish successfully with GOOMs.

```python
import torch
import generalized_orders_of_magnitude as goom
from tqdm import tqdm

goom.config.keep_logs_finite = True
goom.config.float_dtype = torch.float32

DEVICE = 'cuda'                                # change as needed
n = 1_000_000                                  # maximum chain length
n_runs = 30                                    # number of runs per matrix size
d_list = [8, 16, 32, 64, 128, 256, 512, 1024]  # square matrix sizes

longest_chains = []
for dtype in [torch.float32, torch.float64]:
    for run_number in tqdm(range(n_runs), desc=f'Runs over R with {dtype}'):
        for d in d_list:
            state = torch.randn(d, d, dtype=dtype, device=DEVICE)
            for t in range(n):
                update = torch.randn(d, d, dtype=dtype, device=DEVICE)
                state = torch.matmul(state, update)
                if not state.isfinite().all().item():
                    break
            longest_chains.append({
                'method': 'MatMul_over_R', 'dtype_name': str(dtype),
                'run_number': run_number, 'd': d, 'n_completed': t + 1,
            })

for run_number in tqdm(range(n_runs), desc="Runs over GOOMs with torch.complex64"):
    for d in d_list:
        log_state = goom.log(torch.randn(d, d, dtype=torch.float32, device=DEVICE))
        for t in range(n):
            log_update = goom.log(torch.randn(d, d, dtype=torch.float32, device=DEVICE))
            log_state = goom.log_matmul_exp(log_state, log_update)
            if not log_state.isfinite().all().item():
                break
        longest_chains.append({
            'method': 'LogMatMulExp_over_GOOMs', 'dtype_name': 'torch.complex64',
            'run_number': run_number, 'd': d, 'n_completed': t + 1,
        })

print(*longest_chains, sep='\n')
```

### 2. Parallel Estimation of the Spectrum of Lyapunov Exponents over GOOMs

See https://github.com/glassroom/parallel_lyapunov_exponents.


### 3. Deep RNN Modeling Sequences with Non-Diagonal SSMs over GOOMs

See https://github.com/glassroom/goom_ssm_rnn


## Compromises and Limitations

TODO: Show that LMME can, in principle, be implemented as log-sume-exp of elementwise sums. Explain current compromise implementation, due to memory bandwidth issues with obvious approaches.


## Background

The work here originated with casual conversations over email between us, the authors, in which we wondered if it might be possible to find a succinct expression for computing non-diagonal linear recurrences in parallel, by mapping them to the domain of complex logarithms. Our casual conversations gradually evolved into the development of generalized orders of magnitude, along with an algorithm for estimating Lyapunov exponents in parallel, and a novel method for selectively resetting interim states in a parallel prefix scan. We hope others find our work and our code useful.


## Citing

