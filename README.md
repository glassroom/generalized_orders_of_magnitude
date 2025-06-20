# generalized_orders_of_magnitude

Initial reference implementation of generalized orders of magnitude (GOOMs), for Pytorch. This repository implements GOOMs complex-typed Pytorch tensors that exponentiate elementwise to floating-point tensors. They enable you to work with real numbers that exceed the limits of torch.float32 and torch.float64.

TODO: A simple example showing floats failing but GOOMs succeeding


## Installing

```
pip install git+https://github.com/glassroom/generalized_orders_of_magnitude
```

Alternatively, you can download a single file to your project directory: [generalized_orders_of_magnitude.py](generalized_orders_of_magnitude/generalized_orders_of_magnitude.py).

The only dependency is a recent version of Pytorch.


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

TODO: Write brief intro about LMME.

```python
import torch
import generalized_orders_of_magnitude as goom

DEVICE = 'cuda'  # change as needed

# A matrix multiplication:
A = torch.randn(5, 4, device=DEVICE)
B = torch.randn(4, 3, device=DEVICE)
Y = A @ B
print('Y:\n{}\n'.format(Y))

# The same multiplication, over GOOMs:
log_A = goom.log(A)
log_B = goom.log(B)
log_Y = goom.log_matmul_exp(log_A, log_B)
print('exp(log_Y):\n{}\n'.format(goom.exp(log_Y)))
```

### Chains of Matrix Products over GOOMs 

Note: To be able to run the sample code below, you must first install [`torch_parallel_scan`](https://github.com/glassroom/torch_parallel_scan/).

```python
import torch
import generalized_orders_of_magnitude as goom
import torch_parallel_scan as tps  # install from https://github.com/glassroom/torch_parallel_scan/

DEVICE = 'cuda'  # change as needed

# A chain of matrix products:
n, d = (5, 4)
A = torch.randn(n, d, d, device=DEVICE) / (d ** 0.5)
Y = torch.linalg.multi_dot([*A])
print('Y:\n{}\n'.format(Y))

# The same chain, executed over GOOMs:
log_A = goom.log(A)
log_Y = tps.reduce_scan(log_A, goom.log_matmul_exp, dim=-3)
print('exp(log_Y):\n{}\n'.format(goom.exp(log_Y)))
```

## Configuration Options

Our library has three configuration options, set to sensible defaults. They are:

* `goom.config.keep_logs_finite` (boolean): If True, `goom.log()` always returns finite values. If False, `goom.log()` returns `float("-inf")` values for any real input values numerically equal to zero. Default: True.

* `goom.config.cast_all_logs_to_complex` (boolean): If True, `goom.log()` always returns complex-typed tensors. If False, `goom.log()` returns float tensors if all real input elements are equal to or greater than zero. Default: True.

* `goom.config.float_dtype` (torch.dtype): Float dtype of real and imaginary components of complex logarithms, and of real logarithms. Default: `torch.float32`, _i.e._, complex-typed GOOMs are represented as `torch.complex64` tensors with `torch.float32` real and imaginary components. For greater precision, set `goom.config.float_dtype = torch.float64`. Note: We have only tested this configuration option with `torch.float32` and `torch.float64`.


## Replicating Published Results

### Experiment 1: Chains of Matrix Products that Compound Element Magnitudes toward Infinity

TODO: Describe experiment with chains of matrix products here. Maybe show a plot.

**WARNING: Running the code will take a LONG time, because all chains successfully finish with GOOMs.**

```python
import tqdm
import torch
import generalized_orders_of_magnitude as goom
assert goom.config.keep_logs_finite == True and float_dtype == torch.float32

DEVICE = 'cuda'  # change as needed

n = 1_000_000    # maximum chain length
n_runs = 30      # number of runs per matrix size
d_list = [2 ** (i+1) for i in range(2, 10)]  # list of square matrix dims to try

longest_chains = []
for dtype in [torch.float32, torch.float64]:
    for run_number in tqdm(range(n_runs), desc=f'Runs over R with {dtype}'):
        for d in d_list:
            S = torch.randn(d, d, dtype=dtype, device=DEVICE)
            for t in range(n):
                A = torch.randn(d, d, dtype=dtype, device=DEVICE)
                S = S & A
                if not S.isfinite().all().item():
                    break
            longest_chains.append({
                'method': 'MatMul_over_R', 'dtype_name': str(dtype),
                'run_number': run_number, 'd': d, 'n_completed': t + 1,
            })

for run_number in tqdm(range(n_runs), desc="Runs over GOOMs with torch.complex64"):
    for d in d_list:
        log_S = goom.log(torch.randn(d, d, dtype=torch.float32, device=DEVICE))
        for t in range(n):
            log_A = goom.log(torch.randn(d, d, dtype=torch.float32, device=DEVICE))
            log_S = goom.log_matmul_exp(log_S, log_A)
            if not log_S.isfinite().all().item():
                break
        longest_chains.append({
            'method': 'LogMatMulExp_over_GOOMs', 'dtype_name': 'torch.complex64',
            'run_number': run_number, 'd': d, 'n_completed': t + 1,
        })

print(longest_chains)
```

### Experiment 2: Parallel Estimation of the Spectrum of Lyapunov Exponents of Dynamical Systems

See https://github.com/glassroom/parallel_lyapunov_exponents.


### Experiment 3: Deep RNN Capturing Sequential Dependencies with a Non-Diagonal SSM over GOOMs

See https://github.com/glassroom/goom_ssm_rnn


## Background

The work here originated with casual conversations over email between us, the authors, in which we wondered if it might be possible to find a succinct expression for computing non-diagonal linear recurrences in parallel, by mapping them to the domain of complex logarithms. Our casual conversations gradually evolved into the development of generalized orders of magnitude, an algorithm for estimating Lyapunov exponents in parallel, and a novel method for selectively resetting interim states in a parallel prefix scan. We hope others find our work and our code useful.


## Citing

