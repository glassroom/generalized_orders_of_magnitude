# generalized_orders_of_magnitude

Reference implementation of Generalized Orders of Magnitude.

TODO: Write a brief introduction here.


## Matrix Multiplication over GOOMs

TODO: Write about equivalency here.

```python
import torch
import generalized_orders_of_magnitude as goom

# A matrix multiplication:
A = torch.randn(5, 4)
B = torch.randn(4, 3)
Y = A @ B
print('Y:\n{}\n'.format(Y))

# The same multiplication, over GOOMs:
log_A = goom.log(A)
log_B = goom.log(B)
log_Y = goom.log_matmul_exp(log_A, log_B)
print('exp(log_Y):\n{}\n'.format(goom.exp(log_Y)))
```

## Chains of Matrix Products over GOOMs 

TODO: Write about chains of products here.

Install `torch_parallel_scan` from [https://github.com/glassroom/torch_parallel_scan/].

```python
import torch
import generalized_orders_of_magnitude as goom
import torch_parallel_scan as tps

# A chain of matrix products:
n, d = (5, 4)
A = torch.randn(n, d, d, device=DEVICE) / math.sqrt(d)
Y = torch.linalg.multi_dot([*A])
print('Y:\n{}\n'.format(Y))

# The same chain, over GOOMs:
log_A = goom.log(A)
log_Y = tps.reduce_scan(log_A, goom.log_matmul_exp, dim=-3)
print('exp(log_Y):\n{}\n'.format(goom.exp(log_Y)))
```

## Replicating Published Results

TODO: Describe experiment with chains of matrix products here. Maybe show a plot.

**WARNING: Running this code will take a LONG time, because all chains successfully finish all steps with GOOMs.**

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
            S = None
            for t in range(n):
                new_mat = torch.randn(d, d, dtype=dtype, device=DEVICE)
                S = new_mat if S is None else S @ new_mat
                if not S.isfinite().all().item():
                    break
            longest_chains.append({
                'method': 'MatMul_over_R', 'dtype_name': str(dtype),
                'run_number': run_number, 'd': d, 'n_completed': t + 1,
            })

for run_number in tqdm(range(n_runs), desc="Runs over GOOMs with torch.complex64"):
    for d in d_list:
        log_S = None
        for t in range(n):
            new_log_mat = goom.log(torch.randn(d, d, dtype=torch.float32, device=DEVICE))
            log_S = new_log_mat if log_S is None else goom.log_matmul_exp(log_S, new_log_mat)
            if not log_S.isfinite().all().item():
                break
        longest_chains.append({
            'method': 'LogMatMulExp_over_GOOMs', 'dtype_name': 'torch.complex64',
            'run_number': run_number, 'd': d, 'n_completed': t + 1,
        })

print(longest_chains)
```


## Configuration Options

Our library has three configuration options, set to sensible defaults. They are:

* `goom.config.keep_logs_finite` (boolean): If True, `goom.log()` always returns finite values. If False, `goom.log()` will return `float("-inf")` values for any real input elements that are zero. Default: True.

* `goom.config.cast_all_logs_to_complex` (boolean): If True, `goom.log()` always returns complex-typed tensors. If False, `goom.log()` will return float tensors when all real input elements are equal to or greater than zero. Default: True.

* `goom.config.float_dtype` (torch.dtype): Float dtype of real and imaginary components of complex logarithms, and of real logarithms. Default: `torch.float32`, _i.e._, complex-typed GOOMs are represented as `torch.complex64` tensors with `torch.float32` real and imaginary components. Note: We have only tested our code with this option set to `torch.float32` and `torch.float64`.
```
