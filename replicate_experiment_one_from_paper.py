# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import generalized_orders_of_magnitude as goom

from tqdm import tqdm

# Configure goom library:
goom.config.keep_logs_finite = True
goom.config.float_dtype = torch.float32

# Specify a device:
DEVICE = 'cuda'

# Configure experiment:
n_runs = 30                                    # number of runs per matrix size
d_list = [8, 16, 32, 64, 128, 256, 512, 1024]  # square matrix sizes
n_steps = 1_000_000                            # maximum chain length

# Run experiment:
print("Running experiment 1:")

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
                'method': 'MatMul_over_R',
                'dtype_name': str(dtype),
                'run_number': run_number,
                'd': d,
                'n_completed': t + 1,
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
            'method': 'LogMatMulExp_over_GOOMs',
            'dtype_name': 'torch.complex64',
            'run_number': run_number,
            'd': d,
            'n_completed': t + 1,
        })

# Save results of experiment:
torch.save(longest_chains, 'longest_chains.pt')  # load with torch.load('longest_chains.pt')

# Plot results of experiment and save plot:
fig, axis = plt.subplots(figsize=(11, 4), layout='constrained')

for dtype_name in ['torch.complex64', 'torch.float64', 'torch.float32']:
    if 'complex' in dtype_name:
        label ='LMME over GOOMs, ' + dtype_name.split('.')[-1].title()
    else:
        label = 'Conventional MatMul, ' + dtype_name.split('.')[-1].title()

    df = pd.DataFrame(longest_chains)
    df = df[df['dtype_name'] == dtype_name].copy()
    df = df.drop(['method', 'dtype_name', 'run_number'], axis=1)
    df = df.groupby('d').agg(['mean', 'min', 'max', 'std']).droplevel(0, axis=1)
    df['mean'].plot(ax=axis, lw=2, marker='o', solid_capstyle='round', label=label)

axis.set(xscale='log', xticks=d_list, xticklabels=d_list, xlabel='Square Matrix Size')
axis.set(yscale='log', yticks=[10**i for i in range(1, int(np.log10(n_steps) + 1))], ylabel='Longest Chain')
axis.grid()
axis.legend(loc='upper center', ncols=3, framealpha=0, bbox_to_anchor=(0.5, -0.2))

fig.savefig('fig_longest_chains.png', dpi=300)
