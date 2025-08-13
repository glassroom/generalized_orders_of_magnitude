# coding: utf-8

import torch
import torch.utils.benchmark

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import generalized_orders_of_magnitude as goom

np.seterr(all='ignore')  # only for prettier command-line output; OK to remove


# Global constants:

DEVICE = 'cuda'  # change as needed
TORCH_FLOAT_DTYPES_TO_TEST = [torch.float32, torch.float64]
FIGSIZE = (7, 3.5)


# Helper functions:

def cast_np_float128_to_torch_float(x, torch_dtype, device):
    match torch_dtype:
        case torch.float32:
            return torch.tensor(x.astype(np.float32), device=device)
        case torch.float64:
            return torch.tensor(x.astype(np.float64), device=device)
        case _:
            raise ValueError(f'Unsupported torch dtype {torch_dtype}')

def get_goom_and_float_titles(dtype):
    match dtype:
        case np.float32 | torch.float32:
            return 'Complex64 GOOM', 'Float32'
        case np.float64 | torch.float64:
            return 'Complex128 GOOM', 'Float64'
        case _:
            raise ValueError(f'Unsupported dtype {dtype}')


# Functions for plotting figures:

def plot_one_arg_errors(x, y, y_via_goom, y_via_float, func_desc):
    assert all((x.dtype == np.float128, y.dtype == np.float128)), 'Reference values must have dtype np.float128'
    goom_title, float_title = get_goom_and_float_titles(y_via_float.dtype)

    fig, axes = plt.subplots(ncols=2, figsize=FIGSIZE, sharex=True, sharey=True, layout='constrained')

    axis = axes[0]
    axis.grid()
    axis.set(title=goom_title, xlabel=r'$\log_{10} x $')
    axis.plot(np.log10(x), np.log10(np.abs(y - y_via_goom)), alpha=0.7, lw=0.5)
    
    axis = axes[1]
    axis.grid()
    axis.set(title=float_title, xlabel=r'$\log_{10} x $')
    axis.plot(np.log10(x), np.log10(np.abs(y - y_via_float)), alpha=0.7, lw=0.5)

    fig.suptitle(f'Magnitude of Error versus Float128 for {func_desc}')
    axes[0].set(ylabel=r'$\log_{10} \left| y - \hat{y} \right|$')
    return fig


def plot_two_arg_errors(x, y, z, z_via_goom, z_via_float, func_desc):
    assert all((x.dtype == np.float128, y.dtype == np.float128, z.dtype == np.float128)), 'Reference values must have dtype np.float128'
    goom_title, float_title = get_goom_and_float_titles(z_via_float.dtype)
    img_extent = [np.log10(x).min(), np.log10(x).max(), np.log10(y).min(), np.log10(y).max()]

    fig, axes = plt.subplots(ncols=2, figsize=FIGSIZE, sharex=True, sharey=True, layout='constrained')

    axis = axes[0]
    axis.set(title=goom_title, xlabel=r'$\log_{10} x $', ylabel=r'$\log_{10} y$')
    img = np.log10(np.abs(z - z_via_goom))
    _colorable = axis.imshow(img, origin='lower', extent=img_extent, vmax=np.ceil(img.max()), cmap='viridis')

    axis = axes[1]
    axis.set(title=float_title, xlabel=r'$\log_{10} x $')
    img = np.log10(np.abs(z - z_via_float))
    axis.imshow(img, origin='lower', extent=img_extent, vmax=np.ceil(img.max()), cmap='viridis')

    fig.colorbar(_colorable, shrink=0.7, ax=axes.ravel().tolist(), label=r'$\log_{10} \left| z - \hat{z} \right| $')   
    fig.suptitle(f'Magnitude of Error versus Float128 for {func_desc}')
    return fig


def plot_matmul_errors(z, z_via_goom, z_via_float, matmul_desc):
    assert z.dtype == np.float128, 'Reference values must have dtype np.float128'
    goom_title, float_title = get_goom_and_float_titles(z_via_float.dtype)

    z_norm = np.linalg.norm(z, ord='fro')
    normalized_errors_via_goom = (z - z_via_goom).flatten() / z_norm
    normalized_errors_via_float = (z - z_via_float).flatten() / z_norm

    _, bins = np.histogram(normalized_errors_via_goom, bins=1000)

    fig, axes = plt.subplots(ncols=2, figsize=FIGSIZE, sharex=True, sharey=True, layout='constrained')

    axis = axes[0]
    axis.grid()
    axis.set(title=goom_title, xlabel=r'Number of Elements', yscale='symlog')
    axis.set(ylabel=r'$\frac{ Z - \hat{Z} }{ \| Z \|_2 }$')
    axis.hist(normalized_errors_via_goom, bins=bins, orientation='horizontal', alpha=0.7)

    axis = axes[1]
    axis.grid()
    axis.set(title=float_title, xlabel=r'Number of Elements', yscale='symlog')
    axis.hist(normalized_errors_via_float, bins=bins, orientation='horizontal', alpha=0.7)

    fig.suptitle(f'Histogram of Normalized Errors versus Float128 for {matmul_desc}')
    lim = 10 ** np.round(np.log10(np.abs(normalized_errors_via_goom).max()))
    axes[0].set(yticks=[lim * r for r in (-1, -0.5, 0, 0.5, 1)], ylim=(-lim, lim))
    return fig


def plot_execution_times(times, dtype):
    goom_title, float_title = get_goom_and_float_titles(dtype)
    n_runs = times[0]['n_runs']
    fig, axis = plt.subplots(figsize=FIGSIZE, layout='constrained')
    df = pd.DataFrame(times)  # ; print(df)
    df.plot.barh(x='func_desc', y='relative_time', ax=axis, legend=False)
    axis.set(title=f'Relative Execution Time for One- and Two-Argument Functions,\n{goom_title} versus {float_title}, 100M Elements in Parallel')
    axis.set(ylabel='', xlabel=f'Multiple of Execution Time\n(Mean of {n_runs} runs)')
    axis.grid(axis='x')
    return fig


# Code for computing comparisons, generating figures, and saving them:

for dtype in TORCH_FLOAT_DTYPES_TO_TEST:

    goom_title, float_title = get_goom_and_float_titles(dtype)
    goom_camel, float_camel = (s.lower().replace(' ', '_') for s in [goom_title, float_title])

    goom.config.float_dtype = dtype  # set dtype of GOOM real and imag components


    # Errors on one-argument functions:
    print(f'### Errors on one-argument functions, {goom_title} vs {float_title} ###')

    p = np.round(np.abs(np.log10(torch.finfo(dtype).resolution)))   # min/max power of 10 to test
    x = 10 ** np.linspace(-p, p, 1024 * 10_000, dtype=np.float128)  # np.float128
    float_x = cast_np_float128_to_torch_float(x, dtype, DEVICE)     # torch.float32   OR torch.float64
    log_x = goom.log(float_x)                                       # torch.complex64 OR torch.complex128

    print(f'Reciprocals, {goom_title} vs {float_title}...')
    y = 1.0 / x
    y_via_float = (1.0 / float_x).to('cpu').numpy()
    y_via_goom = goom.exp(torch.complex(real=-log_x.real, imag=log_x.imag)).to('cpu').numpy()
    fig = plot_one_arg_errors(x, y, y_via_goom, y_via_float, r'Reciprocals, $y = 1 / x$')
    fig.savefig(f'appendix_fig_{goom_camel}_vs_{float_camel}_errors_on_reciprocals.png', dpi=300)

    print(f'Square roots, {goom_title} vs {float_title}...')
    y = np.sqrt(x)
    y_via_float = torch.sqrt(float_x).to('cpu').numpy()
    y_via_goom = goom.exp(log_x * 0.5).to('cpu').numpy()
    fig = plot_one_arg_errors(x, y, y_via_goom, y_via_float, r'Square Roots, $y = \sqrt{x}$')
    fig.savefig(f'appendix_fig_{goom_camel}_vs_{float_camel}_errors_on_square_roots.png', dpi=300)

    print(f'Squares, {goom_title} vs {float_title}...')
    y = x ** 2
    y_via_float = (float_x ** 2).to('cpu').numpy()
    y_via_goom = goom.exp(log_x * 2).to('cpu').numpy()
    fig = plot_one_arg_errors(x, y, y_via_goom, y_via_float, r'Squares, $y = x^2$')
    fig.savefig(f'appendix_fig_{goom_camel}_vs_{float_camel}_errors_on_squares.png', dpi=300)

    print(f'Natural logarithms, {goom_title} vs {float_title}...')
    y = np.log(x)
    y_via_float = torch.log(float_x).to('cpu').numpy()
    y_via_goom = log_x.real.to('cpu').numpy()
    fig = plot_one_arg_errors(x, y, y_via_goom, y_via_float, r'Natural Logarithms, $y = \log x$')
    fig.savefig(f'appendix_fi_{goom_camel}_vs_{float_camel}_errors_on_natural_logarithms.png', dpi=300)

    print(f'Comparing exponentials, {goom_title} vs {float_title}...')
    # Use smaller magnitudes to test exp():
    x = 10 ** np.linspace(-1, 1, 1024 * 10_000, dtype=np.float128)         # np.float128
    float_x = cast_np_float128_to_torch_float(x, dtype, DEVICE)            # torch.float32   OR torch.float64
    log_x = goom.log(float_x)                                              # torch.complex64 OR torch.complex128
    y = np.exp(x)
    y_via_float = torch.exp(float_x).to('cpu').numpy()
    y_via_goom = torch.exp(goom.exp(log_x)).to('cpu').numpy()
    fig = plot_one_arg_errors(x, y, y_via_goom, y_via_float, r'Exponentials, $y = e^x$')
    fig.savefig(f'appendix_fig_{goom_camel}_vs_{float_camel}_errors_on_exponentials.png', dpi=300)


    # Errors on two-argument functions:
    print(f'### Errors on two-argument functions, {goom_title} vs {float_title} ###')

    p = np.round(np.abs(np.log10(torch.finfo(dtype).resolution)))   # min/max power of 10 to test
    x = 10 ** np.linspace(-p, p, 1024 * 10, dtype=np.float128)      # np.float128
    float_x = cast_np_float128_to_torch_float(x, dtype, DEVICE)     # torch.float32   OR torch.float64
    log_x = goom.log(float_x)                                       # torch.complex64 OR torch.complex128
    y, float_y, log_y = (x, float_x, log_x)                         # second argument                       

    float_desc = str(dtype).split('.')[-1].lower()
    goom_desc = 'complex64goom' if float_desc.endswith('32') else 'complex128goom'

    print(f'Scalar addition, {goom_title} vs {float_title}...')
    z = x[None, :] + y[:, None]
    z_via_float = (float_x[None, :] + float_y[:, None]).to('cpu').numpy()
    z_via_goom = (goom.exp(log_x)[None, :] + goom.exp(log_y)[:, None]).to('cpu').numpy()
    fig = plot_two_arg_errors(x, y, z, z_via_goom, z_via_float, r'Scalar Addition, $z = x + y$')
    fig.savefig(f'appendix_fig_{goom_camel}_vs_{float_camel}_errors_on_scalar_addition.png', dpi=300)

    print(f'Scalar multiplication, {goom_title} vs {float_title}...')
    z = x[None, :] * y[:, None]
    z_via_float = (float_x[None, :] * float_y[:, None]).to('cpu').numpy()
    z_via_goom = goom.exp(log_x[None, :] + log_y[:, None]).to('cpu').numpy()
    fig = plot_two_arg_errors(x, y, z, z_via_goom, z_via_float, r'Scalar Multiplication, $z = x y$')
    fig.savefig(f'appendix_fig_errors_{goom_camel}_vs_{float_camel}_on_scalar_multiplication.png', dpi=300)


    # Errors on matrix products:
    print(f'### Errors on a matrix product, {goom_title} vs {float_title} ###')

    d = 1024
    print(f'{d}x{d} matrix product, {goom_title} vs {float_title}...')

    float_x = torch.randn(d, d, dtype=goom.config.float_dtype, device=DEVICE)
    float_y = torch.randn(d, d, dtype=goom.config.float_dtype, device=DEVICE)
    z_via_float = (float_x @ float_y).to('cpu').numpy()

    log_x = goom.log(float_x)
    log_y = goom.log(float_y)
    z_via_goom = goom.exp(goom.log_matmul_exp(log_x, log_y)).to('cpu').numpy()

    x = float_x.to('cpu').numpy().astype(np.float128)
    y = float_y.to('cpu').numpy().astype(np.float128)
    z = x @ y  # note: very slow on a CPU!

    _matmul_desc = \
        f'a Matrix Product, $Z = X Y$,\n' \
        + f'where $X, Y$ are {d}×{d} Matrices with Elements Sampled from ' \
        + r'$\mathcal{N}(0, 1)$'
    fig = plot_matmul_errors(z, z_via_goom, z_via_float, _matmul_desc)
    fig.savefig(f'appendix_fig_{goom_camel}_vs_{float_camel}_errors_on_matrix_product.png', dpi=300)


    # Execution times on one-argument functions:
    print(f'### Execution times on one-argument functions, {goom_title} vs {float_title} ###')

    float_x = torch.rand(1024 * 100_000, dtype=dtype, device=DEVICE)  # torch.float32   OR torch.float64
    float_y = torch.rand(1024 * 100_000, dtype=dtype, device=DEVICE)  # torch.float32   OR torch.float64

    log_x = goom.log(float_x)                                         # torch.complex64 OR torch.complex128
    log_y = goom.log(float_y)                                         # torch.complex64 OR torch.complex128

    # Helper one-argument functions:
    def _log_reciprocal_exp_func(log_x):
        log_x.real *= -1
        return log_x

    def _log_square_root_exp_func(log_x):
        log_x *= 0.5
        return log_x

    def _log_square_exp_func(log_x):
        log_x *= 2
        return log_x

    def _log_exp_func(log_x):
        return log_x

    def _log_exp_exp_func(log_x):
        return goom.exp(log_x)

    n_runs = 30
    times = []

    for func_desc, float_func, goom_func in [
        ['Reciprocals',         lambda float_x: 1.0 / float_x,        _log_reciprocal_exp_func],
        ['Square Roots',        lambda float_x: torch.sqrt(float_x),  _log_square_root_exp_func],
        ['Squares',             lambda float_x: float_x ** 2,         _log_square_exp_func],
        ['Natural Logarithms',  lambda float_x: torch.log(float_x),   _log_exp_func],
        ['Exponentials',        lambda float_x: torch.exp(float_x),   _log_exp_exp_func],
    ]:
        print(f'{func_desc}, {n_runs} runs, {goom_title} vs {float_title}...')

        goom_mean_time = torch.utils.benchmark.Timer(
            stmt='goom_func(log_x)',
            globals={ 'goom_func': goom_func, 'log_x': log_x, }
        ).timeit(n_runs).mean

        float_mean_time = torch.utils.benchmark.Timer(
            stmt='float_func(float_x)',
            globals={ 'float_func': float_func, 'float_x': float_x, }
        ).timeit(n_runs).mean

        times.append({
            'func_desc': func_desc,
            'n_runs': n_runs,
            'relative_time': goom_mean_time / float_mean_time,
        })

    # Execution times on two-argument functions:
    print(f'### Execution times on two-argument functions, {goom_title} vs {float_title} ###')

    for func_desc, float_func, goom_func in [
        ['Scalar Addition',  lambda float_x, float_y: float_x + float_y,   goom.log_add_exp],
        ['Scalar Product',   lambda float_x, float_y: float_x * float_y,   lambda log_x, log_y: log_x + log_y],
    ]:
        print(f'{func_desc}, {n_runs} runs, {goom_title} vs {float_title}...')

        goom_mean_time = torch.utils.benchmark.Timer(
            stmt='goom_func(log_x, log_y)',
            globals={ 'goom_func': goom_func, 'log_x': log_x, 'log_y': log_y}
        ).timeit(n_runs).mean

        float_mean_time = torch.utils.benchmark.Timer(
            stmt='float_func(float_x, float_y)',
            globals={ 'float_func': float_func, 'float_x': float_x, 'float_y': float_y, }
        ).timeit(n_runs).mean

        times.append({
            'func_desc': func_desc,
            'n_runs': n_runs,
            'relative_time': goom_mean_time / float_mean_time,
        })

    fig = plot_execution_times(times, dtype)
    fig.savefig(f'appendix_fig_{goom_camel}_vs_{float_camel}_execution_times.png', dpi=300)
