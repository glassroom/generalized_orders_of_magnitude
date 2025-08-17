# coding: utf-8

import torch
import torch.utils.benchmark

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import generalized_orders_of_magnitude as goom


# Global constants:

DEVICE = 'cuda'  # change as needed
TORCH_FLOAT_DTYPES_TO_TEST = [torch.float32, torch.float64]  # the two highest-precision floats supported by cuda

N_RUNS_FOR_TIME_BENCHMARKS = 30                         # will measure execution time as the mean of this number of runs
N_SAMPLES_FOR_TIME_AND_MEM_BENCHMARKS = 1024 * 100_000  # will measure execution time for this number of samples in parallel
N_SAMPLES_FOR_ONE_ARG_ERRORS = 1024 * 10_000            # will measure error vs Float128 for this number of samples
N_SAMPLES_FOR_TWO_ARG_ERRORS = 1024 * 10                # will measure error vs Float128 for *the square of* this number of samples
N_DIMS_FOR_TESTED_MATMUL = 1024                         # will compare error, time, and memory for a square matrix of this size

FIG_SIZE = (7, 3.5)                   # size of all figures in inches
FIG_DPI = 300                         # dots per inch of all figures, when saved as PNGs
FIG_FONTSIZE = 8                      # reference font size for all images
FIG_FILENAME_PREFIX = 'appendix_fig'  # all PNG files will have names starting with this prefix


# Global configuration:

torch.set_float32_matmul_precision('highest')      # for comparing matmul precision to Float32
plt.rcParams.update({'font.size': FIG_FONTSIZE })  # base font size for figures
np.seterr(all='ignore')                            # for prettier command-line output; OK to remove


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

    fig, axes = plt.subplots(ncols=2, figsize=FIG_SIZE, sharex=True, sharey=True, layout='constrained')

    axis = axes[0]
    axis.grid()
    axis.set(title=goom_title, xlabel=r'$\log_{10} x$')
    axis.plot(np.log10(x), np.log10(np.abs(y - y_via_goom)), alpha=0.7, lw=0.5)
    
    axis = axes[1]
    axis.grid()
    axis.set(title=float_title, xlabel=r'$\log_{10} x$')
    axis.plot(np.log10(x), np.log10(np.abs(y - y_via_float)), alpha=0.7, lw=0.5)

    fig.suptitle(f'Magnitude of Error versus Float128 for {func_desc}')
    axes[0].set(ylabel=r'$\log_{10} \left| y - \hat{y} \right|$')
    fig.text(0.0, 0.0, '$y:$ Float128\n$\\hat{y}:$ Tested Value', fontsize=FIG_FONTSIZE * 5/8, ha="left", va='bottom')
    return fig


def plot_two_arg_errors(x, y, z, z_via_goom, z_via_float, func_desc):
    assert all((x.dtype == np.float128, y.dtype == np.float128, z.dtype == np.float128)), 'Reference values must have dtype np.float128'
    goom_title, float_title = get_goom_and_float_titles(z_via_float.dtype)
    img_extent = [np.log10(x).min(), np.log10(x).max(), np.log10(y).min(), np.log10(y).max()]

    fig, axes = plt.subplots(ncols=2, figsize=FIG_SIZE, sharex=True, sharey=True, layout='constrained')

    axis = axes[0]
    axis.set(title=goom_title, xlabel=r'$\log_{10} x$', ylabel=r'$\log_{10} y$')
    img = np.log10(np.abs(z - z_via_goom))
    vmax = np.ceil(img.max())
    _colorable = axis.imshow(img, origin='lower', extent=img_extent, vmax=vmax, cmap='viridis')
    axis.set(xticks=axis.get_yticks())

    axis = axes[1]
    axis.set(title=float_title, xlabel=r'$\log_{10} x$')
    img = np.log10(np.abs(z - z_via_float))
    axis.imshow(img, origin='lower', extent=img_extent, vmax=vmax, cmap='viridis')
    axis.set(xticks=axis.get_yticks())

    fig.colorbar(_colorable, shrink=0.7, ax=axes.ravel().tolist(), label=r'$\log_{10} \left| z - \hat{z} \right| $')   
    fig.suptitle(f'Magnitude of Error versus Float128 for {func_desc}')
    fig.text(0.9, 0.0, '$z:$ Float128\n$\\hat{z}:$ Tested Value', fontsize=FIG_FONTSIZE * 5/8, ha="left", va='bottom')
    return fig


def plot_matmul_errors(Z, Z_via_goom, Z_via_float, matmul_desc):
    assert Z.dtype == np.float128, 'Reference values must have dtype np.float128'
    goom_title, float_title = get_goom_and_float_titles(Z_via_float.dtype)

    Z_norm = np.linalg.norm(Z, ord='fro')
    normalized_errors_via_goom = (Z - Z_via_goom).flatten() / Z_norm
    normalized_errors_via_float = (Z - Z_via_float).flatten() / Z_norm

    _, bins = np.histogram(normalized_errors_via_goom, bins=1000)

    fig, axes = plt.subplots(ncols=2, figsize=FIG_SIZE, sharex=True, sharey=True, layout='constrained')

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
    fig.text(0.0, 0.0, '$Z:$ Float128\n$\\hat{Z}:$ Tested Value', fontsize=FIG_FONTSIZE * 5/8, ha="left", va='bottom')
    return fig


def plot_execution_times(times, dtype):
    goom_title, float_title = get_goom_and_float_titles(dtype)

    fig, axis = plt.subplots(figsize=FIG_SIZE, layout='constrained')
    df = pd.DataFrame(times)
    df.plot.barh(ax=axis, x='func_desc', y='relative_time', legend=False, alpha=0.7)
    axis.set(xlabel=f'Execution Time, {goom_title} as a Multiple of {float_title}\n(Mean of {N_RUNS_FOR_TIME_BENCHMARKS} Runs)')
    axis.set(ylabel='')
    axis.invert_yaxis()
    axis.grid(axis='x')

    axis.set(title=f'Execution Time on an Nvidia GPU, {goom_title} as a Multiple of {float_title}')
    d, n_in_millions = (N_DIMS_FOR_TESTED_MATMUL, int(10 ** np.round(np.log10(N_SAMPLES_FOR_TIME_AND_MEM_BENCHMARKS) - 6)))
    fig.text(0.0, 0.0, f'Scalar functions: {n_in_millions}M in parallel\nMatrix product: {d}×{d} mats', fontsize=FIG_FONTSIZE * 5/8, ha="left", va='bottom')
    return fig


def plot_peak_memory_allocs(peak_allocs, dtype):
    goom_title, float_title = get_goom_and_float_titles(dtype)

    fig, axis = plt.subplots(figsize=FIG_SIZE, layout='constrained')
    df = pd.DataFrame(peak_allocs)
    df.plot.barh(ax=axis, x='func_desc', y='relative_peak_alloc', legend=False, alpha=0.7)
    axis.set(xlabel=f'Peak Memory Allocated, {goom_title} as a Multiple of {float_title}\n(Including Memory Allocated to Input, Interim, and Output Tensors)')
    axis.set(ylabel='')
    axis.invert_yaxis()
    axis.grid(axis='x')

    axis.set(title=f'Peak Memory Allocated on an Nvidia GPU, {goom_title} as a Multiple of {float_title}')
    d, n_in_millions = (N_DIMS_FOR_TESTED_MATMUL, int(10 ** np.round(np.log10(N_SAMPLES_FOR_TIME_AND_MEM_BENCHMARKS) - 6)))
    fig.text(0.0, 0.0, f'Scalar functions: {n_in_millions}M in parallel\nMatrix product: {d}×{d} mats', fontsize=FIG_FONTSIZE * 5/8, ha="left", va='bottom')
    return fig


# Function for saving and closing figures:

def save_and_close_fig(fig, fig_desc):
    fig.savefig(f'{FIG_FILENAME_PREFIX}_{fig_desc}.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)  # so memory can be released


# Functions for running comparisons and generating figures:

def compare_errors():

    dtype = goom.config.float_dtype

    goom_title, float_title = get_goom_and_float_titles(dtype)
    goom_camel, float_camel = (s.lower().replace(' ', '_') for s in [goom_title, float_title])

    print(f'\n### Errors on one-argument functions, {goom_title} vs {float_title} ###')

    p = np.round(np.abs(np.log10(torch.finfo(dtype).resolution)))                  # max neg/pos power of 10 to test
    x = 10 ** np.linspace(-p, p, N_SAMPLES_FOR_ONE_ARG_ERRORS, dtype=np.float128)  # np.float128
    float_x = cast_np_float128_to_torch_float(x, dtype, DEVICE)                    # torch.float32   OR torch.float64
    log_x = goom.log(float_x)                                                      # torch.complex64 OR torch.complex128

    print(f'Reciprocals, {goom_title} vs {float_title}...')
    y = 1.0 / x
    y_via_float = (1.0 / float_x).to('cpu').numpy()
    y_via_goom = goom.exp(torch.complex(real=-log_x.real, imag=log_x.imag)).to('cpu').numpy()
    fig = plot_one_arg_errors(x, y, y_via_goom, y_via_float, r'Reciprocals, $y = 1 / x$')
    save_and_close_fig(fig, fig_desc=f'{goom_camel}_vs_{float_camel}_errors_on_reciprocals')

    print(f'Square roots, {goom_title} vs {float_title}...')
    y = np.sqrt(x)
    y_via_float = torch.sqrt(float_x).to('cpu').numpy()
    y_via_goom = goom.exp(log_x * 0.5).to('cpu').numpy()
    fig = plot_one_arg_errors(x, y, y_via_goom, y_via_float, r'Square Roots, $y = \sqrt{x}$')
    save_and_close_fig(fig, fig_desc=f'{goom_camel}_vs_{float_camel}_errors_on_square_roots')

    print(f'Squares, {goom_title} vs {float_title}...')
    y = x ** 2
    y_via_float = (float_x ** 2).to('cpu').numpy()
    y_via_goom = goom.exp(log_x * 2).to('cpu').numpy()
    fig = plot_one_arg_errors(x, y, y_via_goom, y_via_float, r'Squares, $y = x^2$')
    save_and_close_fig(fig, fig_desc=f'{goom_camel}_vs_{float_camel}_errors_on_squares')

    print(f'Natural logarithms, {goom_title} vs {float_title}...')
    y = np.log(x)
    y_via_float = torch.log(float_x).to('cpu').numpy()
    y_via_goom = log_x.real.to('cpu').numpy()
    fig = plot_one_arg_errors(x, y, y_via_goom, y_via_float, r'Natural Logarithms, $y = \log x$')
    save_and_close_fig(fig, fig_desc=f'{goom_camel}_vs_{float_camel}_errors_on_natural_logarithms')

    print(f'Exponentials, {goom_title} vs {float_title}...')
    # Use smaller magnitudes to test exp():
    x = 10 ** np.linspace(-5, 1, N_SAMPLES_FOR_ONE_ARG_ERRORS, dtype=np.float128)  # np.float128
    float_x = cast_np_float128_to_torch_float(x, dtype, DEVICE)                    # torch.float32   OR torch.float64
    log_x = goom.log(float_x)                                                      # torch.complex64 OR torch.complex128
    y = np.exp(x)
    y_via_float = torch.exp(float_x).to('cpu').numpy()
    y_via_goom = torch.exp(goom.exp(log_x)).to('cpu').numpy()
    fig = plot_one_arg_errors(x, y, y_via_goom, y_via_float, r'Exponentials, $y = e^x$')
    save_and_close_fig(fig, fig_desc=f'{goom_camel}_vs_{float_camel}_errors_on_exponentials')


    # Measure errors on two-argument functions:
    print(f'\n### Errors on two-argument functions, {goom_title} vs {float_title} ###')

    p = np.round(np.abs(np.log10(torch.finfo(dtype).resolution)))                  # max neg/pos power of 10 to test
    x = 10 ** np.linspace(-p, p, N_SAMPLES_FOR_TWO_ARG_ERRORS, dtype=np.float128)  # np.float128
    float_x = cast_np_float128_to_torch_float(x, dtype, DEVICE)                    # torch.float32   OR torch.float64
    log_x = goom.log(float_x)                                                      # torch.complex64 OR torch.complex128
    y, float_y, log_y = (x, float_x, log_x)                                        # second argument                       

    float_desc = str(dtype).split('.')[-1].lower()
    goom_desc = 'complex64goom' if float_desc.endswith('32') else 'complex128goom'

    print(f'Scalar addition, {goom_title} vs {float_title}...')
    z = x[None, :] + y[:, None]
    z_via_float = (float_x[None, :] + float_y[:, None]).to('cpu').numpy()
    z_via_goom = (goom.exp(log_x)[None, :] + goom.exp(log_y)[:, None]).to('cpu').numpy()
    fig = plot_two_arg_errors(x, y, z, z_via_goom, z_via_float, r'Scalar Addition, $z = x + y$')
    save_and_close_fig(fig, fig_desc=f'{goom_camel}_vs_{float_camel}_errors_on_scalar_addition')

    print(f'Scalar product, {goom_title} vs {float_title}...')
    z = x[None, :] * y[:, None]
    z_via_float = (float_x[None, :] * float_y[:, None]).to('cpu').numpy()
    z_via_goom = goom.exp(log_x[None, :] + log_y[:, None]).to('cpu').numpy()
    fig = plot_two_arg_errors(x, y, z, z_via_goom, z_via_float, r'Scalar Product, $z = x y$')
    save_and_close_fig(fig, fig_desc=f'{goom_camel}_vs_{float_camel}_errors_on_scalar_product')


    # Measure errors on matrix products:
    print(f'\n### Errors on a matrix product, {goom_title} vs {float_title} ###')

    d = N_DIMS_FOR_TESTED_MATMUL
    print(f'{d}x{d} matrix product, {goom_title} vs {float_title}...')

    float_X = torch.randn(d, d, dtype=goom.config.float_dtype, device=DEVICE)
    float_Y = torch.randn(d, d, dtype=goom.config.float_dtype, device=DEVICE)
    Z_via_float = (float_X @ float_Y).to('cpu').numpy()

    log_X = goom.log(float_X)
    log_Y = goom.log(float_Y)
    Z_via_goom = goom.exp(goom.log_matmul_exp(log_X, log_Y)).to('cpu').numpy()

    X = float_X.to('cpu').numpy().astype(np.float128)
    Y = float_Y.to('cpu').numpy().astype(np.float128)
    Z = X @ Y  # note: very slow on a CPU!

    _matmul_desc = \
        f'a Matrix Product, $Z = X Y$,\n' \
        + f'where $X, Y$ are {d}×{d} Matrices with Elements Sampled from ' \
        + r'$\mathcal{N}(0, 1)$'
    fig = plot_matmul_errors(Z, Z_via_goom, Z_via_float, _matmul_desc)
    save_and_close_fig(fig, fig_desc=f'{goom_camel}_vs_{float_camel}_errors_on_matrix_product')


def compare_execution_times():

    dtype = goom.config.float_dtype

    goom_title, float_title = get_goom_and_float_titles(dtype)
    goom_camel, float_camel = (s.lower().replace(' ', '_') for s in [goom_title, float_title])

    print(f'\n### Execution times, {goom_title} vs {float_title} ###')

    float_x = torch.rand(N_SAMPLES_FOR_TIME_AND_MEM_BENCHMARKS, dtype=dtype, device=DEVICE)
    float_y = torch.rand(N_SAMPLES_FOR_TIME_AND_MEM_BENCHMARKS, dtype=dtype, device=DEVICE)
    log_x = goom.log(float_x)
    log_y = goom.log(float_y)

    d = N_DIMS_FOR_TESTED_MATMUL
    float_X = torch.randn(d, d, dtype=goom.config.float_dtype, device=DEVICE)
    float_Y = torch.randn(d, d, dtype=goom.config.float_dtype, device=DEVICE)
    log_X = goom.log(float_X)
    log_Y = goom.log(float_Y)

    # Specify functions for comparing execution times, *in-place wherever possible*:
    def _log_reciprocal_exp_in_place(log_x):
        log_x.real.mul_(-1)
        return log_x

    def _log_sqrt_exp_in_place(log_x):
        log_x.mul_(0.5)
        return log_x

    def _log_square_exp_in_place(log_x):
        log_x.mul_(2)
        return log_x

    def _log_add_exp_mostly_in_place(log_x, log_y):
        return goom.log(goom.exp(log_x).add_(goom.exp(log_y)))  # no need to log-scale over tested values

    func_metadata = [
        # inp_desc,       func desc,             float func,              goom func,
        [ 'one_scalar',   'Reciprocals',         lambda x: 1.0 / x,       _log_reciprocal_exp_in_place,           ],
        [ 'one_scalar',   'Square Roots',        torch.sqrt_,             _log_sqrt_exp_in_place,                 ],
        [ 'one_scalar',   'Squares',             lambda x: x ** 2,        _log_square_exp_in_place,               ],
        [ 'one_scalar',   'Natural Logarithms',  torch.log_,              lambda log_x: log_x,                    ],  # gooms already are natural logs
        [ 'one_scalar',   'Exponentials',        torch.exp_,              torch.exp_,                             ],  # keeps GOOMs in complex log-domain
        [ 'two_scalars',  'Scalar Addition',     lambda x, y: x.add_(y),  _log_add_exp_mostly_in_place,           ],
        [ 'two_scalars',  'Scalar Product',      lambda x, y: x.mul_(y),  lambda log_x, log_y: log_x.add_(log_y), ],
        [ 'two_matrices', 'Matrix Product',      torch.matmul,            goom.log_matmul_exp,                    ],
    ]

    times = []
    for inp_desc, func_desc, float_func, goom_func in func_metadata:
        print(f'{func_desc.capitalize()}, {N_RUNS_FOR_TIME_BENCHMARKS} runs, {goom_title} vs {float_title}...')

        goom_mean_time = torch.utils.benchmark.Timer(
            stmt={
                'one_scalar': 'goom_func(log_x)',
                'two_scalars': 'goom_func(log_x, log_y)',
                'two_matrices': 'goom_func(log_Y, log_Y)',
            }[inp_desc],
            globals={
                'goom_func': goom_func,
                'log_x': log_x, 'log_y': log_y,
                'log_X': log_X, 'log_Y': log_Y,
            }
        ).timeit(N_RUNS_FOR_TIME_BENCHMARKS).mean

        float_mean_time = torch.utils.benchmark.Timer(
            stmt={
                'one_scalar': 'float_func(float_x)',
                'two_scalars': 'float_func(float_x, float_y)',
                'two_matrices': 'float_func(float_X, float_X)',
            }[inp_desc],
            globals={
                'float_func': float_func,
                'float_x': float_x, 'float_y': float_y,
                'float_X': float_X, 'float_Y': float_Y,
            }
        ).timeit(N_RUNS_FOR_TIME_BENCHMARKS).mean

        times.append({
            'func_desc': func_desc,
            'relative_time': goom_mean_time / float_mean_time,
        })

    fig = plot_execution_times(times, dtype)
    save_and_close_fig(fig, fig_desc=f'{goom_camel}_vs_{float_camel}_execution_times')


def compare_memory_allocated():

    dtype = goom.config.float_dtype

    goom_title, float_title = get_goom_and_float_titles(dtype)
    goom_camel, float_camel = (s.lower().replace(' ', '_') for s in [goom_title, float_title])

    print(f'\n### Memory Allocated, {goom_title} vs {float_title} ###')

    # Specify functions for comparing memory allocated, *never in-place*:
    func_metadata = [
        # inp_desc,       func desc,             float func,          goom func,
        [ 'one_scalar',   'Reciprocals',         lambda x: 1.0 / x,   lambda log_x: log_x * -1,           ],
        [ 'one_scalar',   'Square Roots',        torch.sqrt,          lambda log_x: log_x * 0.5,          ],
        [ 'one_scalar',   'Squares',             lambda x: x ** 2,    lambda log_x: log_x * 2,            ],
        [ 'one_scalar',   'Natural Logarithms',  torch.log,           lambda log_x: log_x,                ],  # gooms already are natural logs
        [ 'one_scalar',   'Exponentials',        torch.exp,           torch.exp,                          ],  # keeps GOOMs in complex log-domain
        [ 'two_scalars',  'Scalar Addition',     lambda x, y: x + y,  goom.log_add_exp,                   ],
        [ 'two_scalars',  'Scalar Product',      lambda x, y: x * y,  lambda log_x, log_y: log_x + log_y, ],
        [ 'two_matrices', 'Matrix Product',      torch.matmul,        goom.log_matmul_exp,                ],
    ]

    d = N_DIMS_FOR_TESTED_MATMUL
    peak_allocs = []
    for inp_desc, func_desc, float_func, goom_func in func_metadata:
        print(f'{func_desc.capitalize()}, {goom_title} vs {float_title}...')

        torch.cuda.empty_cache()
        torch.cuda.memory.reset_peak_memory_stats(device=DEVICE)
        match inp_desc:
            case 'one_scalar':
                log_x = goom.log(torch.rand(N_SAMPLES_FOR_TIME_AND_MEM_BENCHMARKS, dtype=dtype, device=DEVICE))
                log_y = goom_func(log_x)
                goom_peak_alloc = torch.cuda.memory.max_memory_allocated(device=DEVICE)
                del log_x, log_y
            case 'two_scalars':
                log_x = goom.log(torch.rand(N_SAMPLES_FOR_TIME_AND_MEM_BENCHMARKS, dtype=dtype, device=DEVICE))
                log_y = goom.log(torch.rand(N_SAMPLES_FOR_TIME_AND_MEM_BENCHMARKS, dtype=dtype, device=DEVICE))
                log_z = goom_func(log_x, log_y)
                goom_peak_alloc = torch.cuda.memory.max_memory_allocated(device=DEVICE)
                del log_x, log_y, log_z
            case 'two_matrices':
                log_X = goom.log(torch.randn(d, d, dtype=goom.config.float_dtype, device=DEVICE))
                log_Y = goom.log(torch.randn(d, d, dtype=goom.config.float_dtype, device=DEVICE))
                log_Z = goom_func(log_X, log_Y)
                goom_peak_alloc = torch.cuda.memory.max_memory_allocated(device=DEVICE)
                del log_X, log_Y, log_Z

        torch.cuda.empty_cache()
        torch.cuda.memory.reset_peak_memory_stats(device=DEVICE)
        match inp_desc:
            case 'one_scalar':
                float_x = torch.rand(N_SAMPLES_FOR_TIME_AND_MEM_BENCHMARKS, dtype=dtype, device=DEVICE)
                float_y = float_func(float_x)
                float_peak_alloc = torch.cuda.memory.max_memory_allocated(device=DEVICE)
                del float_x, float_y
            case 'two_scalars':
                float_x = torch.rand(N_SAMPLES_FOR_TIME_AND_MEM_BENCHMARKS, dtype=dtype, device=DEVICE)
                float_y = torch.rand(N_SAMPLES_FOR_TIME_AND_MEM_BENCHMARKS, dtype=dtype, device=DEVICE)
                float_z = float_func(float_x, float_y)
                float_peak_alloc = torch.cuda.memory.max_memory_allocated(device=DEVICE)
                del float_x, float_y, float_z
            case 'two_matrices':
                float_X = torch.randn(d, d, dtype=goom.config.float_dtype, device=DEVICE)
                float_Y = torch.randn(d, d, dtype=goom.config.float_dtype, device=DEVICE)
                float_Z = float_func(float_X, float_Y)
                float_peak_alloc = torch.cuda.memory.max_memory_allocated(device=DEVICE)
                del float_X, float_Y, float_Z

        peak_allocs.append({
            'func_desc': func_desc,
            'relative_peak_alloc': goom_peak_alloc / float_peak_alloc,
        })

    fig = plot_peak_memory_allocs(peak_allocs, dtype)
    save_and_close_fig(fig, fig_desc=f'{goom_camel}_vs_{float_camel}_peak_memory_allocated')



# Code for computing comparisons, generating figures, and saving them:

for dtype in TORCH_FLOAT_DTYPES_TO_TEST:

    goom_title, float_title = get_goom_and_float_titles(dtype)
    print(f'\n### {goom_title} vs {float_title} ###')

    # Set torch float dtype for comparisons:
    goom.config.float_dtype = dtype

    # Run comparisons:
    compare_errors()
    compare_execution_times()
    compare_memory_allocated()


print(f'\nFinished. All figures have been saved as files named "{FIG_FILENAME_PREFIX}*.png."')