# coding: utf-8

import math
import torch

from dataclasses import dataclass


# Global configuration (modifiable after importing this file):

@dataclass
class Config:

    keep_logs_finite: bool = True
    "If True, log() always returns finite values. Otherwise, it does not."

    cast_all_logs_to_complex: bool = True
    "If True, log() always returns complex tensors. Otherwise, it does not."

    float_dtype: torch.dtype = torch.float32
    "Float dtype of real logarithms and components of complex logarithms."

config = Config()


# Helper functions for elementwise log(abs()) and exp():

class _CustomizedTorchAbs(torch.autograd.Function):
    """
    Applies torch.abs(), but with derivatives that are -1 for negative input
    values or 1 for non-negative ones, including at zero, for backpropagation.
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(inp):
        out = torch.abs(inp)
        return out

    @staticmethod
    def setup_context(ctx, inp_tup, out):
        inp, = inp_tup
        ctx.save_for_backward(inp)

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        return grad_output * torch.where(inp < 0, -1, 1)


class _CustomizedTorchLog(torch.autograd.Function):
    """
    Applies torch.log(), but with derivatives that are always finite for
    backpropagation, and, if specified in config, keeping outputs finite.
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(inp):
        log_inp = torch.log(inp)
        snn = torch.finfo(config.float_dtype).smallest_normal
        finite_floor = math.log(snn) * 2  # exps to zero in float_dtype
        keep_finite_idx = (log_inp < finite_floor) & config.keep_logs_finite
        out = torch.where(keep_finite_idx, finite_floor, log_inp)
        return out

    @staticmethod
    def setup_context(ctx, inp_tup, out):
        inp, = inp_tup
        ctx.save_for_backward(inp)

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        eps = torch.finfo(inp.dtype).eps
        return grad_output / (inp + eps)


class _CustomizedTorchExp(torch.autograd.Function):
    """
    Applies torch.exp(), but with derivatives that are always non-zero for
    backpropagation. Works with both float and complex input tensors.
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(inp):
        out = torch.exp(inp)
        return out

    @staticmethod
    def setup_context(ctx, inp_tup, out):
        ctx.save_for_backward(out)

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        eps = torch.finfo(out.real.dtype).eps
        signed_eps = torch.where(out.real < 0, -eps, eps)
        return grad_output * (out + signed_eps)


# Functions for computing elementwise log() and exp():

def log(x):
    "Elementwise log() of real tensor, i.e., a generalized order of magnitude."
    assert not x.is_complex(), "Input must be a float tensor, not a complex one."
    abs_x = _CustomizedTorchAbs.apply(x)
    log_abs_x = _CustomizedTorchLog.apply(abs_x)
    real = log_abs_x.to(config.float_dtype)
    x_is_neg = (x < 0)
    if torch.any(x_is_neg) or config.cast_all_logs_to_complex:
        return torch.complex(real, imag=x_is_neg.to(real.dtype) * torch.pi)
    else:
        return real

def exp(log_x):
    "Elementwise exp() of a tensor with generalized orders of magnitude."
    return _CustomizedTorchExp.apply(log_x).real


# Functions for computing log-matmul-exp:

def log_matmul_exp(log_x1, log_x2, min_c=0):
    """
    Broadcastable log(exp(log_x1) @ exp(log_x2)), implemented by calling
    PyTorch's existing implementation of matmul over scaled float tensors.
    Inputs:
        log_x1: log-tensor of shape [..., d1, d2].
        log_x2: log-tensor of shape [..., d2, d3].
        min_c: (optional) float, minimum log-scaling constant. Default: 0.
    Outputs:
        log_y: log-tensor of shape [..., d1, d3].
    """
    c1 = log_x1.real.detach().max(dim=-1, keepdim=True).values.clamp(min=min_c)
    c2 = log_x2.real.detach().max(dim=-2, keepdim=True).values.clamp(min=min_c)
    scaled_y = torch.matmul(exp(log_x1 - c1), exp(log_x2 - c2))
    log_y = log(scaled_y) + c1 + c2
    return log_y

def alternate_log_matmul_exp(log_x1, log_x2, chunk_size=32):
    """
    Broadcastable log(exp(log_x1) @ exp(log_x2)), implemented by composition
    of vmapped log-sum-exp-of-sum operations. Much slower, but more precise.
    Inputs:
        log_x1: log-tensor of shape [..., d1, d2].
        log_x2: log-tensor of shape [..., d2, d3].
        chunk_size: (optional) int, chunk size for torch.vmap. Default: 32.
    Outputs:
        log_y: log-tensor of shape [..., d1, d3].
    """
    # Get log-scaling constants:
    c1 = log_x1.real.detach().max(dim=-1, keepdim=True).values
    c2 = log_x2.real.detach().max(dim=-2, keepdim=True).values

    # Get log-scaled operands:
    log_s1 = log_x1 - c1
    log_s2 = log_x2 - c2

    # Broadcast preceding dims and flatten them into a single dim:
    d1, d2, d3 = (*log_s1.shape[-2:], log_s2.shape[-1])
    broadcast_szs = torch.broadcast_shapes(log_s1.shape[:-2], log_s2.shape[:-2])
    log_s1 = log_s1.expand(*broadcast_szs, d1, d2).view(-1, d1, d2)                # [<flattened dim>, d1, d2]
    log_s2 = log_s2.expand(*broadcast_szs, d2, d3).view(-1, d2, d3)                # [<flattened dim>, d2, d3]

    # Define vmapped sum-exp-of-outer-sum operations:
    _vve = lambda row_vec, col_vec: exp(row_vec + col_vec).sum()                   # vec, vec -> scalar
    _mve = torch.vmap(_vve, in_dims=(0, None), out_dims=0, chunk_size=chunk_size)  # mat, vec -> vec
    _mme = torch.vmap(_mve, in_dims=(None, 1), out_dims=1, chunk_size=chunk_size)  # mat, mat -> mat
    _multi_mme = torch.vmap(_mme, chunk_size=chunk_size)                           # n mats, n mats -> n mats

    # Compute, reshape, and return result:
    log_y = log(_multi_mme(log_s1, log_s2)) + c1 + c2                              # [<flattened dims>, d1, d3]
    return log_y.view(*broadcast_szs, d1, d3)                                      # [..., d1, d3]


# Functions for computing log-sums and log-means of exponentials:

def log_add_exp(log_x1, log_x2):
    "Elementwise log(exp(log_x1) + exp(log_x2))."
    if log_x1.is_complex() or log_x2.is_complex():
        c = max(log_x1.real.detach().max(), log_x2.real.detach().max())
        x = exp(log_x1 - c) + exp(log_x2 - c)
        return log(x) + c
    else:
        return torch.logaddexp(log_x1, log_x2)

def log_sum_exp(log_x, dim):
    "Log-sum-exp over dim of log_x."
    if log_x.is_complex():
        c = log_x.real.detach().max(dim=dim, keepdim=True).values
        x = exp(log_x - c).sum(dim=dim)
        return log(x) + c.squeeze(dim)
    else:
        return torch.logsumexp(log_x, dim=dim)

def log_mean_exp(log_x, dim):
    "Log-mean-exp over dim of log_x."
    log_n = log_x.real.new_tensor(float(log_x.size(dim))).log()
    return log_sum_exp(log_x, dim=dim) - log_n


# Functions for computing cumulative log-sums and log-means of exponentials:

def log_cum_sum_exp(log_x, dim):
    "Log-cumulative-sum-exp over dim of log_x."
    if log_x.is_complex():
        c = log_x.real.detach().max(dim=dim, keepdim=True).values
        x = exp(log_x - c).cumsum(dim=dim)
        return log(x) + c
    else:
        return torch.logcumsumexp(log_x, dim=dim)

def log_cum_mean_exp(log_x, dim):
    "Log-cumulative-mean-exp over dim of log_x."
    log_x = log_x.movedim(dim, -1)
    log_n = torch.arange(1, log_x.size(-1) + 1, dtype=log_x.real.dtype, device=log_x.device).log()
    log_x = log_cum_sum_exp(log_x, dim=-1) - log_n
    return log_x.movedim(-1, dim)


# Convenience functions:

def log_negate_exp(log_x):
    "Elementwise negate log_x, a generalized order of magnitude."
    if log_x.is_complex():
        return log_x + torch.complex(log_x.real.new([0.0]), log_x.imag.new([torch.pi]))
    else:
        return torch.complex(log_x.real, torch.full_like(log_x.real, torch.pi))

def scale(log_x, dim, max_real=2):
    "Scale log_x so max real component over dim is max_real."
    c = log_x.real.detach().max(dim, keepdim=True).values
    return log_x - c + max_real

def scaled_exp(log_x, dim, max_real=2):
    "Scale log_x so max real component over dim is max_real, and exponentiate."
    return exp(scale(log_x, dim, max_real))

def log_triu_exp(log_x, diagonal_offset=0):
    "Mask log_x in last two dims so its exp() is an upper-triangular matrix."
    keep_idx = log_x.real.new_ones(log_x.shape[-2:]).triu(diagonal_offset).bool()
    log_y = log_x.clone()
    log_y.real.masked_fill_(~keep_idx, float('-inf'))
    return log_y

def log_tril_exp(log_x, diagonal_offset=0):
    "Mask log_x in last two dims so its exp() is a lower-triangular matrix."
    keep_idx = log_x.real.new_ones(log_x.shape[-2:]).tril(diagonal_offset).bool()
    log_y = log_x.clone()
    log_y.real.masked_fill_(~keep_idx, float('-inf'))
    return log_y

def log_rmsnorm_exp(log_x, dim):
    "Scale log_x over dim so exp() has unit root mean squared elements."
    log_d = math.log(float(log_x.size(dim)))
    log_x = log_x - 0.5 * (log_sum_exp(2 * log_x, dim=dim).unsqueeze(dim) - log_d)
    return log_x

def log_unitsquash_exp(log_x):
    "Elementwise squash log_x so exp() is between -1 and 1."
    assert log_x.is_complex(), "Input must be a complex tensor, not a float one."
    squashed_real = torch.nn.functional.logsigmoid(log_x.real)
    return torch.complex(squashed_real, log_x.imag)
