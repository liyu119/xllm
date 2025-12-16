import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import pytest

MAX_CORES = 65535

@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None,
    "HAS_Z": lambda args: args["Z"] is not None,
})
@triton.jit
def layer_norm_fwd_kernel(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row, stride_y_row, stride_z_row,
    M, N, eps,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    N_CORES: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    n_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    if row < remain:
        n_iters = n_iters + 1

    for i in tl.range(n_iters):
        X_base = X + (i * BLOCK_ROWS * stride_x_row) + row * stride_x_row + group * N
        Y_base = Y + (i * BLOCK_ROWS * stride_y_row) + row * stride_y_row + group * N
        if HAS_Z:
            Z_base = Z + (i * BLOCK_ROWS * stride_z_row) + row * stride_z_row + group * N
        if not IS_RMS_NORM:
            Mean_base = Mean + (i * BLOCK_ROWS) + group * M
        Rstd_base = Rstd + (i * BLOCK_ROWS) + group * M
        W_base = W + group * N
        if HAS_BIAS:
            B_base = B + group * N
        
        cols = tl.arange(0, BLOCK_N)
        x = tl.load(X_base + cols, mask=cols < N, other=0.).to(tl.float32)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=cols < N).to(tl.float32)
            x *= z * tl.sigmoid(z)
        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=0) / N
            tl.store(Mean_base + row, mean)
            xbar = tl.where(cols < N, x - mean, 0.)
            var = tl.sum(xbar * xbar, axis=0) / N
        else:
            xbar = tl.where(cols < N, x, 0.)
            var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd_base + row, rstd)
        
        mask = cols < N
        w = tl.load(W_base + cols, mask=mask).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(B_base + cols, mask=mask).to(tl.float32)
        x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        y = x_hat * w + b if HAS_BIAS else x_hat * w
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        tl.store(Y_base + cols, y, mask=mask)

def _layer_norm_fwd(
    x, weight, bias, eps, z=None, out=None, group_size=None,
    norm_before_gate=True, is_rms_norm=False
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = (torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
            if not is_rms_norm else None)
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("Feature dim too large for fused kernel")
    
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M if M < MAX_CORES else MAX_CORES, ngroups)
    with torch.npu.device(x.device.index):
        layer_norm_fwd_kernel[grid](
            x, out, weight, bias, z, mean, rstd,
            x.stride(0), out.stride(0), z.stride(0) if z is not None else 0,
            M, group_size, eps,
            BLOCK_N=BLOCK_N,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            N_CORES=MAX_CORES,
            num_warps=num_warps,
        )
    return out, mean, rstd

class RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, group_size=None, norm_before_gate=False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate

    def forward(self, x, z=None):
        x_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if not x.is_contiguous():
            x = x.contiguous()
        
        if z is not None:
            z = z.view_as(x)
            if not z.is_contiguous():
                z = z.contiguous()
        
        y, _, _ = _layer_norm_fwd(
            x,
            self.weight,
            self.bias,
            self.eps,
            z=z,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            is_rms_norm=True
        )
        return y.view(x_shape)

class GoldenRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)

# default param hidden_size = head_v_dim = 128
@pytest.mark.parametrize("num_tokens", [1, 3, 7, 16])
def test_rmsnorm_gated(num_tokens, hidden_size=128, eps=1e-5):
    norm = RMSNormGated(hidden_size=hidden_size, eps=eps, norm_before_gate=True)
    norm_golden = GoldenRMSNormGated(hidden_size=hidden_size, eps=eps)
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16)
    gate = torch.randn(num_tokens, hidden_size, dtype=torch.float16)
    hidden_states_golden = norm_golden(hidden_states, gate)
    hidden_states_npu = norm(hidden_states, gate)
    assert torch.allclose(hidden_states_golden, hidden_states_npu, atol=1e-3, rtol=1e-3)
    print(f"test_rmsnorm_gated: test passed for num_tokens={num_tokens}")