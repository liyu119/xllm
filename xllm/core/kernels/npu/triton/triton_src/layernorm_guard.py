# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This backward pass is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.
# mypy: ignore-errors

import torch
# from vllm.triton_utils import tl, triton
import torch
import triton
import torch_npu
import triton.language as tl
import torch.nn.functional as F
from typing import Optional, Tuple
import pytest

MAX_CORES = 65535


@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None,
    "HAS_Z": lambda args: args["Z"] is not None,
})
@triton.jit
def layer_norm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X_base
    N,  # number of columns in X_base
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    N_CORES: tl.constexpr,
):
    # Map the program id to the row of X_base and Y_base it should compute.
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    n_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    if row < remain:
        n_iters = n_iters + 1

    for i in tl.range(n_iters):
        X_base = X + (i * BLOCK_ROWS *
                      stride_x_row) + row * stride_x_row + group * N
        Y_base = Y + (i * BLOCK_ROWS *
                      stride_y_row) + row * stride_y_row + group * N
        if HAS_Z:
            Z_base = Z + (i * BLOCK_ROWS *
                          stride_z_row) + row * stride_z_row + group * N
        if not IS_RMS_NORM:
            Mean_base = Mean + (i * BLOCK_ROWS) + group * M
        Rstd_base = Rstd + (i * BLOCK_ROWS) + group * M
        W_base = W + group * N
        if HAS_BIAS:
            B_base = B + group * N
        # Compute mean and variance
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
        # Normalize and apply linear transformation
        mask = cols < N
        w = tl.load(W_base + cols, mask=mask).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(B_base + cols, mask=mask).to(tl.float32)
        x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        y = x_hat * w + b if HAS_BIAS else x_hat * w
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        # Write output
        tl.store(Y_base + cols, y, mask=mask)


def _layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
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
    assert weight.shape == (N, )
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N, )
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = (torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device)
            if not is_rms_norm else None)
    rstd = torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M if M < MAX_CORES else MAX_CORES, ngroups)
    with torch.npu.device(x.device.index):
        layer_norm_fwd_kernel[grid](
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_N=BLOCK_N,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            N_CORES=MAX_CORES,
            num_warps=num_warps,
        )
    return out, mean, rstd


class LayerNormFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        z=None,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=False,
    ):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""

        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if z is not None:
            assert z.shape == x_shape_og
            z = z.reshape(-1, z.shape[-1])
            if z.stride(-1) != 1:
                z = z.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y, mean, rstd = _layer_norm_fwd(
            x,
            weight,
            bias,
            eps,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
        return y.reshape(x_shape_og)
    
def custom_layer_norm(x, weight, bias=None, eps=1e-6, z=None, group_size=None, norm_before_gate=True, is_rms_norm=False):
    return LayerNormFn.apply(x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm)

def layer_norm_golden_cpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    z: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False
) -> torch.Tensor:
    """
    基于 PyTorch 原生函数的 CPU 参考实现（Golden）
    完全对齐自定义 LayerNorm 的功能，支持分组归一化、RMSNorm、Z 分支门控等所有特性
    """
    # 1. 输入预处理（和自定义核一致：展平为 2D、确保 CPU 执行）
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1]).cpu().contiguous()  # (M, N) = (batch*seq_len, feat_dim)
    M, N = x.shape
    
    # 2. 辅助变量初始化（确保设备/类型一致）
    weight = weight.cpu().contiguous()
    bias = bias.cpu().contiguous() if bias is not None else None
    z = z.reshape(-1, z.shape[-1]).cpu().contiguous() if z is not None else None
    if z is not None:
        assert z.shape == (M, N), f"Z shape {z.shape} must match X shape ({M}, {N})"
    
    # 3. 分组配置（和自定义核逻辑一致）
    if group_size is None:
        group_size = N
    assert N % group_size == 0, f"N={N} must be divisible by group_size={group_size}"
    ngroups = N // group_size
    
    # 4. 门控逻辑（若有 Z 且后置门控：先应用 z * sigmoid(z)）
    if z is not None and not norm_before_gate:
        gate = z * torch.sigmoid(z)  # (M, N)
        x = x * gate
    
    # 5. 分组归一化（用 unfold 拆分分组，避免手动循环）
    # 拆分：(M, N) → (M, ngroups, group_size) → (M*ngroups, group_size)（适配 torch.layer_norm）
    x_grouped = x.unfold(dimension=1, size=group_size, step=group_size)  # (M, ngroups, group_size)
    
    x_grouped = x_grouped.reshape(-1, group_size)  # (M*ngroups, group_size)
    
    # 6. 核心归一化（LayerNorm vs RMSNorm）
    if not is_rms_norm:
        # 标准 LayerNorm：均值中心化 + 方差归一化（复用 PyTorch 原生实现）
        x_norm = torch.layer_norm(
            x_grouped, 
            normalized_shape=(group_size,), 
            weight=None,  # 后续统一应用权重偏置，保持和自定义核一致
            bias=None,
            eps=eps
        )
    else:
        # RMSNorm：无中心化，仅均方根归一化（x / sqrt(mean(x²) + eps)）
        mean_sq = torch.mean(x_grouped ** 2, dim=-1, keepdim=True)  # (M*ngroups, 1)
        x_norm = x_grouped * torch.rsqrt(mean_sq + eps)  # rsqrt = 1/sqrt
    
    # 7. 还原分组形状：(M*ngroups, group_size) → (M, ngroups, group_size) → (M, N)
    x_norm = x_norm.reshape(M, ngroups, group_size)  # 还原分组维度
    x_norm = x_norm.contiguous().view(M, N)  # 合并分组 → (M, N)
    
    # 8. 线性变换（权重 + 偏置，和自定义核逻辑一致）
    y = x_norm * weight  # (M, N) * (N,) → 广播相乘
    if bias is not None:
        y = y + bias
    
    # 9. 门控逻辑（若有 Z 且前置门控：归一化后应用 z * sigmoid(z)）
    if z is not None and norm_before_gate:
        gate = z * torch.sigmoid(z)  # (M, N)
        y = y * gate
    
    # 10. 恢复原始形状并返回
    return y.reshape(x_shape_og)

def test_custom_layer_norm():
    """
    测试用例：覆盖关键场景，验证数值正确性
    场景：不同维度、有无偏置、有无 Z 分支、门控顺序、LayerNorm/RMSNorm、分组归一化
    """
    # 测试配置（覆盖所有关键参数组合）
    test_cases = [
        # (batch_size, seq_len, feat_dim, has_bias, has_z, norm_before_gate, is_rms_norm, group_size)
        (2, 8, 128, False, False, True, False, None),    # 基础 LayerNorm（无偏置、无 Z）
        (2, 8, 128, True, False, True, False, None),     # LayerNorm + 偏置
        (2, 8, 128, True, True, True, False, None),      # LayerNorm + 偏置 + Z（前置门控）
        (2, 8, 128, True, True, False, False, None),     # LayerNorm + 偏置 + Z（后置门控）
        (2, 8, 128, False, True, True, True, None),      # RMSNorm + Z（前置门控）
        (4, 16, 256, True, False, True, False, 64),      # 分组 LayerNorm（group_size=64）
        (1, 4, 64, False, True, False, True, 32),        # 分组 RMSNorm + Z（后置门控）
    ]
    
    # 数值验证阈值（GPU 浮点精度差异可接受范围）
    atol = 1e-4  # 绝对误差
    rtol = 1e-3  # 相对误差
    
    # 检查是否有 npu 设备
    has_npu = torch.npu.is_available()
    device = torch.device("npu" if torch.npu.is_available() else "cpu")

    for idx, (batch_size, seq_len, feat_dim, has_bias, has_z, norm_before_gate, is_rms_norm, group_size) in enumerate(test_cases):
        print(f"\n测试用例 {idx+1}:")
        print(f"配置: batch={batch_size}, seq={seq_len}, feat={feat_dim}, bias={has_bias}, Z={has_z}, "
              f"norm_before_gate={norm_before_gate}, RMS={is_rms_norm}, group_size={group_size}")
        
        # 1. 生成随机输入（CPU 生成后移至 npu）
        torch.manual_seed(42)  # 固定种子，确保可复现
        x = torch.randn(batch_size, seq_len, feat_dim, dtype=torch.float32)  # (B, S, D)
        weight = torch.randn(feat_dim, dtype=torch.float32)  # 权重（必须有）
        bias = torch.randn(feat_dim, dtype=torch.float32) if has_bias else None
        z = torch.randn(batch_size, seq_len, feat_dim, dtype=torch.float32) if has_z else None
        eps = 1e-6
        
        # 2. 计算 CPU Golden 输出
        golden_output = layer_norm_golden_cpu(
            x=x, weight=weight, bias=bias, eps=eps, z=z,
            group_size=group_size, norm_before_gate=norm_before_gate, is_rms_norm=is_rms_norm
        )
        
        # 3. 计算自定义核输出（仅当有 npu 时）
        if has_npu:
            x_npu = x.npu()
            weight_npu = weight.npu()
            bias_npu = bias.npu() if bias is not None else None
            z_npu = z.npu() if z is not None else None
            
            custom_output = custom_layer_norm(
                x=x_npu, weight=weight_npu, bias=bias_npu, eps=eps, z=z_npu,
                group_size=group_size, norm_before_gate=norm_before_gate, is_rms_norm=is_rms_norm
            ).cpu()  # 移回 CPU 对比
            
            # 4. 数值验证
            max_abs_err = torch.max(torch.abs(custom_output - golden_output))
            max_rel_err = torch.max(torch.abs(custom_output - golden_output) / (torch.abs(golden_output) + eps))
            
            print(f"  最大绝对误差: {max_abs_err:.6f} (阈值: {atol})")
            print(f"  最大相对误差: {max_rel_err:.6f} (阈值: {rtol})")
            
            # 断言验证（误差在可接受范围内）
            assert max_abs_err < atol, f"绝对误差超标: {max_abs_err:.6f} > {atol}"
            assert max_rel_err < rtol, f"相对误差超标: {max_rel_err:.6f} > {rtol}"
            print("  ✅ 数值验证通过！")
            
    print("\n" + "="*50)
    print("所有测试用例执行完成！")
if __name__ == "__main__":
    test_custom_layer_norm()


