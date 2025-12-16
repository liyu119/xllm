# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
# mypy: ignore-errors

import os
from typing import Optional

import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
import pytest

if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    div = tldevice.fast_dividef
    exp = tldevice.fast_expf
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:

    @triton.jit
    def div_normal(x, y):
        return x / y

    div = div_normal
    exp = tl.exp
    log = tl.log
    log2 = tl.log2


@triton.heuristics({
    'USE_INITIAL_STATE':
    lambda args: args['h0'] is not None,
    'IS_VARLEN':
    lambda args: args['cu_seqlens'] is not None,
    "IS_CONTINUOUS_BATCHING":
    lambda args: args['ssm_state_indices'] is not None,
    "IS_SPEC_DECODING":
    lambda args: args['num_accepted_tokens'] is not None,
})
@triton.jit(do_not_specialize=['N', 'T'])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N,  # num of sequences
    T,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(
            tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        # no tokens to process for this sequence
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            p_h0 = h0 + tl.load(ssm_state_indices + i_n * stride_indices_seq +
                                i_t).to(tl.int64) * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * K * V
        p_h0 = p_h0 + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        p_q = q + (bos * H + i_h) * K + o_k + H * K * i_t
        p_k = k + (bos * H + i_h) * K + o_k + H * K * i_t
        p_v = v + (bos * HV + i_hv) * V + o_v + HV * V * i_t
        if IS_BETA_HEADWISE:
            p_beta = beta + (bos * HV + i_hv) * V + o_v + HV * V * i_t
        else:
            p_beta = beta + bos * HV + i_hv + HV * i_t
        p_g = g + bos * HV + i_hv + HV * i_t
        p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v + HV * V * i_t

        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        # [BK, BV]
        # b_h *= tl.exp(b_g)
        b_h *= exp(b_g)
        # [BV]
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        # [BK, BV]
        b_h += b_k[:, None] * b_v[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # keep the states for multi-query tokens
        if INPLACE_FINAL_STATE:
            p_ht = ht + tl.load(ssm_state_indices + i_n * stride_indices_seq +
                                i_t).to(tl.int64) * stride_final_state_token
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
        p_ht = p_ht + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    o = q.new_empty(NK, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, K, V, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state

class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                g: torch.Tensor,
                beta: torch.Tensor,
                scale: float,
                initial_state: torch.Tensor,
                inplace_final_state: bool = True,
                cu_seqlens: Optional[torch.LongTensor] = None,
                ssm_state_indices: Optional[torch.Tensor] = None,
                num_accepted_tokens: Optional[torch.Tensor] = None,
                use_qk_l2norm_in_kernel: bool = False):
        o, final_state = fused_recurrent_gated_delta_rule_fwd(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g.contiguous(),
            beta=beta.contiguous(),
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

        return o, final_state


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused recurrent gated delta rule.

    Expected shapes:
      - q, k: [B, T, H, K]
      - v:    [B, T, HV, V]
      - g:    [B, T, HV]
      - beta: [B, T, HV]
      - initial_state: [N, HV, K, V]
    """
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            f"Please flatten variable-length inputs before processing.")
    if scale is None:
        scale = k.shape[-1]**-0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        inplace_final_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        use_qk_l2norm_in_kernel,
    )
    return o, final_state

def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

# the golden of fused_recurrent_gated_delta_rule
def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1) # [B, HV, D]
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

# default param is for qwen3-next tp4
@pytest.mark.parametrize("batch, T", [(1, 1), (4, 1), (8, 1), (16, 1)])
def test_recurrent_fused_gated_delta_rule(
    batch: int,
    T: int,
    num_heads: int = 4,
    num_v_heads: int = 8,
    k_head_dim: int = 128,
    v_head_dim: int = 128,
    device: str = "npu:0",
) -> None:
    """Simple accuracy test comparing Triton kernel with golden PyTorch version."""
    torch.manual_seed(0)
    dtype = torch.float16
    L = batch * T
    q = torch.randn(batch, T, num_heads, k_head_dim, dtype=dtype)
    k = torch.randn(batch, T, num_heads, k_head_dim, dtype=dtype)
    v = torch.randn(batch, T, num_v_heads, v_head_dim, dtype=dtype)
    g = torch.randn(batch, T, num_v_heads, dtype=torch.float32)
    beta = torch.randn(batch, T, num_v_heads, dtype=torch.float32)
    initial_state = torch.randn(batch, T, num_v_heads, k_head_dim, v_head_dim, dtype=torch.float32)

    if num_v_heads // num_heads > 1:
        q_ = q.repeat_interleave(num_v_heads // num_heads, dim=2)
        k_ = k.repeat_interleave(num_v_heads // num_heads, dim=2)
    golden_o, golden_state = torch_recurrent_gated_delta_rule(
        q_,
        k_,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    # Triton kernel on target device
    q_d = q.reshape(1, L, num_heads, k_head_dim).to(device)
    k_d = k.reshape(1, L, num_heads, k_head_dim).to(device)
    v_d = v.reshape(1, L, num_v_heads, v_head_dim).to(device)
    g_d = g.reshape(1, L, num_v_heads).to(device)
    beta_d = beta.reshape(1, L, num_v_heads).to(device)
    init_d = initial_state.to(device)
    culen = [i for i in range(0, batch + 1)]
    cu_seqlens = torch.LongTensor(culen).to(device)

    o_d, state_d = fused_recurrent_gated_delta_rule(
        q=q_d,
        k=k_d,
        v=v_d,
        g=g_d,
        beta=beta_d,
        initial_state=init_d,
        cu_seqlens=cu_seqlens,
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )

    o = o_d.cpu()
    state = state_d.cpu()
    o = o.reshape(golden_o.shape)
    assert torch.allclose(golden_o, o, atol=1e-3, rtol=1e-3), "Output mismatch"
    assert torch.allclose(golden_state, state, atol=1e-2, rtol=1e-2), "State mismatch"
    print(f"fused_recurrent_gated_delta_rule: test passed for batch={batch}, T={T}")


if __name__ == "__main__":
    pass
    # test_recurrent_fused_gated_delta_rule(batch=1, T=1)