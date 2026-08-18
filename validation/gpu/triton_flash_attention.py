# Copyright 2026. Tesseract Hackathon submission.
"""A minimal causal FlashAttention forward kernel in Triton, with block
sizes (BLOCK_M, BLOCK_N) exposed as real, configurable kernel launch
parameters, matching Br and Bc in tesseracts/attention-cost-model. This is
the missing piece the CPU-only demo in validation/tiny_transformer.py
does not have: a plain JAX einsum has no tile-size argument at all, since
tiling is a GPU kernel implementation detail. This module requires a CUDA
GPU and `torch` + `triton` installed; it cannot run on the CPU-only
environment the rest of this project was developed and verified in (see
the README's Setup and Phase 5 sections), which is why it lives in its own
`validation/gpu/` subdirectory with its own dependency list
(requirements-gpu.txt) instead of the top-level requirements.txt.

Forward pass only, deliberately: the cost model in
tesseracts/attention-cost-model only models forward-pass HBM traffic and
compute, not a backward pass, so a forward-only benchmark is what actually
corresponds to what the cost model claims to predict. A full trainable
version would additionally need a matching backward kernel, which is a
substantially larger undertaking and out of scope here.

Adapted from the block structure of Triton's own FlashAttention tutorial
(online softmax, one query block per program, looping over key/value
blocks), trimmed to the essentials needed for this benchmark.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out,
    stride_qh, stride_qm, stride_qd,
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_oh, stride_om, stride_od,
    seq_len, head_dim,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)  # flattened (batch, head) index

    Q += off_bh * stride_qh
    K += off_bh * stride_kh
    V += off_bh * stride_vh
    Out += off_bh * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d[None, :] < head_dim

    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len) & d_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Causal: query block start_m only ever attends to key blocks up to
    # and including its own position, so the loop can stop early.
    end_n = (start_m + 1) * BLOCK_M
    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n[:, None] < seq_len

        k_ptrs = K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask & d_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask & d_mask, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    acc = acc / l_i[:, None]
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < seq_len) & d_mask)


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_m: int,
    block_n: int,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Causal FlashAttention forward pass with explicit tile sizes.

    q, k, v: (batch, num_heads, seq_len, head_dim), same shape, CUDA, fp16
    or bf16. Returns a tensor of the same shape.
    """
    assert q.shape == k.shape == v.shape
    assert q.is_cuda, "This kernel requires a CUDA tensor; see the module docstring."
    batch, num_heads, seq_len, head_dim = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    out = torch.empty_like(q)
    block_d = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(seq_len, block_m), batch * num_heads)

    _flash_attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        out.stride(1), out.stride(2), out.stride(3),
        seq_len, head_dim, sm_scale,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d,
    )
    return out


def reference_attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Plain PyTorch causal attention, for correctness-checking the Triton
    kernel above (see tests/gpu/test_triton_flash_attention.py).
    """
    head_dim = q.shape[-1]
    sm_scale = 1.0 / (head_dim ** 0.5)
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * sm_scale
    seq_len = q.shape[2]
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.einsum("bhqk,bhkd->bhqd", probs, v)
