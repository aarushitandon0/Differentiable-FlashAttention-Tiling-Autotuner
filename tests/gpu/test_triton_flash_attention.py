# Copyright 2026. Tesseract Hackathon submission.
"""Correctness test for validation/gpu/triton_flash_attention.py.

Run this FIRST on Colab (or any CUDA machine), before trusting anything
validation/gpu/benchmark.py reports: it checks the Triton kernel's output
against a plain PyTorch reference implementation across a few different
(Br, Bc) tile size choices, including tile sizes that do not evenly divide
seq_len, since that is exactly the kind of edge case a tiling bug tends to
show up in.

Requires a CUDA GPU; skipped automatically otherwise. Not part of the main
`pytest tests/` run for that reason, see validation/gpu/requirements-gpu.txt
and the README's Phase 5 section for how to run this on Colab.
"""

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from validation.gpu.triton_flash_attention import (  # noqa: E402
    flash_attention_forward,
    reference_attention_forward,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


@pytest.mark.parametrize("seq_len,block_m,block_n", [
    (128, 32, 32),
    (128, 64, 16),
    (100, 32, 32),   # seq_len not a multiple of block_m/block_n
    (64, 128, 128),  # block sizes larger than seq_len (the Br > seq_len edge case)
])
def test_triton_matches_reference(seq_len, block_m, block_n):
    torch.manual_seed(0)
    batch, num_heads, head_dim = 2, 4, 32
    q = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

    try:
        out_triton = flash_attention_forward(q, k, v, block_m=block_m, block_n=block_n)
    except triton.runtime.errors.OutOfResources as e:
        # A real hardware shared-memory ceiling, not a correctness bug: this
        # GPU's per-block shared memory is smaller than what this tile size
        # needs even with num_stages=1 (no software pipelining). This is
        # informative in its own right -- it is the literal GPU-enforced
        # version of the SRAM-feasibility branch tesseracts/attention-cost-model
        # models in software -- so it is reported as an expected failure
        # rather than a hard test failure.
        pytest.xfail(f"Tile size ({block_m}, {block_n}) exceeds this GPU's shared memory: {e}")
        return

    out_ref = reference_attention_forward(q, k, v)

    torch.testing.assert_close(out_triton, out_ref, atol=2e-2, rtol=2e-2)
