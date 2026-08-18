# Copyright 2026. Tesseract Hackathon submission.
"""Real GPU benchmark: measures actual forward-pass latency of the Triton
FlashAttention kernel (validation/gpu/triton_flash_attention.py) under the
trained policy's recommended tile size versus a fixed naive tile size, and
reports it alongside the cost model's predicted latency for the same two
configurations.

This is the piece the CPU-only demo in validation/throughput_compare.py
explicitly could not provide: a real, measured, tile-size-driven latency
number from an actual configurable-tile-size GPU kernel. Run
tests/gpu/test_triton_flash_attention.py FIRST to confirm the kernel
itself is correct before trusting anything this script reports.

Requires a CUDA GPU (developed for Google Colab's free T4 tier; see the
README's Phase 5 section for setup steps). Has NOT been executed by the
assistant that wrote it, since no GPU was available in that environment;
run it and report back what happens.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402

from baselines.common import make_client, workload_apply_inputs  # noqa: E402
from policy.model import TilingPolicy  # noqa: E402
from policy.sample_workloads import HARDWARE_PROFILES, TILE_SIZE_MIN, Workload  # noqa: E402
from validation.gpu.triton_flash_attention import flash_attention_forward  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

NAIVE_BLOCK = 32
WARMUP_ITERS = 10
TIMED_ITERS = 50


def load_policy(checkpoint_path: Path) -> TilingPolicy:
    skeleton = TilingPolicy(jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(checkpoint_path, like=skeleton)


def round_to_valid_block_size(x: float) -> int:
    """Round a continuous tile-size prediction to a multiple of 16 (Triton
    block sizes need not be powers of 2, but tensor-core-aligned multiples
    of 16 are the conventional, efficient choice, and match the grid this
    project's grid-search baseline sweeps).
    """
    return max(16, int(round(x / 16.0)) * 16)


def measure_latency_us(fn, *args) -> float | None:
    """Returns mean measured latency in microseconds, or None if this tile
    size does not fit in the GPU's shared memory (a real hardware limit,
    not a bug; see triton_flash_attention.py's num_stages docstring).
    """
    try:
        for _ in range(WARMUP_ITERS):
            fn(*args)
        torch.cuda.synchronize()
    except triton.runtime.errors.OutOfResources:
        return None

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) * 1000.0 / TIMED_ITERS  # ms -> us, averaged


def evaluate_cost_model(client, workload: Workload, br: float, bc: float) -> dict:
    return client.apply(workload_apply_inputs(workload, br, bc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hardware-index", type=int, default=0, help="Index into policy.sample_workloads.HARDWARE_PROFILES")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU visible to torch. This script must be run on a GPU runtime "
            "(e.g. a Colab notebook with Runtime > Change runtime type > GPU). "
            "See the README's Phase 5 section."
        )

    sram, bw, flops, dtype_bytes, hw_name = HARDWARE_PROFILES[args.hardware_index]
    workload = Workload(
        seq_len=float(args.seq_len), head_dim=float(args.head_dim),
        num_heads=float(args.num_heads), batch_size=float(args.batch_size),
        sram_size_bytes=float(sram), hbm_bandwidth_gb_s=float(bw),
        compute_throughput_flops=float(flops), dtype_bytes=float(dtype_bytes),
        hardware_name=hw_name,
    )
    print(f"Workload: seq_len={workload.seq_len:.0f}, head_dim={workload.head_dim:.0f}, "
          f"num_heads={workload.num_heads:.0f}, batch_size={workload.batch_size:.0f}, "
          f"hardware={workload.hardware_name} (SRAM budget only; actual GPU below may differ)")
    print(f"Actual torch CUDA device: {torch.cuda.get_device_name(0)}")

    policy = load_policy(args.checkpoint)
    br_raw, bc_raw = policy(jnp.asarray(workload.to_feature_vector()))
    policy_block_m = round_to_valid_block_size(float(br_raw))
    policy_block_n = max(TILE_SIZE_MIN, round_to_valid_block_size(float(bc_raw)))
    print(f"\nPolicy recommendation: Br={float(br_raw):.2f} -> BLOCK_M={policy_block_m}, "
          f"Bc={float(bc_raw):.2f} -> BLOCK_N={policy_block_n}")
    print(f"Naive baseline: BLOCK_M=BLOCK_N={NAIVE_BLOCK}")

    client = make_client()
    policy_cost = evaluate_cost_model(client, workload, policy_block_m, policy_block_n)
    naive_cost = evaluate_cost_model(client, workload, NAIVE_BLOCK, NAIVE_BLOCK)
    policy_predicted_us = float(policy_cost["predicted_latency_us"])
    naive_predicted_us = float(naive_cost["predicted_latency_us"])
    print(f"\nCost model's own SRAM-feasibility prediction: policy tile sram_utilization="
          f"{float(policy_cost['sram_utilization']):.3f}, naive tile sram_utilization="
          f"{float(naive_cost['sram_utilization']):.3f} (> 1.0 means the cost model predicts "
          f"this tile does not fit in the assumed SRAM budget)")

    torch.manual_seed(0)
    shape = (args.batch_size, args.num_heads, args.seq_len, args.head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)

    policy_measured_us = measure_latency_us(
        flash_attention_forward, q, k, v, policy_block_m, policy_block_n
    )
    naive_measured_us = measure_latency_us(
        flash_attention_forward, q, k, v, NAIVE_BLOCK, NAIVE_BLOCK
    )

    def fmt(x):
        return f"{x:14.3f}" if x is not None else f"{'N/A (OOM)':>14s}"

    print("\n--- Predicted (Tesseract A cost model) vs. measured (real Triton kernel, real GPU) ---")
    print(f"{'':25s} {'predicted us':>14s} {'measured us':>14s}")
    print(f"{'Policy tile':25s} {policy_predicted_us:14.3f} {fmt(policy_measured_us)}")
    print(f"{'Naive tile':25s} {naive_predicted_us:14.3f} {fmt(naive_measured_us)}")

    if policy_measured_us is None or naive_measured_us is None:
        print(
            "\nAt least one tile size did not fit in this GPU's shared memory even with "
            "num_stages=1 (see triton_flash_attention.py). This is itself a real correlation "
            "data point: check whether the cost model's sram_utilization above also predicted "
            "infeasibility (> 1.0) for the same tile -- if so, the cost model's SRAM branch "
            "correctly anticipated a real hardware limit, just a stricter one (this GPU's actual "
            "shared-memory budget) than the SRAM budget this project's hardware profiles assume. "
            "See the README's Limitations section."
        )
    else:
        predicted_speedup = naive_predicted_us / policy_predicted_us
        measured_speedup = naive_measured_us / policy_measured_us
        print(f"\nPredicted speedup (policy vs. naive): {predicted_speedup:.3f}x")
        print(f"Measured speedup (policy vs. naive):   {measured_speedup:.3f}x")
        print(
            "\nThis is the real correlation check the CPU-only demo could not provide: whether the "
            "cost model's predicted speedup direction and rough magnitude are reflected in a real "
            "measured GPU kernel. It is still one workload shape on one GPU, not a validated general "
            "correlation; see the README's Phase 5 section for how to read this result."
        )


if __name__ == "__main__":
    main()
