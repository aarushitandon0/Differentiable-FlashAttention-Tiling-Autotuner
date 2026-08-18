# Copyright 2026. Tesseract Hackathon submission.
"""Phase 5 validation demo: connect the trained tiling policy to a real
transformer's shape, and separately report real measured training
throughput for that transformer.

This script deliberately reports two different kinds of numbers side by
side, and does not conflate them:

  1. Predicted latency from the cost model (Tesseract A), for the trained
     policy's recommended tile size versus a fixed naive tile size, at the
     tiny transformer's actual (seq_len, head_dim, num_heads, batch_size)
     shape. This exercises the real, already-verified cost model and
     policy end to end, on a shape that comes from an actual model rather
     than a synthetic one.
  2. Real measured tokens-per-second from actually training that
     transformer for a few steps on CPU (validation/tiny_transformer.py),
     using a standard dense attention implementation.

These two are NOT causally connected here: the tiny transformer's attention
implementation is a plain JAX einsum with no tile-size argument, since
tiling is a GPU kernel implementation detail that a high-level dense
attention op does not expose. Changing the policy's recommended tile size
would not change the real measured number in (2) at all, on this
implementation, on CPU. Reporting them together is meant to show that the
full pipeline (workload shape -> trained policy -> tile-size recommendation
-> cost model prediction) runs correctly against a real model's shape, and
that a real transformer trains correctly with this project's stack, not to
claim a validated real-hardware speedup. See the README's "Phase 5" section
for the reasoning behind this scope, and what a real GPU validation would
additionally require.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from baselines.common import evaluate_latency, make_client  # noqa: E402
from policy.model import TilingPolicy  # noqa: E402
from policy.sample_workloads import HARDWARE_PROFILES, Workload  # noqa: E402
from validation.tiny_transformer import GPTConfig, train  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

NAIVE_BR = 32.0
NAIVE_BC = 32.0


def load_policy(checkpoint_path: Path) -> TilingPolicy:
    skeleton = TilingPolicy(jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(checkpoint_path, like=skeleton)


def workload_for_tiny_transformer(config: GPTConfig, batch_size: int, hardware_index: int = 0) -> Workload:
    sram, bw, flops, dtype_bytes, name = HARDWARE_PROFILES[hardware_index]
    return Workload(
        seq_len=float(config.seq_len),
        head_dim=float(config.head_dim),
        num_heads=float(config.num_heads),
        batch_size=float(batch_size),
        sram_size_bytes=float(sram),
        hbm_bandwidth_gb_s=float(bw),
        compute_throughput_flops=float(flops),
        dtype_bytes=float(dtype_bytes),
        hardware_name=name,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--hardware-index", type=int, default=0, help="Index into policy.sample_workloads.HARDWARE_PROFILES")
    args = parser.parse_args()

    config = GPTConfig(
        vocab_size=0,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    # vocab_size is only known once the dataset is loaded; not needed for
    # the workload shape (head_dim, num_heads, seq_len, batch_size) below.

    workload = workload_for_tiny_transformer(config, args.batch_size, args.hardware_index)
    print(f"Tiny transformer shape: seq_len={workload.seq_len:.0f}, head_dim={workload.head_dim:.0f}, "
          f"num_heads={workload.num_heads:.0f}, batch_size={workload.batch_size:.0f}, "
          f"hardware={workload.hardware_name}")

    # --- 1. Cost-model prediction: policy's recommendation vs. fixed naive tiling ---
    client = make_client()
    policy = load_policy(args.checkpoint)
    br, bc = policy(jnp.asarray(workload.to_feature_vector()))
    br, bc = float(br), float(bc)

    policy_latency = evaluate_latency(client, workload, br, bc)
    naive_latency = evaluate_latency(client, workload, NAIVE_BR, NAIVE_BC)

    print("\n--- Cost-model prediction (Tesseract A), not measured on real hardware ---")
    print(f"Policy-recommended tile:  Br={br:.2f}, Bc={bc:.2f} -> predicted latency {policy_latency:.4f} us")
    print(f"Fixed naive tile:         Br={NAIVE_BR:.0f}, Bc={NAIVE_BC:.0f} -> predicted latency {naive_latency:.4f} us")

    # --- 2. Real measured training throughput, independent of tile size ---
    print("\n--- Real measured training run (validation/tiny_transformer.py, CPU, standard dense attention) ---")
    _, loss_history, seconds_per_step, tokens_per_second = train(
        config, steps=args.train_steps, batch_size=args.batch_size
    )
    print(f"Loss: {loss_history[0]:.4f} -> {loss_history[-1]:.4f} over {args.train_steps} steps")
    print(f"Measured: {seconds_per_step * 1000:.2f} ms/step, {tokens_per_second:.0f} tokens/sec")

    print(
        "\nNote: the measured throughput above does not vary with the tile size reported above it. "
        "This CPU run uses a standard dense attention implementation with no tile-size argument; "
        "the cost-model prediction and the real measured throughput are reported together to show "
        "the full pipeline runs correctly end to end against a real model's shape, not to claim a "
        "validated tile-size-driven speedup. See the README's Phase 5 section."
    )


if __name__ == "__main__":
    main()
