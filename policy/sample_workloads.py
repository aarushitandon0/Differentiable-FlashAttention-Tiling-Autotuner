# Copyright 2026. Tesseract Hackathon submission.
"""Sampler for a realistic distribution of transformer attention workload shapes.

Grounded in real model families rather than arbitrary ranges:
  - head_dim in {64, 96, 128}: GPT-2/GPT-3 use 64 (e.g. 12-head d_model=768 ->
    head_dim=64) and 128 (e.g. GPT-3 175B: 96 heads x 128); Llama-2/3 7B-70B
    use 128; some mid-size models (e.g. GPT-NeoX-20B) use 96.
  - num_heads in {8, 16, 32}: covers small (8, e.g. GPT-2 small-ish /
    Mistral-7B-class with GQA query heads), medium (16, e.g. GPT-2 XL /
    Llama-2 13B-class) and large (32, e.g. Llama-2 7B/70B, GPT-NeoX-20B)
    configurations.
  - Two regimes reflecting real LLM serving:
      * decode-like: seq_len in [1, 32], batch_size large (serving many
        concurrent single-token decode steps).
      * prefill-like: seq_len in [512, 8192], batch_size small (one or a
        few long prompts being prefilled/trained on).
  - Hardware descriptors are sampled from a small set of real accelerator
    roofline numbers (A100, H100, and a mid-range consumer/edge-ish GPU) so
    the policy also has to generalize across hardware, not just workload
    shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fixed seed for the held-out evaluation set, deliberately different from
# whatever seed a training run uses, so held-out shapes are never part of
# the training RNG stream (see held_out_workloads / baselines/compare.py).
HELD_OUT_SEED = 20260816

HEAD_DIMS = (64, 96, 128)
NUM_HEADS = (8, 16, 32)

# (sram_size_bytes, hbm_bandwidth_gb_s, compute_throughput_flops, dtype_bytes, name)
# SRAM: per-SM shared memory/L1 budget available to a kernel (conservative,
# leaves room for other allocations). Compute throughput is bf16 dense
# matmul peak.
HARDWARE_PROFILES = (
    (164 * 1024, 2039.0, 312e12, 2.0, "A100-80GB"),
    (228 * 1024, 3350.0, 989e12, 2.0, "H100-80GB-SXM"),
    (100 * 1024, 760.0, 165e12, 2.0, "L4-24GB"),
)

# Valid tile-size range the policy is allowed to predict into (matches the
# powers-of-2 grid the Triton-autotuning baseline searches, see
# baselines/grid_search.py).
TILE_SIZE_MIN = 16.0
TILE_SIZE_MAX = 128.0


@dataclass(frozen=True)
class Workload:
    seq_len: float
    head_dim: float
    num_heads: float
    batch_size: float
    sram_size_bytes: float
    hbm_bandwidth_gb_s: float
    compute_throughput_flops: float
    dtype_bytes: float
    hardware_name: str

    def to_feature_vector(self) -> np.ndarray:
        """Normalized features for the policy MLP's input.

        Sizes are log-scaled (they span orders of magnitude: seq_len from 1
        to 8192, compute throughput from 1e14 to 1e15) so the MLP sees
        comparable-magnitude inputs.
        """
        return np.array(
            [
                np.log2(self.seq_len),
                np.log2(self.head_dim),
                np.log2(self.num_heads),
                np.log2(self.batch_size),
                np.log2(self.sram_size_bytes),
                np.log2(self.hbm_bandwidth_gb_s),
                np.log2(self.compute_throughput_flops),
            ],
            dtype=np.float32,
        )


FEATURE_DIM = 7


def sample_workloads(n: int, rng: np.random.Generator, decode_frac: float = 0.4) -> list[Workload]:
    """Sample `n` workloads across a decode-like / prefill-like mixture."""
    workloads = []
    n_decode = int(round(n * decode_frac))
    for i in range(n):
        is_decode = i < n_decode
        if is_decode:
            seq_len = float(rng.integers(1, 33))
            batch_size = float(2 ** rng.integers(6, 10))  # 64..512
        else:
            # log-uniform over [512, 8192] so short and long prefills are
            # equally likely to be sampled, rather than long ones dominating.
            seq_len = float(
                2 ** rng.uniform(np.log2(512), np.log2(8192))
            )
            batch_size = float(2 ** rng.integers(0, 4))  # 1..8

        head_dim = float(rng.choice(HEAD_DIMS))
        num_heads = float(rng.choice(NUM_HEADS))
        sram, bw, flops, dtype_bytes, name = HARDWARE_PROFILES[
            rng.integers(0, len(HARDWARE_PROFILES))
        ]

        workloads.append(
            Workload(
                seq_len=seq_len,
                head_dim=head_dim,
                num_heads=num_heads,
                batch_size=batch_size,
                sram_size_bytes=float(sram),
                hbm_bandwidth_gb_s=float(bw),
                compute_throughput_flops=float(flops),
                dtype_bytes=float(dtype_bytes),
                hardware_name=name,
            )
        )
    rng.shuffle(workloads)
    return workloads


def fixed_eval_workloads() -> list[Workload]:
    """A small, fixed, human-readable set of shapes for tracking policy
    predictions over the course of training (independent of the random
    training distribution). Deliberately spans decode -> prefill and
    small -> large head_dim, so tile-size trends vs. seq_len are visible.

    Uses the tightest-SRAM hardware profile (L4) rather than A100. In this
    cost model, Bc only ever receives gradient signal through the SRAM
    feasibility term (it has zero effect on HBM traffic or compute FLOPs on
    its own -- see cost_model.cpp) -- while Br is pushed up for longer
    sequences (fewer outer-loop K/V re-reads). On A100's larger SRAM budget,
    max-sized tiles are still (barely) feasible even at head_dim=128, so the
    Br-vs-Bc SRAM trade-off this model predicts (grow Br for long sequences,
    which forces Bc down to stay within budget) rarely actually engages
    during training and the eval trace would mostly show incidental drift
    through the policy's shared hidden layers rather than a directly-learned
    effect. L4's smaller SRAM budget makes that trade-off bind for the
    larger shapes below, so it can actually show up in the eval predictions.
    """
    sram, bw, flops, dtype_bytes, name = HARDWARE_PROFILES[2]  # L4 (tightest SRAM)
    shapes = [
        (1, 64, 8, 256),
        (32, 64, 8, 128),
        (512, 64, 16, 8),
        (2048, 96, 16, 4),
        (4096, 128, 32, 2),
        (8192, 128, 32, 1),
    ]
    return [
        Workload(
            seq_len=float(s),
            head_dim=float(d),
            num_heads=float(h),
            batch_size=float(b),
            sram_size_bytes=float(sram),
            hbm_bandwidth_gb_s=float(bw),
            compute_throughput_flops=float(flops),
            dtype_bytes=float(dtype_bytes),
            hardware_name=name,
        )
        for (s, d, h, b) in shapes
    ]


def held_out_workloads(n: int = 30) -> list[Workload]:
    """A larger, fixed-seed evaluation set for the baselines comparison
    (Phase 4): grid search, random search, and the trained policy are all
    scored on this same set. Uses HELD_OUT_SEED, independent of whatever
    seed a given training run uses for its training-time sampling, so these
    shapes are never part of the training RNG stream.
    """
    rng = np.random.default_rng(HELD_OUT_SEED)
    return sample_workloads(n, rng, decode_frac=0.4)


def workloads_to_batch(workloads: list[Workload]) -> dict[str, np.ndarray]:
    """Stack a list of Workloads into a dict-of-arrays pytree suitable for
    jax.vmap: one array per field, each of shape (len(workloads),).
    """
    fields = (
        "seq_len",
        "head_dim",
        "num_heads",
        "batch_size",
        "sram_size_bytes",
        "hbm_bandwidth_gb_s",
        "compute_throughput_flops",
        "dtype_bytes",
    )
    batch = {f: np.array([getattr(w, f) for w in workloads], dtype=np.float64) for f in fields}
    batch["features"] = np.stack([w.to_feature_vector() for w in workloads]).astype(np.float64)
    return batch
