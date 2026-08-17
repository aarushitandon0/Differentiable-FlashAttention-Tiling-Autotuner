# Copyright 2026. Tesseract Hackathon submission.
"""Headline comparison: grid search vs. random search vs. the gradient-trained
tiling policy, evaluated on a held-out set of workload shapes never seen
during training (policy.sample_workloads.held_out_workloads).

The fair unit for "evaluation budget" differs by method, and that
difference *is* the headline result:

  - Grid search / random search must spend fresh cost-model evaluations
    for *every new workload* -- they have no memory across workloads. Their
    curves below show mean best-latency-found vs. evaluations spent per
    workload (1..len(TILE_GRID)**2).
  - The trained policy pays its cost once, during training (amortized over
    every workload it was trained on), and then needs *zero* additional
    cost-model evaluations for a brand-new held-out workload: it's a single
    forward pass through a small MLP. Its result is plotted as a horizontal
    reference line at x=0 (labelled with the one apply() call spent purely
    to *report* its achieved latency, not to search for it).

Run (after training a policy checkpoint with `python -m policy.train`):
    python -m baselines.compare --checkpoint runs/<run_id>/policy_final.eqx
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from baselines.common import evaluate_latency, make_client  # noqa: E402
from baselines.grid_search import TILE_GRID, best_latency_at_budget, grid_search_trace  # noqa: E402
from baselines.random_search import random_search_trace  # noqa: E402
from policy.model import TilingPolicy  # noqa: E402
from policy.sample_workloads import held_out_workloads  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Rough per-sample cost-model evaluation count for the gradient method's
# amortized training cost, reported for transparency (not used as the
# per-workload deployment budget, which is zero -- see module docstring):
# 1 forward `apply` (already needed for the loss value) + up to 2 evals per
# differentiable input for a central-difference vector_jacobian_product
# (Br, Bc only -- the other Differentiable-marked schema fields are never
# included in vjp_inputs during training, since JAX only requests
# cotangents for inputs actually on the policy's differentiation path).
EVALS_PER_TRAINING_SAMPLE = 1 + 2 * 2


def load_policy(checkpoint_path: Path) -> TilingPolicy:
    skeleton = TilingPolicy(jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(checkpoint_path, like=skeleton)


def training_evals_from_loss_curve(loss_curve_path: Path, batch_size: int) -> int:
    with open(loss_curve_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0
    last_step = int(rows[-1]["step"])
    return last_step * batch_size * EVALS_PER_TRAINING_SAMPLE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-held-out", type=int, default=30)
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "compare")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    client = make_client()
    workloads = held_out_workloads(args.num_held_out)
    max_budget = len(TILE_GRID) ** 2

    # --- Grid search: mean best-latency-vs-budget across held-out set -----
    grid_traces = [grid_search_trace(client, w) for w in workloads]
    grid_curve = [
        float(np.mean([best_latency_at_budget(t, b) for t in grid_traces]))
        for b in range(1, max_budget + 1)
    ]

    # --- Random search: mean (+ std across seeds) best-latency-vs-budget --
    random_curves_per_seed = []
    for seed in range(args.random_seeds):
        rng = np.random.default_rng(seed)
        traces = [random_search_trace(client, w, max_budget, rng) for w in workloads]
        random_curves_per_seed.append(
            [float(np.mean([t[b - 1][1] for t in traces])) for b in range(1, max_budget + 1)]
        )
    random_curve_mean = list(np.mean(random_curves_per_seed, axis=0))
    random_curve_std = list(np.std(random_curves_per_seed, axis=0))

    # --- Trained policy: one-shot prediction, zero search evaluations ----
    policy = load_policy(args.checkpoint)
    policy_latencies = []
    for w in workloads:
        br, bc = policy(jnp.asarray(w.to_feature_vector()))
        policy_latencies.append(evaluate_latency(client, w, float(br), float(bc)))
    policy_mean_latency = float(np.mean(policy_latencies))

    # --- Report training evaluation budget for transparency --------------
    loss_curve_path = args.checkpoint.parent / "loss_curve.csv"
    training_evals = None
    if loss_curve_path.exists():
        # batch_size isn't stored in loss_curve.csv; caller must know it
        # matched their training run. Default matches policy/train.py's
        # own --batch-size default.
        training_evals = training_evals_from_loss_curve(loss_curve_path, batch_size=32)

    # --- Write results ------------------------------------------------------
    results = {
        "num_held_out_workloads": len(workloads),
        "grid_search": {"budgets": list(range(1, max_budget + 1)), "mean_best_latency_us": grid_curve},
        "random_search": {
            "budgets": list(range(1, max_budget + 1)),
            "mean_best_latency_us": random_curve_mean,
            "std_best_latency_us": random_curve_std,
        },
        "trained_policy": {
            "per_workload_search_evals": 0,
            "mean_latency_us": policy_mean_latency,
            "amortized_training_evals": training_evals,
        },
    }
    results_path = args.out_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Grid search:   best latency at budget={max_budget}: {grid_curve[-1]:.2f} us (mean over {len(workloads)} held-out workloads)")
    print(f"Random search: best latency at budget={max_budget}: {random_curve_mean[-1]:.2f} +/- {random_curve_std[-1]:.2f} us")
    print(f"Trained policy: {policy_mean_latency:.2f} us, using 0 search evaluations per new workload"
          + (f" (amortized {training_evals:,} evals across training)" if training_evals else ""))
    print(f"Results written to {results_path}")

    try:
        _plot(grid_curve, random_curve_mean, random_curve_std, policy_mean_latency, max_budget, args.out_dir)
    except ImportError:
        print("matplotlib not installed; skipping plot (results JSON is still written).")


def _plot(grid_curve, random_curve_mean, random_curve_std, policy_mean_latency, max_budget, out_dir):
    import matplotlib.pyplot as plt

    budgets = list(range(1, max_budget + 1))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(budgets, grid_curve, marker="o", label="Grid search", color="#4C78A8")
    ax.plot(budgets, random_curve_mean, marker="s", label="Random search", color="#F58518")
    lo = [m - s for m, s in zip(random_curve_mean, random_curve_std)]
    hi = [m + s for m, s in zip(random_curve_mean, random_curve_std)]
    ax.fill_between(budgets, lo, hi, color="#F58518", alpha=0.2)
    ax.axhline(
        policy_mean_latency,
        color="#54A24B",
        linestyle="--",
        label="Trained policy (0 search evals/workload)",
    )
    ax.set_xlabel("Cost-model evaluations spent per (new, held-out) workload")
    ax.set_ylabel("Mean best predicted latency (us)")
    ax.set_title("Tile-size search: evaluations vs. latency on held-out workloads")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "evals_vs_latency.png"
    fig.savefig(out_path, dpi=150)
    print(f"Plot written to {out_path}")


if __name__ == "__main__":
    main()
