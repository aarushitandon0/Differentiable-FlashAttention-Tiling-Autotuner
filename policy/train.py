# Copyright 2026. Tesseract Hackathon submission.
"""Train the tiling policy (Tesseract B) end-to-end through the attention
cost model (Tesseract A), backpropagating through Tesseract A's
finite-difference Jacobian via tesseract-jax's `apply_tesseract`.

Composition, concretely:

    workload features --[policy: plain JAX/Equinox, standard autodiff]--> Br, Bc
    (Br, Bc, workload, hardware) --[apply_tesseract -> Tesseract A]--> predicted_latency_us
    loss = predicted_latency_us + soft SRAM-infeasibility penalty
    jax.grad(loss) flows through apply_tesseract's custom_vjp (backed by
    Tesseract A's finite-difference vector_jacobian_product endpoint) and
    then through the policy's own parameters via ordinary JAX autodiff.

Run:
    python -m policy.train --steps 2000 --batch-size 32
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)  # cost model schema uses Float64

import jax.numpy as jnp  # noqa: E402
from tesseract_core import Tesseract  # noqa: E402
from tesseract_jax import apply_tesseract  # noqa: E402

from policy.model import TilingPolicy  # noqa: E402
from policy.sample_workloads import (  # noqa: E402
    fixed_eval_workloads,
    sample_workloads,
    workloads_to_batch,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
COST_MODEL_API_PATH = REPO_ROOT / "tesseracts" / "attention-cost-model" / "tesseract_api.py"


def make_cost_model_client() -> Tesseract:
    """Local (no-Docker) Tesseract client for fast iteration during
    training. For the containerized/deployed version, swap this for
    `Tesseract.from_image("attention-cost-model")` against a built image --
    the rest of the training loop is unchanged since apply_tesseract's
    interface is identical either way.
    """
    return Tesseract.from_tesseract_api(str(COST_MODEL_API_PATH))


def batch_loss_and_metrics(
    policy: TilingPolicy,
    batch: dict,
    cost_model: Tesseract,
    soft_penalty_weight: float,
):
    """Mean loss + per-sample metrics over a batch of workloads."""

    def per_sample(features, seq_len, head_dim, num_heads, batch_size,
                    sram_size_bytes, hbm_bandwidth_gb_s, compute_throughput_flops, dtype_bytes):
        br, bc = policy(features)
        cost_inputs = {
            "Br": br,
            "Bc": bc,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "num_heads": num_heads,
            "batch_size": batch_size,
            "sram_size_bytes": sram_size_bytes,
            "hbm_bandwidth_gb_s": hbm_bandwidth_gb_s,
            "compute_throughput_flops": compute_throughput_flops,
            "dtype_bytes": dtype_bytes,
        }
        out = apply_tesseract(cost_model, cost_inputs, vmap_method="sequential")
        latency = out["predicted_latency_us"]
        sram_util = out["sram_utilization"]
        # Soft penalty on top of the cost model's own hard SRAM-infeasibility
        # jump: encourages the policy's gradient signal to point back toward
        # the feasible region even from deep inside the infeasible region,
        # where (as verified during Phase 1) the hard penalty alone is
        # *locally flat* and gives zero gradient.
        soft_penalty = jax.nn.relu(sram_util - 1.0) ** 2
        loss = latency + soft_penalty_weight * soft_penalty
        return loss, (latency, sram_util, br, bc)

    losses, aux = jax.vmap(per_sample)(
        batch["features"],
        batch["seq_len"],
        batch["head_dim"],
        batch["num_heads"],
        batch["batch_size"],
        batch["sram_size_bytes"],
        batch["hbm_bandwidth_gb_s"],
        batch["compute_throughput_flops"],
        batch["dtype_bytes"],
    )
    latency, sram_util, br, bc = aux
    metrics = {
        "mean_latency_us": jnp.mean(latency),
        "mean_sram_utilization": jnp.mean(sram_util),
        "frac_infeasible": jnp.mean((sram_util > 1.0).astype(jnp.float64)),
        "mean_br": jnp.mean(br),
        "mean_bc": jnp.mean(bc),
    }
    return jnp.mean(losses), metrics


def make_train_step(cost_model: Tesseract, soft_penalty_weight: float, learning_rate: float):
    optimizer = optax.adam(learning_rate)

    @eqx.filter_jit
    def train_step(policy, opt_state, batch):
        (loss, metrics), grads = eqx.filter_value_and_grad(
            lambda p: batch_loss_and_metrics(p, batch, cost_model, soft_penalty_weight),
            has_aux=True,
        )(policy)
        updates, opt_state = optimizer.update(grads, opt_state, policy)
        policy = eqx.apply_updates(policy, updates)
        return policy, opt_state, loss, metrics

    return train_step, optimizer


def evaluate_fixed_predictions(policy: TilingPolicy, eval_workloads) -> list[dict]:
    rows = []
    for w in eval_workloads:
        br, bc = policy(jnp.asarray(w.to_feature_vector()))
        rows.append(
            {
                "seq_len": w.seq_len,
                "head_dim": w.head_dim,
                "num_heads": w.num_heads,
                "batch_size": w.batch_size,
                "hardware": w.hardware_name,
                "Br": float(br),
                "Bc": float(bc),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--soft-penalty-weight", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--decode-frac", type=float, default=0.4)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    np_rng = np.random.default_rng(args.seed)
    jax_key = jax.random.PRNGKey(args.seed)

    policy = TilingPolicy(jax_key)
    cost_model = make_cost_model_client()
    train_step, optimizer = make_train_step(cost_model, args.soft_penalty_weight, args.lr)
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    eval_workloads = fixed_eval_workloads()

    loss_curve_path = run_dir / "loss_curve.csv"
    eval_predictions_path = run_dir / "eval_predictions.jsonl"

    with open(loss_curve_path, "w", newline="") as loss_f:
        loss_writer = csv.writer(loss_f)
        loss_writer.writerow(
            ["step", "loss", "mean_latency_us", "mean_sram_utilization", "frac_infeasible", "mean_br", "mean_bc"]
        )

        with open(eval_predictions_path, "w") as eval_f:
            for step in range(1, args.steps + 1):
                workloads = sample_workloads(args.batch_size, np_rng, decode_frac=args.decode_frac)
                batch = jax.tree_util.tree_map(jnp.asarray, workloads_to_batch(workloads))

                policy, opt_state, loss, metrics = train_step(policy, opt_state, batch)

                if step % args.log_every == 0 or step == 1:
                    row = [
                        step,
                        float(loss),
                        float(metrics["mean_latency_us"]),
                        float(metrics["mean_sram_utilization"]),
                        float(metrics["frac_infeasible"]),
                        float(metrics["mean_br"]),
                        float(metrics["mean_bc"]),
                    ]
                    loss_writer.writerow(row)
                    loss_f.flush()
                    print(
                        f"step {step:5d} | loss {row[1]:10.3f} | "
                        f"latency_us {row[2]:9.3f} | sram_util {row[3]:.3f} | "
                        f"infeasible {row[4]:.2%} | Br {row[5]:6.2f} | Bc {row[6]:6.2f}"
                    )

                if step % args.eval_every == 0 or step == args.steps:
                    predictions = evaluate_fixed_predictions(policy, eval_workloads)
                    eval_f.write(json.dumps({"step": step, "predictions": predictions}) + "\n")
                    eval_f.flush()

    checkpoint_path = run_dir / "policy_final.eqx"
    eqx.tree_serialise_leaves(checkpoint_path, policy)
    print(f"\nDone. Loss curve: {loss_curve_path}")
    print(f"Eval predictions over training: {eval_predictions_path}")
    print(f"Final policy checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
