# Copyright 2026. Tesseract Hackathon submission.
"""End-to-end integration tests for policy/train.py's training loop and
baselines/compare.py's checkpoint-progress evaluation, run for real against
the compiled cost model (not mocked). Kept to a handful of steps and a
small batch size so the suite stays fast; correctness of the cost model
itself and of the policy's output bounds is covered separately in
test_cost_model_wrapper.py and test_model.py. This file exists to catch
regressions in the wiring between them (batch construction, vmap axes,
checkpoint bookkeeping) that those narrower tests cannot see.
"""

import csv
import sys

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from baselines.compare import evaluate_checkpoint_progress  # noqa: E402
from policy import train as train_module  # noqa: E402
from policy.model import TilingPolicy  # noqa: E402
from policy.sample_workloads import held_out_workloads, sample_workloads, workloads_to_batch  # noqa: E402


@pytest.fixture(scope="module")
def cost_model_client():
    return train_module.make_cost_model_client()


def test_train_step_runs_and_updates_policy(cost_model_client):
    policy = TilingPolicy(jax.random.PRNGKey(0))
    train_step, optimizer = train_module.make_train_step(
        cost_model_client, soft_penalty_weight=50.0, learning_rate=1e-1
    )
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    rng = np.random.default_rng(0)
    workloads = sample_workloads(4, rng)
    batch = jax.tree_util.tree_map(jnp.asarray, workloads_to_batch(workloads))

    before = jax.tree_util.tree_leaves(eqx.filter(policy, eqx.is_array))
    new_policy, new_opt_state, loss, metrics = train_step(policy, opt_state, batch)
    after = jax.tree_util.tree_leaves(eqx.filter(new_policy, eqx.is_array))

    assert np.isfinite(float(loss))
    assert any(
        not jnp.array_equal(b, a) for b, a in zip(before, after)
    ), "train_step did not change any policy parameter"


def test_main_training_loop_produces_expected_outputs(tmp_path, monkeypatch):
    argv = [
        "policy.train",
        "--steps", "3",
        "--batch-size", "4",
        "--log-every", "1",
        "--eval-every", "3",
        "--checkpoint-every", "1",
        "--out-dir", str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_module.main()

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert (run_dir / "loss_curve.csv").exists()
    assert (run_dir / "eval_predictions.jsonl").exists()
    assert (run_dir / "policy_final.eqx").exists()

    manifest_path = run_dir / "checkpoints" / "manifest.csv"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3  # one checkpoint per step, --checkpoint-every 1
    for row in rows:
        assert (run_dir / "checkpoints" / f"step_{int(row['step']):07d}.eqx").exists()
        assert int(row["cumulative_evals"]) > 0

    with open(run_dir / "loss_curve.csv") as f:
        loss_rows = list(csv.DictReader(f))
    assert len(loss_rows) == 3
    assert all(np.isfinite(float(r["loss"])) for r in loss_rows)


def test_evaluate_checkpoint_progress_matches_manifest_length(tmp_path, monkeypatch, cost_model_client):
    argv = [
        "policy.train",
        "--steps", "2",
        "--batch-size", "4",
        "--log-every", "1",
        "--eval-every", "2",
        "--checkpoint-every", "1",
        "--out-dir", str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_module.main()

    run_dir = next(tmp_path.iterdir())
    manifest_path = run_dir / "checkpoints" / "manifest.csv"

    workloads = held_out_workloads(3)
    progress = evaluate_checkpoint_progress(manifest_path, cost_model_client, workloads)

    assert len(progress["cumulative_evals"]) == 2
    assert len(progress["mean_latency_us"]) == 2
    assert all(np.isfinite(v) for v in progress["mean_latency_us"])
    assert progress["cumulative_evals"] == sorted(progress["cumulative_evals"])
