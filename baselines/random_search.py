# Copyright 2026. Tesseract Hackathon submission.
"""Random-search baseline: sample (Br, Bc) uniformly at random from the same
valid continuous tile-size range the policy predicts into
(policy.sample_workloads.TILE_SIZE_MIN/MAX), evaluate against the cost
model, and track the best latency found -- same evaluation-budget
accounting as grid_search.py, for the same headline comparison plot.

Unlike the grid baseline, random search can land on non-power-of-2 tile
sizes, so it's not strictly dominated by the grid; it's included because
it's the standard "how much is the grid structure buying you" control in
any autotuning comparison.
"""

from __future__ import annotations

import numpy as np
from tesseract_core import Tesseract

from baselines.common import evaluate_latency
from policy.sample_workloads import TILE_SIZE_MAX, TILE_SIZE_MIN, Workload


def random_search_trace(
    client: Tesseract,
    workload: Workload,
    num_evals: int,
    rng: np.random.Generator,
) -> list[tuple[int, float]]:
    """Same (num_evals_so_far, best_latency_so_far) trace format as
    grid_search_trace, for a fixed evaluation budget `num_evals`.
    """
    trace = []
    best = float("inf")
    for n in range(1, num_evals + 1):
        br = float(rng.uniform(TILE_SIZE_MIN, TILE_SIZE_MAX))
        bc = float(rng.uniform(TILE_SIZE_MIN, TILE_SIZE_MAX))
        latency = evaluate_latency(client, workload, br, bc)
        best = min(best, latency)
        trace.append((n, best))
    return trace


if __name__ == "__main__":
    from baselines.common import make_client
    from policy.sample_workloads import held_out_workloads

    client = make_client()
    rng = np.random.default_rng(0)
    workloads = held_out_workloads(5)
    for i, w in enumerate(workloads):
        trace = random_search_trace(client, w, num_evals=16, rng=rng)
        print(f"workload {i} (seq_len={w.seq_len:.0f}, hw={w.hardware_name}): "
              f"best latency over {len(trace)} evals = {trace[-1][1]:.2f} us")
