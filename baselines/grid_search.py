# Copyright 2026. Tesseract Hackathon submission.
"""Grid-search baseline: sweep a standard Triton-autotuning-style tile-size
grid (powers of 2 from 16 to 128, matching the policy's own valid output
range in policy/sample_workloads.py) against the cost model, per workload.

This is the realistic status-quo baseline this whole project is arguing
against: for every new workload shape, re-run the full grid against the
cost/latency oracle to find a good tile size. Tracks best latency found as
a function of number of cost-model evaluations spent, so it can be plotted
against the gradient-trained policy at matched evaluation budgets (see
compare.py).
"""

from __future__ import annotations

from tesseract_core import Tesseract

from baselines.common import evaluate_latency
from policy.sample_workloads import Workload

TILE_GRID = (16.0, 32.0, 64.0, 128.0)


def grid_search_trace(client: Tesseract, workload: Workload, tile_grid=TILE_GRID) -> list[tuple[int, float]]:
    """Evaluate every (Br, Bc) in tile_grid x tile_grid, in order.

    Returns a list of (num_evals_so_far, best_latency_so_far) after each
    evaluation, so callers can read off "best latency achievable with a
    budget of k evaluations" for any k <= len(tile_grid)**2.
    """
    trace = []
    best = float("inf")
    n = 0
    for br in tile_grid:
        for bc in tile_grid:
            latency = evaluate_latency(client, workload, br, bc)
            n += 1
            best = min(best, latency)
            trace.append((n, best))
    return trace


def best_latency_at_budget(trace: list[tuple[int, float]], budget: int) -> float:
    """Best latency achieved using only the first `budget` evaluations."""
    best = float("inf")
    for n, best_so_far in trace:
        if n > budget:
            break
        best = best_so_far
    return best


if __name__ == "__main__":
    from baselines.common import make_client
    from policy.sample_workloads import held_out_workloads

    client = make_client()
    workloads = held_out_workloads(5)
    for i, w in enumerate(workloads):
        trace = grid_search_trace(client, w)
        print(f"workload {i} (seq_len={w.seq_len:.0f}, hw={w.hardware_name}): "
              f"best latency over {len(trace)} evals = {trace[-1][1]:.2f} us")
