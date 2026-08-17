# Copyright 2026. Tesseract Hackathon submission.
"""Tests for baselines/grid_search.py and baselines/random_search.py."""

import numpy as np
import pytest

from baselines.common import make_client
from baselines.grid_search import TILE_GRID, best_latency_at_budget, grid_search_trace
from baselines.random_search import random_search_trace
from policy.sample_workloads import held_out_workloads


@pytest.fixture(scope="module")
def client():
    return make_client()


@pytest.fixture(scope="module")
def workload():
    return held_out_workloads(1)[0]


def test_grid_search_trace_length(client, workload):
    trace = grid_search_trace(client, workload)
    assert len(trace) == len(TILE_GRID) ** 2


def test_grid_search_best_latency_is_monotonically_nonincreasing(client, workload):
    trace = grid_search_trace(client, workload)
    bests = [b for _, b in trace]
    assert all(bests[i] >= bests[i + 1] - 1e-9 for i in range(len(bests) - 1))


def test_best_latency_at_budget_matches_trace(client, workload):
    trace = grid_search_trace(client, workload)
    for n, best in trace:
        assert best_latency_at_budget(trace, n) == pytest.approx(best)


def test_random_search_trace_length_and_monotonicity(client, workload):
    rng = np.random.default_rng(0)
    trace = random_search_trace(client, workload, num_evals=16, rng=rng)
    assert len(trace) == 16
    bests = [b for _, b in trace]
    assert all(bests[i] >= bests[i + 1] - 1e-9 for i in range(len(bests) - 1))


def test_random_search_is_seeded_reproducibly(client, workload):
    trace_a = random_search_trace(client, workload, num_evals=8, rng=np.random.default_rng(42))
    trace_b = random_search_trace(client, workload, num_evals=8, rng=np.random.default_rng(42))
    assert trace_a == trace_b
