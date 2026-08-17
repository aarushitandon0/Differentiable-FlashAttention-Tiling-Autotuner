# Copyright 2026. Tesseract Hackathon submission.
"""Tests for tesseracts/attention-cost-model/tesseract_api.py: the Python
Tesseract wrapper around the compiled C++ cost model, run through the real
tesseract_core.Tesseract local (no-Docker) client. Requires the shared
library to already be built (see tesseracts/attention-cost-model/lib/, or
the README's build instructions) since these tests exercise the actual
apply/jacobian/vector_jacobian_product endpoints end to end, not the C++
directly (that is covered by
tesseracts/attention-cost-model/tests/test_cost_model.cpp).
"""

import jax

jax.config.update("jax_enable_x64", True)

import pytest  # noqa: E402
from tesseract_core import Tesseract  # noqa: E402

from policy.train import COST_MODEL_API_PATH  # noqa: E402

BASE_INPUTS = dict(
    seq_len=2048.0,
    head_dim=64.0,
    num_heads=16.0,
    batch_size=1.0,
    sram_size_bytes=164 * 1024.0,
    hbm_bandwidth_gb_s=2000.0,
    compute_throughput_flops=312e12,
    dtype_bytes=2.0,
)


@pytest.fixture(scope="module")
def client():
    return Tesseract.from_tesseract_api(str(COST_MODEL_API_PATH))


def test_feasible_tile_has_no_penalty(client):
    out = client.apply({**BASE_INPUTS, "Br": 32.0, "Bc": 32.0})
    assert out["sram_utilization"] <= 1.0
    assert out["predicted_latency_us"] > 0


def test_infeasible_tile_has_hard_penalty(client):
    feasible = client.apply({**BASE_INPUTS, "Br": 32.0, "Bc": 32.0})
    infeasible = client.apply(
        {**BASE_INPUTS, "Br": 512.0, "Bc": 512.0, "sram_size_bytes": 164 * 1024.0}
    )
    assert infeasible["sram_utilization"] > 1.0
    # Same shape/tiling but with a large enough SRAM budget to be feasible,
    # to recover the unpenalized latency and check the penalty ratio.
    unpenalized = client.apply(
        {**BASE_INPUTS, "Br": 512.0, "Bc": 512.0, "sram_size_bytes": 16.0 * 1024 * 1024}
    )
    ratio = infeasible["predicted_latency_us"] / unpenalized["predicted_latency_us"]
    assert ratio == pytest.approx(50.0, rel=1e-6)


def test_gradient_nonzero_for_realistic_noninteger_workload(client):
    # Regression test for the ceil(seq_len / Br) staircase bug: a workload
    # where seq_len is not an exact multiple of Br must still show a
    # nonzero gradient in the memory-bound regime.
    inputs = {
        **BASE_INPUTS,
        "seq_len": 2993.9756014731174,
        "Br": 92.23177260555327,
        "Bc": 48.87400299113736,
    }
    jac = client.jacobian(inputs, jac_inputs={"Br", "Bc"}, jac_outputs={"predicted_latency_us"})
    assert jac["predicted_latency_us"]["Br"] != 0.0


def test_gradient_zero_for_bc_in_feasible_region(client):
    # Bc has no effect on latency outside the SRAM branch, by design (see
    # README's "What the policy learned" section) -- this should hold in
    # the feasible interior.
    inputs = {**BASE_INPUTS, "Br": 64.0, "Bc": 64.0}
    jac = client.jacobian(inputs, jac_inputs={"Br", "Bc"}, jac_outputs={"predicted_latency_us"})
    assert jac["predicted_latency_us"]["Bc"] == 0.0


def test_abstract_eval_returns_scalar_shapes(client):
    avals = client.abstract_eval({k: {"shape": (), "dtype": "float64"} for k in
                                    ["Br", "Bc", *BASE_INPUTS.keys()]})
    for name in ("predicted_latency_us", "hbm_bytes_moved", "sram_utilization"):
        assert avals[name]["shape"] == ()
