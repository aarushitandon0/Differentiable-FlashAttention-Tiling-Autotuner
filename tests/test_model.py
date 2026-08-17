# Copyright 2026. Tesseract Hackathon submission.
"""Tests for policy/model.py: the tiling policy MLP's output stays within
the valid tile-size range for arbitrary inputs, since nothing downstream
(the cost model, the baselines) clips it.
"""

import jax
import jax.numpy as jnp
import numpy as np

from policy.model import TilingPolicy
from policy.sample_workloads import FEATURE_DIM, TILE_SIZE_MAX, TILE_SIZE_MIN, sample_workloads


def test_output_shape():
    policy = TilingPolicy(jax.random.PRNGKey(0))
    out = policy(jnp.zeros(FEATURE_DIM))
    assert out.shape == (2,)


def test_output_always_within_valid_tile_range():
    policy = TilingPolicy(jax.random.PRNGKey(0))
    rng = np.random.default_rng(0)
    for w in sample_workloads(50, rng):
        br, bc = policy(jnp.asarray(w.to_feature_vector()))
        assert TILE_SIZE_MIN <= float(br) <= TILE_SIZE_MAX
        assert TILE_SIZE_MIN <= float(bc) <= TILE_SIZE_MAX


def test_output_within_range_for_extreme_inputs():
    # Large-magnitude features (e.g. far outside the training distribution)
    # should still saturate into the valid range via the sigmoid bound,
    # never escape it.
    policy = TilingPolicy(jax.random.PRNGKey(0))
    for value in (-1000.0, 1000.0):
        features = jnp.full((FEATURE_DIM,), value)
        br, bc = policy(features)
        assert TILE_SIZE_MIN <= float(br) <= TILE_SIZE_MAX
        assert TILE_SIZE_MIN <= float(bc) <= TILE_SIZE_MAX


def test_different_seeds_give_different_policies():
    p1 = TilingPolicy(jax.random.PRNGKey(0))
    p2 = TilingPolicy(jax.random.PRNGKey(1))
    features = jnp.ones(FEATURE_DIM)
    out1 = p1(features)
    out2 = p2(features)
    assert not jnp.allclose(out1, out2)


def test_is_differentiable_wrt_parameters():
    import equinox as eqx

    policy = TilingPolicy(jax.random.PRNGKey(0))
    features = jnp.ones(FEATURE_DIM)

    def f(p):
        br, bc = p(features)
        return br + bc

    grads = eqx.filter_grad(f)(policy)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert len(leaves) > 0
    assert any(float(jnp.linalg.norm(leaf)) > 0 for leaf in leaves)
