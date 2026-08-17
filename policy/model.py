# Copyright 2026. Tesseract Hackathon submission.
"""Tiling-policy MLP (Tesseract B): workload descriptor -> continuous (Br, Bc).

Small MLP with a smooth, bounded output parameterization: raw MLP logits are
passed through a sigmoid and rescaled into [TILE_SIZE_MIN, TILE_SIZE_MAX], so
predictions always stay in a physically valid tile-size range without a
non-differentiable clip. Because the sigmoid saturates smoothly at the
boundaries, gradients from the cost model's SRAM-infeasibility branch never
have to fight a clipping discontinuity on top of the branch's own
discontinuity.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from policy.sample_workloads import FEATURE_DIM, TILE_SIZE_MAX, TILE_SIZE_MIN


class TilingPolicy(eqx.Module):
    layers: list

    def __init__(self, key: jax.Array, hidden_sizes: tuple[int, ...] = (64, 64, 32)):
        sizes = [FEATURE_DIM, *hidden_sizes, 2]  # 2 outputs: Br, Bc
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i])
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        x = features
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        logits = self.layers[-1](x)  # shape (2,)
        # Smooth bounded parameterization -> valid tile-size range.
        tile_range = TILE_SIZE_MAX - TILE_SIZE_MIN
        tiles = TILE_SIZE_MIN + tile_range * jax.nn.sigmoid(logits)
        return tiles  # [Br, Bc]


def predict_tiles(policy: TilingPolicy, features: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    br, bc = policy(features)
    return br, bc
