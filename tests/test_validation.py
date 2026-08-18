# Copyright 2026. Tesseract Hackathon submission.
"""Tests for the Phase 5 validation demo (validation/tiny_transformer.py,
validation/throughput_compare.py). Kept to a handful of steps and small
dimensions to stay fast; see those modules' docstrings for the explicit
scope of what this demo does and does not validate (a real, working
transformer trained with this project's stack, and the trained policy's
recommendation for its shape; not a real tile-size-driven speedup, since
the attention implementation here has no tile-size knob to vary).
"""

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from policy.model import TilingPolicy  # noqa: E402
from validation.throughput_compare import workload_for_tiny_transformer  # noqa: E402
from validation.tiny_transformer import GPTConfig, TinyGPT, get_batch, load_dataset, loss_fn, train  # noqa: E402


def test_dataset_loads():
    data, stoi, itos = load_dataset()
    assert len(data) > 0
    assert len(stoi) == len(itos)
    assert all(0 <= v < len(stoi) for v in stoi.values())


def test_model_forward_pass_shape():
    data, stoi, _ = load_dataset()
    config = GPTConfig(vocab_size=len(stoi), seq_len=16, d_model=32, num_heads=4, num_layers=2)
    model = TinyGPT(config, jax.random.PRNGKey(0))
    rng = np.random.default_rng(0)
    x_batch, y_batch = get_batch(data, config.seq_len, batch_size=4, rng=rng)
    logits = jax.vmap(model)(x_batch)
    assert logits.shape == (4, config.seq_len, config.vocab_size)
    loss = loss_fn(model, x_batch, y_batch)
    assert np.isfinite(float(loss))
    assert float(loss) > 0


def test_training_reduces_loss():
    config = GPTConfig(vocab_size=0, seq_len=16, d_model=32, num_heads=4, num_layers=2)
    _, loss_history, seconds_per_step, tokens_per_second = train(
        config, steps=30, batch_size=8, learning_rate=1e-3, seed=0, log_every=1000
    )
    assert len(loss_history) == 30
    # Not monotonic step to step (minibatch noise), but the mean of the
    # second half should be meaningfully lower than the first value.
    assert np.mean(loss_history[-5:]) < loss_history[0]
    assert seconds_per_step > 0
    assert tokens_per_second > 0


def test_workload_for_tiny_transformer_matches_config():
    config = GPTConfig(vocab_size=0, seq_len=64, d_model=128, num_heads=4, num_layers=4)
    workload = workload_for_tiny_transformer(config, batch_size=32, hardware_index=0)
    assert workload.seq_len == 64
    assert workload.head_dim == 32  # d_model / num_heads
    assert workload.num_heads == 4
    assert workload.batch_size == 32


def test_policy_produces_valid_recommendation_for_tiny_transformer_shape():
    from policy.sample_workloads import TILE_SIZE_MAX, TILE_SIZE_MIN

    config = GPTConfig(vocab_size=0, seq_len=64, d_model=128, num_heads=4, num_layers=4)
    workload = workload_for_tiny_transformer(config, batch_size=32, hardware_index=0)
    policy = TilingPolicy(jax.random.PRNGKey(0))
    br, bc = policy(jnp.asarray(workload.to_feature_vector()))
    assert TILE_SIZE_MIN <= float(br) <= TILE_SIZE_MAX
    assert TILE_SIZE_MIN <= float(bc) <= TILE_SIZE_MAX
