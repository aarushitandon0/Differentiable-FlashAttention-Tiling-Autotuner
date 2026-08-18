# Copyright 2026. Tesseract Hackathon submission.
"""A small, real, character-level GPT trained on TinyShakespeare, in JAX/Equinox,
for the Phase 5 validation demo.

Scope, stated plainly: this trains a real transformer with real data and reports
real measured loss and wall-clock throughput. It uses a standard dense
(einsum-based) causal self-attention implementation, which has no tile-size
knob to vary, since tiling is a GPU kernel implementation detail that a
high-level JAX einsum does not expose. Running this on CPU (no GPU was
available in the environment this was developed in, see the README) means
there is no real tiling effect to measure here at all. This script exists to
demonstrate that the trained tiling policy's recommendation is being computed
for a real model's real shape, and that a real transformer can in fact be
trained with this project's stack end to end, not to demonstrate a real
tile-size-driven speedup. See validation/throughput_compare.py for how the
policy's recommendation is reported alongside this script's real training
numbers, and for the explicit statement of what is and is not validated by
doing so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

DATA_PATH = Path(__file__).parent / "data" / "tinyshakespeare.txt"


@dataclass
class GPTConfig:
    vocab_size: int
    seq_len: int = 64
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    mlp_ratio: int = 4

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.num_heads == 0
        return self.d_model // self.num_heads


class CausalSelfAttention(eqx.Module):
    qkv_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, config: GPTConfig, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.qkv_proj = eqx.nn.Linear(config.d_model, 3 * config.d_model, key=k1)
        self.out_proj = eqx.nn.Linear(config.d_model, config.d_model, key=k2)
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (seq_len, d_model)
        seq_len, d_model = x.shape
        qkv = jax.vmap(self.qkv_proj)(x)  # (seq_len, 3 * d_model)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def split_heads(t):
            return t.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)  # (num_heads, seq_len, head_dim)

        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum("hqd,hkd->hqk", q, k) * scale
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        scores = jnp.where(causal_mask, scores, -jnp.inf)
        probs = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.einsum("hqk,hkd->hqd", probs, v)  # (num_heads, seq_len, head_dim)
        attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, d_model)
        return jax.vmap(self.out_proj)(attn_out)


class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, config: GPTConfig, key: jax.Array):
        k1, k2 = jax.random.split(key)
        hidden = config.d_model * config.mlp_ratio
        self.fc1 = eqx.nn.Linear(config.d_model, hidden, key=k1)
        self.fc2 = eqx.nn.Linear(hidden, config.d_model, key=k2)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda t: self.fc2(jax.nn.gelu(self.fc1(t))))(x)


class Block(eqx.Module):
    ln1: eqx.nn.LayerNorm
    attn: CausalSelfAttention
    ln2: eqx.nn.LayerNorm
    mlp: MLP

    def __init__(self, config: GPTConfig, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.ln1 = eqx.nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config, k1)
        self.ln2 = eqx.nn.LayerNorm(config.d_model)
        self.mlp = MLP(config, k2)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attn(jax.vmap(self.ln1)(x))
        x = x + self.mlp(jax.vmap(self.ln2)(x))
        return x


class TinyGPT(eqx.Module):
    token_embed: eqx.nn.Embedding
    pos_embed: jnp.ndarray
    blocks: list
    ln_f: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    config: GPTConfig = eqx.field(static=True)

    def __init__(self, config: GPTConfig, key: jax.Array):
        keys = jax.random.split(key, config.num_layers + 3)
        self.token_embed = eqx.nn.Embedding(config.vocab_size, config.d_model, key=keys[0])
        self.pos_embed = jax.random.normal(keys[1], (config.seq_len, config.d_model)) * 0.02
        self.blocks = [Block(config, keys[2 + i]) for i in range(config.num_layers)]
        self.ln_f = eqx.nn.LayerNorm(config.d_model)
        self.head = eqx.nn.Linear(config.d_model, config.vocab_size, key=keys[-1])
        self.config = config

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        # token_ids: (seq_len,) int32 -> logits: (seq_len, vocab_size)
        x = jax.vmap(self.token_embed)(token_ids) + self.pos_embed[: token_ids.shape[0]]
        for block in self.blocks:
            x = block(x)
        x = jax.vmap(self.ln_f)(x)
        return jax.vmap(self.head)(x)


def load_dataset(path: Path = DATA_PATH) -> tuple[np.ndarray, dict, dict]:
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in text], dtype=np.int32)
    return data, stoi, itos


def get_batch(data: np.ndarray, seq_len: int, batch_size: int, rng: np.random.Generator):
    ix = rng.integers(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[i : i + seq_len] for i in ix])
    y = np.stack([data[i + 1 : i + seq_len + 1] for i in ix])
    return jnp.asarray(x), jnp.asarray(y)


def loss_fn(model: TinyGPT, x_batch: jnp.ndarray, y_batch: jnp.ndarray) -> jnp.ndarray:
    logits = jax.vmap(model)(x_batch)  # (batch, seq_len, vocab_size)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, y_batch[..., None], axis=-1).squeeze(-1)
    return jnp.mean(nll)


def make_train_step(learning_rate: float):
    optimizer = optax.adamw(learning_rate)

    @eqx.filter_jit
    def train_step(model, opt_state, x_batch, y_batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    return train_step, optimizer


def train(
    config: GPTConfig,
    steps: int,
    batch_size: int,
    learning_rate: float = 3e-4,
    seed: int = 0,
    log_every: int = 20,
    data_path: Path = DATA_PATH,
):
    """Train TinyGPT for `steps` steps and return (model, loss_history,
    measured_seconds_per_step, tokens_per_second).
    """
    data, stoi, itos = load_dataset(data_path)
    config = GPTConfig(vocab_size=len(stoi), **{k: v for k, v in vars(config).items() if k != "vocab_size"})

    key = jax.random.PRNGKey(seed)
    model = TinyGPT(config, key)
    train_step, optimizer = make_train_step(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    rng = np.random.default_rng(seed)
    loss_history = []

    # Warm up (first call includes JIT compilation, excluded from the
    # measured throughput below).
    x_batch, y_batch = get_batch(data, config.seq_len, batch_size, rng)
    model, opt_state, loss = train_step(model, opt_state, x_batch, y_batch)
    loss_history.append(float(loss))

    start = time.perf_counter()
    for step in range(1, steps):
        x_batch, y_batch = get_batch(data, config.seq_len, batch_size, rng)
        model, opt_state, loss = train_step(model, opt_state, x_batch, y_batch)
        loss_history.append(float(loss))
        if step % log_every == 0:
            print(f"step {step:5d} | loss {float(loss):.4f}")
    elapsed = time.perf_counter() - start

    steps_measured = steps - 1
    seconds_per_step = elapsed / max(steps_measured, 1)
    tokens_per_step = batch_size * config.seq_len
    tokens_per_second = tokens_per_step / seconds_per_step if seconds_per_step > 0 else float("nan")

    return model, loss_history, seconds_per_step, tokens_per_second


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = GPTConfig(
        vocab_size=0,  # overwritten in train() from the actual dataset
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    _, loss_history, seconds_per_step, tokens_per_second = train(
        config, steps=args.steps, batch_size=args.batch_size, learning_rate=args.lr, seed=args.seed
    )
    print(f"\nFinal loss: {loss_history[-1]:.4f} (started at {loss_history[0]:.4f})")
    print(f"Measured: {seconds_per_step * 1000:.2f} ms/step, {tokens_per_second:.0f} tokens/sec (post-warmup)")
