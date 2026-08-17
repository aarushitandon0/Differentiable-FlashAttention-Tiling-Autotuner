# Copyright 2026. Tesseract Hackathon submission.
"""Tests for policy/sample_workloads.py: the workload distribution sampler
and the fixed / held-out evaluation sets used by training and the
baselines comparison.
"""

import numpy as np

from policy.sample_workloads import (
    FEATURE_DIM,
    HARDWARE_PROFILES,
    HEAD_DIMS,
    NUM_HEADS,
    TILE_SIZE_MAX,
    TILE_SIZE_MIN,
    fixed_eval_workloads,
    held_out_workloads,
    sample_workloads,
    workloads_to_batch,
)


def test_sample_workloads_respects_ranges():
    rng = np.random.default_rng(0)
    workloads = sample_workloads(200, rng, decode_frac=0.4)
    assert len(workloads) == 200
    for w in workloads:
        assert w.head_dim in HEAD_DIMS
        assert w.num_heads in NUM_HEADS
        assert w.seq_len >= 1
        assert w.batch_size >= 1
        assert any(w.sram_size_bytes == p[0] for p in HARDWARE_PROFILES)


def test_sample_workloads_covers_decode_and_prefill_regimes():
    rng = np.random.default_rng(0)
    workloads = sample_workloads(500, rng, decode_frac=0.5)
    seq_lens = [w.seq_len for w in workloads]
    # Decode-like workloads (seq_len in [1, 32]) and prefill-like workloads
    # (seq_len in [512, 8192]) should both be present; nothing should fall
    # strictly between the two regimes.
    assert any(s <= 32 for s in seq_lens)
    assert any(s >= 512 for s in seq_lens)
    assert not any(32 < s < 512 for s in seq_lens)


def test_feature_vector_shape_and_finiteness():
    rng = np.random.default_rng(1)
    for w in sample_workloads(20, rng):
        feat = w.to_feature_vector()
        assert feat.shape == (FEATURE_DIM,)
        assert np.all(np.isfinite(feat))


def test_fixed_eval_workloads_is_deterministic_and_nonempty():
    a = fixed_eval_workloads()
    b = fixed_eval_workloads()
    assert len(a) > 0
    assert [w.seq_len for w in a] == [w.seq_len for w in b]


def test_fixed_eval_workloads_spans_increasing_seq_len():
    workloads = fixed_eval_workloads()
    seq_lens = [w.seq_len for w in workloads]
    assert seq_lens == sorted(seq_lens)
    assert seq_lens[0] < seq_lens[-1]


def test_held_out_workloads_is_seeded_and_reproducible():
    a = held_out_workloads(10)
    b = held_out_workloads(10)
    assert [w.seq_len for w in a] == [w.seq_len for w in b]
    assert [w.head_dim for w in a] == [w.head_dim for w in b]


def test_held_out_workloads_independent_of_training_rng():
    # held_out_workloads uses its own fixed seed internally, so repeated
    # calls must agree regardless of any external RNG state.
    rng = np.random.default_rng(12345)
    rng.random(1000)  # perturb some unrelated external RNG state
    a = held_out_workloads(10)
    b = held_out_workloads(10)
    assert [w.seq_len for w in a] == [w.seq_len for w in b]


def test_workloads_to_batch_stacks_all_fields():
    rng = np.random.default_rng(0)
    workloads = sample_workloads(5, rng)
    batch = workloads_to_batch(workloads)
    assert batch["features"].shape == (5, FEATURE_DIM)
    for key in ("seq_len", "head_dim", "num_heads", "batch_size", "sram_size_bytes"):
        assert batch[key].shape == (5,)


def test_tile_size_bounds_are_sane():
    assert TILE_SIZE_MIN > 0
    assert TILE_SIZE_MAX > TILE_SIZE_MIN
