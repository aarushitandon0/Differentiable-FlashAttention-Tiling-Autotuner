# Copyright 2026. Tesseract Hackathon submission.
"""Shared helpers for the grid-search / random-search baselines.

Both baselines search directly against Tesseract A (the cost model) via its
plain `apply` endpoint -- no JAX, no gradients, just repeated black-box
evaluation, which is exactly what a Triton autotuning sweep or a random
search over kernel configs does in practice.
"""

from __future__ import annotations

from tesseract_core import Tesseract

from policy.sample_workloads import Workload
from policy.train import make_cost_model_client


def workload_apply_inputs(workload: Workload, br: float, bc: float) -> dict:
    return {
        "Br": br,
        "Bc": bc,
        "seq_len": workload.seq_len,
        "head_dim": workload.head_dim,
        "num_heads": workload.num_heads,
        "batch_size": workload.batch_size,
        "sram_size_bytes": workload.sram_size_bytes,
        "hbm_bandwidth_gb_s": workload.hbm_bandwidth_gb_s,
        "compute_throughput_flops": workload.compute_throughput_flops,
        "dtype_bytes": workload.dtype_bytes,
    }


def evaluate_latency(client: Tesseract, workload: Workload, br: float, bc: float) -> float:
    out = client.apply(workload_apply_inputs(workload, br, bc))
    return float(out["predicted_latency_us"])


def make_client() -> Tesseract:
    return make_cost_model_client()
