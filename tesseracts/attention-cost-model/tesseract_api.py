# Copyright 2026. Tesseract Hackathon submission.
"""Tesseract wrapper around the FlashAttention-style tiling cost model.

The cost model itself (src/cost_model.cpp) is a small analytical
I/O-complexity model, compiled at image build time (see
tesseract_config.yaml) into a shared library and loaded here via ctypes.

Gradients are NOT hand-derived. The model's SRAM-capacity feasibility
branch is a genuine step discontinuity (see cost_model.cpp), which is
exactly the kind of thing symbolic/analytic AD tooling (e.g. plain JAX
autodiff through a `jnp.where`) handles badly -- it differentiates *through*
the branch as written, giving a zero or misleading local gradient that
ignores the possibility of crossing into/out of the feasible region.
Finite differences instead only ever need to *evaluate* compute_attention_cost
at nearby points, so they see the discontinuity's local numerical effect
directly, uniformly, without any special-casing of the branch. That's the
whole reason Tesseract A is wrapped with `finite_difference_*` here instead
of e.g. a JAX/Enzyme backend.
"""

import ctypes
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tesseract_core.runtime import Differentiable, Float64, ShapeDType
from tesseract_core.runtime.experimental import (
    finite_difference_jacobian,
    finite_difference_jvp,
    finite_difference_vjp,
)

#
# Load the compiled cost model shared library.
#
# Baked into the image at /tesseract/lib/cost_model.so by the
# custom_build_steps in tesseract_config.yaml. Falling back to a path next
# to this file lets `apply()` also work when running tests locally against
# a freshly-built .so without a full container build.
_LIB_CANDIDATES = [
    Path("/tesseract/lib/cost_model.so"),
    Path(__file__).parent / "lib" / "cost_model.so",
]
_lib_path = next((p for p in _LIB_CANDIDATES if p.exists()), None)
if _lib_path is None:
    raise FileNotFoundError(
        "Could not locate compiled cost_model shared library. Looked in: "
        f"{[str(p) for p in _LIB_CANDIDATES]}. Did the custom_build_steps "
        "in tesseract_config.yaml run?"
    )

_cost_model_lib = ctypes.CDLL(str(_lib_path))
_cost_model_lib.compute_attention_cost.argtypes = [
    ctypes.c_double,  # Br
    ctypes.c_double,  # Bc
    ctypes.c_double,  # seq_len
    ctypes.c_double,  # head_dim
    ctypes.c_double,  # num_heads
    ctypes.c_double,  # batch_size
    ctypes.c_double,  # sram_size_bytes
    ctypes.c_double,  # hbm_bandwidth_gb_s
    ctypes.c_double,  # compute_throughput_flops
    ctypes.c_double,  # dtype_bytes
    ctypes.POINTER(ctypes.c_double),  # out_predicted_latency_us
    ctypes.POINTER(ctypes.c_double),  # out_hbm_bytes_moved
    ctypes.POINTER(ctypes.c_double),  # out_sram_utilization
]
_cost_model_lib.compute_attention_cost.restype = None


#
# Schemas
#


class InputSchema(BaseModel):
    Br: Differentiable[Float64] = Field(
        description="Row (query) tile size. Continuous during gradient-based "
        "training; the calling policy is responsible for rounding to a "
        "hardware-valid tile size at deployment time."
    )
    Bc: Differentiable[Float64] = Field(
        description="Column (key/value) tile size. Continuous, see Br."
    )
    # These are not fields we ever ask for gradients w.r.t. (jac_inputs /
    # vjp_inputs during training only ever include {"Br", "Bc"}), but they
    # are still marked Differentiable so tesseract-jax's `apply_tesseract`
    # can trace and jax.vmap them as ordinary dynamic array inputs -- e.g.
    # to batch many different workload shapes through one Tesseract call
    # per training step. A plain (non-Differentiable) field would instead
    # be treated as static per-call metadata baked in at trace time, which
    # can't vary across a vmapped batch.
    seq_len: Differentiable[Float64] = Field(
        description="Sequence length (queries == keys/values)."
    )
    head_dim: Differentiable[Float64] = Field(description="Per-head hidden dimension.")
    num_heads: Differentiable[Float64] = Field(description="Number of attention heads.")
    batch_size: Differentiable[Float64] = Field(description="Batch size.")
    sram_size_bytes: Differentiable[Float64] = Field(
        description="Available on-chip SRAM / shared-memory budget per "
        "streaming multiprocessor, in bytes."
    )
    hbm_bandwidth_gb_s: Differentiable[Float64] = Field(
        description="HBM bandwidth of the target device, in GB/s."
    )
    compute_throughput_flops: Differentiable[Float64] = Field(
        description="Peak compute throughput of the target device for the "
        "kernel's numeric precision, in FLOP/s."
    )
    dtype_bytes: Differentiable[Float64] = Field(
        default=2.0,
        description="Element width of the attention kernel's working "
        "precision in bytes (e.g. 2 for fp16/bf16, 4 for fp32).",
    )


class OutputSchema(BaseModel):
    predicted_latency_us: Differentiable[Float64] = Field(
        description="Predicted kernel latency in microseconds: "
        "max(memory_bound_time, compute_bound_time), multiplied by a hard "
        "penalty factor when the (Br, Bc) tile does not fit in SRAM."
    )
    hbm_bytes_moved: Float64 = Field(
        description="Total HBM bytes moved across Q/K/V/O for this tiling, "
        "for debugging/plotting."
    )
    sram_utilization: Differentiable[Float64] = Field(
        description="Fraction (or multiple, if > 1) of the SRAM budget "
        "occupied by one tile's working set. Marked Differentiable because "
        "Tesseract B's training loop backpropagates a soft SRAM-infeasibility "
        "penalty through this output (on top of the cost model's own hard "
        "penalty baked into predicted_latency_us), not just for "
        "debugging/plotting."
    )


#
# Required endpoint
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Evaluate the analytical attention-tiling I/O-complexity cost model."""
    latency_us = ctypes.c_double()
    hbm_bytes = ctypes.c_double()
    sram_util = ctypes.c_double()

    _cost_model_lib.compute_attention_cost(
        inputs.Br,
        inputs.Bc,
        inputs.seq_len,
        inputs.head_dim,
        inputs.num_heads,
        inputs.batch_size,
        inputs.sram_size_bytes,
        inputs.hbm_bandwidth_gb_s,
        inputs.compute_throughput_flops,
        inputs.dtype_bytes,
        ctypes.byref(latency_us),
        ctypes.byref(hbm_bytes),
        ctypes.byref(sram_util),
    )

    return OutputSchema(
        predicted_latency_us=latency_us.value,
        hbm_bytes_moved=hbm_bytes.value,
        sram_utilization=sram_util.value,
    )


#
# Optional endpoints: finite-difference AD.
#
# We deliberately do NOT hand-derive analytic gradients here, and we
# deliberately do NOT smooth out the SRAM branch in cost_model.cpp to make
# analytic AD tractable -- the non-smoothness is the point (see module
# docstring). `finite_difference_*` treats `apply` as a black box and
# perturbs Br/Bc numerically.
#


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    return finite_difference_jacobian(
        apply, inputs, jac_inputs, jac_outputs, algorithm="central", eps=1e-3
    )


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return finite_difference_jvp(
        apply,
        inputs,
        jvp_inputs,
        jvp_outputs,
        tangent_vector,
        algorithm="central",
        eps=1e-3,
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return finite_difference_vjp(
        apply,
        inputs,
        vjp_inputs,
        vjp_outputs,
        cotangent_vector,
        algorithm="central",
        eps=1e-3,
    )


def abstract_eval(abstract_inputs):
    """Output shapes/dtypes are static (all scalars), independent of input shapes."""
    return {
        "predicted_latency_us": ShapeDType(shape=(), dtype="float64"),
        "hbm_bytes_moved": ShapeDType(shape=(), dtype="float64"),
        "sram_utilization": ShapeDType(shape=(), dtype="float64"),
    }
