// Copyright 2026. Tesseract Hackathon submission.
//
// Analytical I/O-complexity cost model for FlashAttention-style tiled
// attention, following the HBM access-count reasoning in the FlashAttention
// paper (Dao et al. 2022): for a given row/column tile size (Br, Bc), the
// outer loop over row-tiles re-reads all of K and V from HBM once per
// row-tile, while Q and O are each touched exactly once. Whether a tile
// size is legal at all is gated by a hard SRAM-capacity branch, which is
// intentionally left as a true step discontinuity rather than a smooth
// penalty (see the SRAM-feasibility branch below) -- this is precisely the
// kind of thing symbolic/analytic AD handles poorly (zero or undefined
// gradient across the branch) but finite differences handle uniformly,
// since FD only ever needs to *evaluate* the function, never differentiate
// it symbolically.
#include "cost_model.h"

#include <algorithm>

namespace {

// Multiplicative jump applied when a tile does not fit in SRAM. This is a
// genuine discontinuity in (Br, Bc): infinitesimally crossing the SRAM
// boundary multiplies latency by this factor, it does not ramp smoothly.
constexpr double kInfeasiblePenaltyMultiplier = 50.0;

double SafePositive(double x) { return std::max(x, 1e-6); }

}  // namespace

extern "C" void compute_attention_cost(
    double Br,
    double Bc,
    double seq_len,
    double head_dim,
    double num_heads,
    double batch_size,
    double sram_size_bytes,
    double hbm_bandwidth_gb_s,
    double compute_throughput_flops,
    double dtype_bytes,
    double* out_predicted_latency_us,
    double* out_hbm_bytes_moved,
    double* out_sram_utilization) {
  Br = SafePositive(Br);
  Bc = SafePositive(Bc);
  seq_len = SafePositive(seq_len);
  head_dim = SafePositive(head_dim);

  // --- SRAM occupancy per tile (the deliberately non-smooth part) -------
  // Per FlashAttention's SRAM budget: a Q tile (Br x d), a K tile and a V
  // tile (Bc x d each), a running O accumulator tile (Br x d), and the
  // scratch score/prob tile S/P (Br x Bc).
  const double sram_tile_elems =
      2.0 * Br * head_dim + 2.0 * Bc * head_dim + Br * Bc;
  const double sram_tile_bytes = sram_tile_elems * dtype_bytes;
  const bool feasible = sram_tile_bytes <= sram_size_bytes;

  // --- HBM traffic --------------------------------------------------------
  // Number of row-tiles the outer loop iterates over. This is deliberately
  // the continuous ratio seq_len / Br, NOT std::ceil(seq_len / Br): a real
  // kernel does loop over an integer number of tiles, but ceil() would make
  // this term piecewise-constant in Br (a "staircase" with zero derivative
  // almost everywhere and all of its gradient signal concentrated at
  // measure-zero exact-division points). That would swamp the one
  // discontinuity this model is deliberately built around -- the SRAM
  // feasibility branch above -- with a second, incidental family of
  // discontinuities. The continuous ratio is the same approximation the
  // FlashAttention paper's own asymptotic (big-O) I/O-complexity analysis
  // uses, and keeps this term's contribution to predicted latency smooth
  // and analytically differentiable, so FD gradients here track the true
  // continuous slope everywhere rather than being zero almost everywhere.
  const double num_row_tiles = seq_len / Br;

  const double per_head_elems_q_and_o = 2.0 * seq_len * head_dim;  // Q read + O write
  const double per_head_elems_kv_rereads =
      num_row_tiles * 2.0 * seq_len * head_dim;  // K,V re-read per row-tile
  const double per_head_elems = per_head_elems_q_and_o + per_head_elems_kv_rereads;

  const double hbm_bytes_moved =
      batch_size * num_heads * per_head_elems * dtype_bytes;

  // --- Compute -------------------------------------------------------------
  // QK^T and P@V each cost ~2*seq_len^2*head_dim FLOPs.
  const double flops =
      batch_size * num_heads * 4.0 * seq_len * seq_len * head_dim;

  // --- Roofline latency ------------------------------------------------
  const double memory_time_us =
      hbm_bytes_moved / (hbm_bandwidth_gb_s * 1e9) * 1e6;
  const double compute_time_us =
      flops / compute_throughput_flops * 1e6;
  const double base_latency_us = std::max(memory_time_us, compute_time_us);

  // Hard, discontinuous infeasibility branch -- NOT a smooth relu/softplus
  // penalty. This is the whole reason the cost model needs a
  // finite-difference AD wrapper instead of analytic/symbolic AD.
  const double predicted_latency_us =
      feasible ? base_latency_us : base_latency_us * kInfeasiblePenaltyMultiplier;

  *out_predicted_latency_us = predicted_latency_us;
  *out_hbm_bytes_moved = hbm_bytes_moved;
  *out_sram_utilization = sram_tile_bytes / sram_size_bytes;
}
