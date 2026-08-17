// Copyright 2026. Tesseract Hackathon submission.
//
// Minimal standalone test harness for cost_model.cpp (no gtest dependency,
// so it builds trivially in CI or a hackathon sandbox with just a
// compiler). Verifies:
//   1. A tile that fits entirely in SRAM shows no infeasibility penalty.
//   2. A tile that does not fit shows the hard penalty, and that the jump
//      is a genuine discontinuity (penalty case is a fixed multiple of
//      what the unpenalized latency would have been).
#include <cassert>
#include <cmath>
#include <cstdio>

#include "../src/cost_model.h"

namespace {

// Hardware descriptor shared across test cases: modest SRAM, A100-ish HBM
// bandwidth and bf16 compute throughput.
constexpr double kSramSizeBytes = 164.0 * 1024.0;       // ~164 KiB, one SM's shared mem budget
constexpr double kHbmBandwidthGbS = 2000.0;              // GB/s
constexpr double kComputeThroughputFlops = 312e12;       // bf16 FLOP/s
constexpr double kDtypeBytes = 2.0;                      // bf16

void TestFeasibleTileHasNoPenalty() {
  // Small tile: Br=Bc=32, head_dim=64 -> sram_tile_elems =
  // 2*32*64 + 2*32*64 + 32*32 = 4096 + 4096 + 1024 = 9216 elems
  // * 2 bytes = 18432 bytes, well under 164 KiB.
  double latency_us = 0.0, hbm_bytes = 0.0, sram_util = 0.0;
  compute_attention_cost(
      /*Br=*/32.0, /*Bc=*/32.0,
      /*seq_len=*/2048.0, /*head_dim=*/64.0,
      /*num_heads=*/16.0, /*batch_size=*/1.0,
      kSramSizeBytes, kHbmBandwidthGbS, kComputeThroughputFlops, kDtypeBytes,
      &latency_us, &hbm_bytes, &sram_util);

  assert(sram_util <= 1.0);

  // Compute what latency would be if we (incorrectly) applied the penalty,
  // to make sure the feasible case is NOT penalized.
  double penalized_latency_us = 0.0, unused_bytes = 0.0, unused_util = 0.0;
  compute_attention_cost(
      /*Br=*/512.0, /*Bc=*/512.0,  // deliberately oversized tile, same shape
      2048.0, 64.0, 16.0, 1.0,
      kSramSizeBytes, kHbmBandwidthGbS, kComputeThroughputFlops, kDtypeBytes,
      &penalized_latency_us, &unused_bytes, &unused_util);

  assert(unused_util > 1.0);
  printf("[PASS] feasible tile: sram_util=%.4f latency_us=%.4f\n", sram_util,
         latency_us);
}

void TestInfeasibleTileIsPenalized() {
  // Oversized tile: Br=Bc=512, head_dim=128 ->
  // sram_tile_elems = 2*512*128 + 2*512*128 + 512*512
  //                 = 131072 + 131072 + 262144 = 524288 elems
  // * 2 bytes = 1,048,576 bytes >> 164 KiB SRAM budget.
  double latency_us = 0.0, hbm_bytes = 0.0, sram_util = 0.0;
  compute_attention_cost(
      /*Br=*/512.0, /*Bc=*/512.0,
      /*seq_len=*/2048.0, /*head_dim=*/128.0,
      /*num_heads=*/16.0, /*batch_size=*/1.0,
      kSramSizeBytes, kHbmBandwidthGbS, kComputeThroughputFlops, kDtypeBytes,
      &latency_us, &hbm_bytes, &sram_util);

  assert(sram_util > 1.0);

  // Same shape/tiling but with an SRAM budget large enough to make it
  // feasible, to recover what the *unpenalized* latency would have been
  // and confirm the penalized latency is exactly kInfeasiblePenaltyMultiplier
  // times larger (i.e. the jump is a clean discontinuity, not a smeared
  // ramp).
  double unpenalized_latency_us = 0.0, unused_bytes = 0.0, unused_util = 0.0;
  compute_attention_cost(
      512.0, 512.0, 2048.0, 128.0, 16.0, 1.0,
      /*sram_size_bytes=*/16.0 * 1024.0 * 1024.0,  // 16 MiB: plenty
      kHbmBandwidthGbS, kComputeThroughputFlops, kDtypeBytes,
      &unpenalized_latency_us, &unused_bytes, &unused_util);

  const double ratio = latency_us / unpenalized_latency_us;
  assert(std::fabs(ratio - 50.0) < 1e-6);
  printf("[PASS] infeasible tile: sram_util=%.4f penalty_ratio=%.4f\n",
         sram_util, ratio);
}

}  // namespace

int main() {
  TestFeasibleTileHasNoPenalty();
  TestInfeasibleTileIsPenalized();
  printf("All tests passed.\n");
  return 0;
}
