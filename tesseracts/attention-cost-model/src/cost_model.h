// Copyright 2026. Tesseract Hackathon submission.
// Analytical I/O-complexity cost model for FlashAttention-style tiled
// attention kernels, in the spirit of the FlashAttention paper's HBM
// access-count analysis.
#pragma once

extern "C" {

// All sizes are in elements unless noted otherwise. dtype_bytes gives the
// element width (e.g. 2 for fp16/bf16, 4 for fp32) so HBM traffic is
// reported in bytes.
//
// Br, Bc are left as `double` (not `int`) even though they represent tile
// sizes in rows/columns, because Tesseract A is invoked with continuous,
// non-integer tile sizes during gradient-based training (Tesseract B emits
// a continuous relaxation of Br/Bc). The tile-count terms below use ceil(),
// which is itself a source of the intentional non-smoothness in this model.
void compute_attention_cost(
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
    double* out_sram_utilization);

}  // extern "C"
