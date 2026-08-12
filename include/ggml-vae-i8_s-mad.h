#pragma once

#include "ggml.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


// INT8 × INT8 vec_dot implementation (output is int32 to avoid overflow)
void ggml_vec_dot_i8_i8(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

// Register-tiled INT8 GEMM:  s[r*bs + c] = dot(vy + r*n, vx + c*n)
//
// The vec_dot-per-(row-block, column) loop in ggml_gemm_i8_i8 re-reads one operand
// for every step of the other. This holds a 4x4 tile of int32 accumulators in
// registers so both operands are loaded once per tile, roughly doubling the ratio of
// multiply-accumulates to loads.
//
// It also drops the sign fold the vec_dot kernels need. vpdpbusd wants one unsigned
// operand, so those kernels compute |x| and push x's sign onto y -- per operand pair,
// which does not amortise across a tile. Instead this adds 128 to the weights (a
// sign-bit flip, making them unsigned) and corrects with -128*sum(y) per row, where
// the row sums are computed once per row block.
//
// Falls back to the existing vec_dot loop without AVX512-VNNI, and for the ragged
// edges when nr or nc is not a multiple of the tile.
void ggml_gemm_i8_i8_tiled(int n, int32_t * s, size_t bs, const void * vx, const void * vy, int nr, int nc);

// ---------------------------------------------------------------------------
// Vectorised epilogues for the fused I8_S ops.
//
// After every fused matmul/conv the runtime makes two full passes over the output:
// int32 -> float with scale and bias while tracking the absolute maximum, then
// float -> int8 with clamp and roundf. Those passes were scalar -- roundf is a libm
// call per element -- and they run on multi-megabyte activations at every VAE layer,
// which made them a large share of encoder time despite doing no matmul work.
//
// Semantics match the scalar loops exactly: mul-then-add (no FMA contraction), and
// round-half-away-from-zero implemented as trunc(v + copysign(0.5, v)), which agrees
// with roundf everywhere in the clamped [-127, 127] range.
// ---------------------------------------------------------------------------

// out[i] = acc[i]*scale + (bias ? bias[i] : bias_scalar); returns max |out[i]|.
float vibeasr_i8s_dequant_absmax(const int32_t * acc, int64_t n, float scale,
                                 const float * bias, float bias_scalar, float * out);

// Fused GEMM + dequant epilogue for the I8_S linear ops:
//   out[r*ldc + c] = dot(vy row r, vx column c) * combined_scale + bias[c]
// over nr activation rows and nc weight columns; returns max |out| over the block.
// On VNNI hosts this runs a packed-B kernel (weights repacked once per tensor,
// cached) whose epilogue writes float directly from the accumulators -- the int32
// intermediate, its memset and the separate dequant pass all disappear; results are
// bit-identical to the unfused sequence, which VIBEASR_GEMM_PACKED=0 restores.
float vibeasr_gemm_i8_f32(int n, const void * vx, const void * vy, int nr, int nc,
                          float combined_scale, const float * bias, float * out,
                          int64_t ldc);

// out[i] = (int8) roundf(clamp(in[i]*inv_scale, -127, 127)); relu clamps at 0.
void vibeasr_i8s_quant_i8(const float * in, int8_t * out, int64_t n,
                          float inv_scale, int relu);

// The ADD_SCALED body: out[i] = a[i]/a_scale * gamma + b[i]/b_scale over span
// [start, start+n) of a tensor with row length ne0; returns max |out[i]|. Uniform
// per element regardless of how callers chunk the span -- see the implementation
// for why that is a determinism requirement.
float vibeasr_i8s_add_scaled_absmax(const int8_t * a, const int8_t * b,
                                    const float * gamma, int64_t gamma_ne0,
                                    int64_t ne0, int64_t start, int64_t n,
                                    float a_scale, float b_scale, float * out);

// Layout-native causal depthwise conv over columns [t0, t1) of x [C, frames],
// C contiguous: out[c + t*C] = sum_j w[c*k+j] * x[c + (t+j-k+1)*C] * combined_scale
// + bias[c], zero-padded on the left. Returns max |out| over the span. Replaces the
// permute/cont/im2col/matmul/cont chain of the ConvNeXt mixer.
float vibeasr_i8s_dwconv_absmax(const int8_t * x, const int8_t * w, const float * bias,
                                int64_t C, int k, int64_t frames, int64_t t0, int64_t t1,
                                float combined_scale, float * out);

// Graph builder for the dw_direct branch of GGML_OP_MUL_MAT_ADD: w [k,1,C] I8_S,
// x [C, frames] I8_S (the ConvNeXt layout, unpermuted), bias F32 [C] -> I8_S
// [C, frames]. Defined in the patched ggml.c.
struct ggml_context;
struct ggml_tensor * ggml_mul_mat_add_dw_direct(struct ggml_context * ctx,
                                                struct ggml_tensor  * w,
                                                struct ggml_tensor  * x,
                                                struct ggml_tensor  * bias);

// Optimized INT8 × INT8 vec_dot for n=4 (process 8 columns simultaneously)
void ggml_vec_dot_i8_i8_n4_col8(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc);

// Optimized INT8 × INT8 vec_dot for n=8 (process 4 columns simultaneously)
void ggml_vec_dot_i8_i8_n8_col4(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc);

// Optimized INT8 × INT8 vec_dot for n=16 (process 2 columns simultaneously)
void ggml_vec_dot_i8_i8_n16_col2(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc);

// Optimized INT8 × INT8 vec_dot for n=2 (process 16 columns simultaneously, AVX only)
void ggml_vec_dot_i8_i8_n2_col16(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc);

// Optimized INT8 × INT8 vec_dot for n=4 (process 2 columns simultaneously, ARM only)
void ggml_vec_dot_i8_i8_n4_col2(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc);

// Optimized INT8 × INT8 vec_dot for n=2 (process 4 columns simultaneously, ARM only)
void ggml_vec_dot_i8_i8_n2_col4(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc);

// Optimized INT8 × INT8 depthwise convolution kernel for n=8
// Typical shape: [8,1,32] x [8,579200,32]
// Process 4 columns at a time to fill 256-bit register
void ggml_vec_dot_i8_i8_batch_n8(
    int32_t * dst_data,
    const int8_t * weight_data,
    const int8_t * input_data,
    int64_t ne00,  // 8
    int64_t ne01,  // 1
    int64_t ne02,  // batch (32)
    int64_t ne10,  // 8
    int64_t ne11  // cols (579200)
    );

#ifdef __cplusplus
}
#endif
