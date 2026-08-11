#include <vector>
#include <type_traits>
#include <assert.h>
#include <cmath>
#include <cstring>
#include "ggml-vae-i8_s-mad.h"
#include "ggml-cpu-impl.h"
#include "lm-config.h"
#include "vae-config.h"
#include "vibeasr-cpu.h"
#include "vibeasr-kernel-stats.h"
#include <cstdlib>

#if defined(VAE_ACT_PARALLEL)
#define VAE_ACT_PARALLEL_SELECTED 1
#else
#define VAE_ACT_PARALLEL_SELECTED 0
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#define QK_I8_S 32
#elif defined(__ARM_NEON)
#define QK_I8_S 8
#else
#define QK_I8_S 32
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#include <immintrin.h>
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
#endif

// ---------------------------------------------------------------------------
// AVX-512 / AVX512-VNNI path
//
// The AVX2 kernels below multiply s8 x s8 by folding the sign of x into y
// (_mm256_sign_epi8), multiplying |x| as u8 through vpmaddubsw, accumulating in
// int16, and flushing to int32 every 32 blocks to stay ahead of int16 overflow.
//
// VNNI's vpdpbusd does u8 x s8 with an int32 accumulator in one instruction, so the
// int16 stage and its periodic flush disappear entirely, and the lane width doubles.
// The sign fold survives, using vpabsb plus a masked negate because AVX-512 has no
// vpsignb. That reproduces the AVX2 kernel bit for bit, including its behaviour at
// x = -128 (which quantize_i8_s never emits: it clamps to +-127).
//
// Selection happens at run time via vibeasr_isa(), so one binary still runs on AVX2.
// ---------------------------------------------------------------------------

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#define VIBEASR_HAS_AVX512_PATH 1
#define VIBEASR_TGT_VNNI   __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")))
#define VIBEASR_TGT_AVX512 __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl")))

VIBEASR_TGT_VNNI static inline int32_t hsum_i32_16(__m512i a) {
    return _mm512_reduce_add_epi32(a);
}

// acc += |x| (u8) * sign(x)*y (s8), 64 lanes at a time.
VIBEASR_TGT_VNNI static inline __m512i dp_signed(__m512i acc, __m512i x, __m512i y) {
    const __mmask64 neg = _mm512_movepi8_mask(x);
    const __m512i   ax  = _mm512_abs_epi8(x);
    const __m512i   sy  = _mm512_mask_sub_epi8(y, neg, _mm512_setzero_si512(), y);
    return _mm512_dpbusd_epi32(acc, ax, sy);
}

// Dot product over nbytes int8 lanes. Two accumulators hide vpdpbusd's latency.
VIBEASR_TGT_VNNI static int32_t vae_dot_vnni(const int8_t * px, const int8_t * py, int nbytes) {
    __m512i a0 = _mm512_setzero_si512();
    __m512i a1 = _mm512_setzero_si512();
    int i = 0;
    for (; i + 128 <= nbytes; i += 128) {
        a0 = dp_signed(a0, _mm512_loadu_si512((const void *)(px + i)),
                           _mm512_loadu_si512((const void *)(py + i)));
        a1 = dp_signed(a1, _mm512_loadu_si512((const void *)(px + i + 64)),
                           _mm512_loadu_si512((const void *)(py + i + 64)));
    }
    for (; i + 64 <= nbytes; i += 64) {
        a0 = dp_signed(a0, _mm512_loadu_si512((const void *)(px + i)),
                           _mm512_loadu_si512((const void *)(py + i)));
    }
    if (i < nbytes) {  // 32-byte remainder: same op under a lane mask
        const __mmask64 m = (__mmask64)((1ULL << (nbytes - i)) - 1);
        a1 = dp_signed(a1, _mm512_maskz_loadu_epi8(m, px + i),
                           _mm512_maskz_loadu_epi8(m, py + i));
    }
    return hsum_i32_16(_mm512_add_epi32(a0, a1));
}

// nblk independent dot products that share one operand -- the shape both the 1xN
// (shared y) and Nx1 (shared x) blocked kernels need. Loading the shared vector once
// per step and reusing it across the block is most of the win at these small n.
//
// sign_from_shared says which operand donates the sign, matching whichever AVX2
// kernel this is standing in for. The choice is arithmetically irrelevant, but
// keeping it aligned makes the two paths bit-identical even at x = -128.
VIBEASR_TGT_VNNI static void vae_dot_vnni_shared(
        const int8_t * shared, const int8_t * const * others,
        int nbytes, int nblk, int sign_from_shared, int32_t * out) {
    __m512i acc[VAE_PARALLEL_SIZE];
    for (int b = 0; b < nblk; b++) acc[b] = _mm512_setzero_si512();

    int i = 0;
    for (; i + 64 <= nbytes; i += 64) {
        const __m512i sv = _mm512_loadu_si512((const void *)(shared + i));
        for (int b = 0; b < nblk; b++) {
            const __m512i ov = _mm512_loadu_si512((const void *)(others[b] + i));
            acc[b] = sign_from_shared ? dp_signed(acc[b], sv, ov) : dp_signed(acc[b], ov, sv);
        }
    }
    if (i < nbytes) {
        const __mmask64 m = (__mmask64)((1ULL << (nbytes - i)) - 1);
        const __m512i sv = _mm512_maskz_loadu_epi8(m, shared + i);
        for (int b = 0; b < nblk; b++) {
            const __m512i ov = _mm512_maskz_loadu_epi8(m, others[b] + i);
            acc[b] = sign_from_shared ? dp_signed(acc[b], sv, ov) : dp_signed(acc[b], ov, sv);
        }
    }
    for (int b = 0; b < nblk; b++) out[b] = hsum_i32_16(acc[b]);
}
#endif  // x86

// ---------------------------------------------------------------------------
// Vectorised epilogues for the fused I8_S ops (see header for context).
// ---------------------------------------------------------------------------

#if defined(VIBEASR_HAS_AVX512_PATH)

VIBEASR_TGT_AVX512 static float dequant_absmax_avx512(
        const int32_t * acc, int64_t n, float scale,
        const float * bias, float bias_scalar, float * out) {
    const __m512 vscale = _mm512_set1_ps(scale);
    const __m512 vbias_s = _mm512_set1_ps(bias_scalar);
    const __m512 signmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    __m512 vmax = _mm512_setzero_ps();

    // The tail is handled with masked full-width ops rather than a scalar loop.
    // This is a determinism requirement, not a micro-optimisation: gcc contracts a
    // scalar tail's mul+add into fma while the vector body rounds the intermediate
    // product, and the body/tail boundary moves with the caller's thread partition.
    // That skew showed up as transcripts changing between -t 2 and -t 4, first
    // diverging by one quantisation step in a semantic-encoder stage-3 block. With
    // every element on the identical instruction path, the result is independent of
    // how callers chunk the range.
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_cvtepi32_ps(_mm512_loadu_si512((const void *)(acc + i)));
        // mul then add, not fmadd: the original scalar loop rounds the intermediate
        // product, and bit-identical output was the acceptance test for the rewrite.
        v = _mm512_mul_ps(v, vscale);
        v = _mm512_add_ps(v, bias ? _mm512_loadu_ps(bias + i) : vbias_s);
        _mm512_storeu_ps(out + i, v);
        vmax = _mm512_max_ps(vmax, _mm512_and_ps(v, signmask));
    }
    if (i < n) {
        const __mmask16 m = (__mmask16)((1u << (n - i)) - 1);
        __m512 v = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(m, acc + i));
        v = _mm512_mul_ps(v, vscale);
        v = _mm512_add_ps(v, bias ? _mm512_maskz_loadu_ps(m, bias + i) : vbias_s);
        _mm512_mask_storeu_ps(out + i, m, v);
        vmax = _mm512_mask_max_ps(vmax, m, vmax, _mm512_and_ps(v, signmask));
    }
    return _mm512_reduce_max_ps(vmax);
}

VIBEASR_TGT_AVX512 static void quant_i8_avx512(
        const float * in, int8_t * out, int64_t n, float inv_scale, int relu) {
    const __m512 vs   = _mm512_set1_ps(inv_scale);
    const __m512 vlo  = _mm512_set1_ps(-127.0f);
    const __m512 vhi  = _mm512_set1_ps(127.0f);
    const __m512 vhalf = _mm512_set1_ps(0.5f);
    const __m512 vsign = _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000));
    const __m128i zero8 = _mm_setzero_si128();

    // Masked tail for the same determinism reason as dequant_absmax_avx512: a
    // scalar tail rounds differently (roundf vs this sequence under contraction)
    // and the boundary moves with the caller's chunking.
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_mul_ps(_mm512_loadu_ps(in + i), vs);
        v = _mm512_min_ps(_mm512_max_ps(v, vlo), vhi);
        // round half away from zero, as roundf does: trunc(v + copysign(0.5, v))
        const __m512 half = _mm512_or_ps(vhalf, _mm512_and_ps(v, vsign));
        __m512i q = _mm512_cvttps_epi32(_mm512_add_ps(v, half));
        __m128i b = _mm512_cvtsepi32_epi8(q);  // saturating, but |q| <= 127 already
        if (relu) b = _mm_max_epi8(b, zero8);
        _mm_storeu_si128((__m128i *)(out + i), b);
    }
    if (i < n) {
        const __mmask16 m = (__mmask16)((1u << (n - i)) - 1);
        __m512 v = _mm512_mul_ps(_mm512_maskz_loadu_ps(m, in + i), vs);
        v = _mm512_min_ps(_mm512_max_ps(v, vlo), vhi);
        const __m512 half = _mm512_or_ps(vhalf, _mm512_and_ps(v, vsign));
        __m512i q = _mm512_cvttps_epi32(_mm512_add_ps(v, half));
        __m128i b = _mm512_cvtsepi32_epi8(q);
        if (relu) b = _mm_max_epi8(b, zero8);
        _mm_mask_storeu_epi8(out + i, m, b);
    }
}

#endif  // VIBEASR_HAS_AVX512_PATH

#if defined(VIBEASR_HAS_AVX512_PATH)

// val[i] = a[i]*inv_a*gamma[(start+i) % ne0] + b[i]*inv_b -- the ADD_SCALED body,
// every element through the identical masked-AVX-512 path. Reciprocals are taken
// once here so callers cannot mix a divide-flavoured scalar loop with a
// reciprocal-flavoured vector loop, which is exactly the skew that made transcripts
// depend on thread count (the scalar/vector boundary moved with the partition).
VIBEASR_TGT_AVX512 static float add_scaled_avx512(
        const int8_t * a, const int8_t * b, const float * gamma_base,
        int64_t ne0, int64_t start, int64_t n,
        float inv_a, float inv_b, float * out) {
    const __m512 va = _mm512_set1_ps(inv_a);
    const __m512 vb = _mm512_set1_ps(inv_b);
    const __m512 signmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    __m512 vmax = _mm512_setzero_ps();

    int64_t i = 0;
    while (i < n) {
        const int64_t col = (start + i) % ne0;
        const int64_t seg = (ne0 - col) < (n - i) ? (ne0 - col) : (n - i);   // stay in row
        const int64_t take = seg < 16 ? seg : 16;
        const __mmask16 m = take == 16 ? (__mmask16) 0xffff
                                       : (__mmask16)((1u << take) - 1);
        const __m512 af = _mm512_cvtepi32_ps(
            _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(m, a + i)));
        const __m512 bf = _mm512_cvtepi32_ps(
            _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(m, b + i)));
        const __m512 gf = gamma_base ? _mm512_maskz_loadu_ps(m, gamma_base + col)
                                     : _mm512_set1_ps(1.0f);
        __m512 v = _mm512_mul_ps(_mm512_mul_ps(af, va), gf);
        v = _mm512_add_ps(v, _mm512_mul_ps(bf, vb));
        _mm512_mask_storeu_ps(out + i, m, v);
        vmax = _mm512_mask_max_ps(vmax, m, vmax, _mm512_and_ps(v, signmask));
        i += take;
    }
    return _mm512_reduce_max_ps(vmax);
}

#endif  // VIBEASR_HAS_AVX512_PATH

float vibeasr_i8s_add_scaled_absmax(const int8_t * a, const int8_t * b,
                                    const float * gamma, int64_t gamma_ne0,
                                    int64_t ne0, int64_t start, int64_t n,
                                    float a_scale, float b_scale, float * out) {
#if defined(VIBEASR_HAS_AVX512_PATH)
    if (vibeasr_isa() >= VIBEASR_ISA_AVX512) {
        // scalar gamma broadcasts by folding it into inv_a
        const int per_channel = gamma_ne0 != 1;
        const float inv_a = per_channel ? 1.0f / a_scale : gamma[0] / a_scale;
        return add_scaled_avx512(a, b, per_channel ? gamma : NULL, ne0, start, n,
                                 inv_a, 1.0f / b_scale, out);
    }
#endif
    const float inv_a = 1.0f / a_scale, inv_b = 1.0f / b_scale;
    float amax = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        const float g = gamma_ne0 == 1 ? gamma[0] : gamma[(start + i) % ne0];
        const float v = (float) a[i] * inv_a * g + (float) b[i] * inv_b;
        out[i] = v;
        const float av = v < 0.0f ? -v : v;
        if (av > amax) amax = av;
    }
    return amax;
}

float vibeasr_i8s_dequant_absmax(const int32_t * acc, int64_t n, float scale,
                                 const float * bias, float bias_scalar, float * out) {
#if defined(VIBEASR_HAS_AVX512_PATH)
    if (vibeasr_isa() >= VIBEASR_ISA_AVX512) {
        return dequant_absmax_avx512(acc, n, scale, bias, bias_scalar, out);
    }
#endif
    float amax = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float v = (float) acc[i] * scale + (bias ? bias[i] : bias_scalar);
        out[i] = v;
        float av = v < 0.0f ? -v : v;
        if (av > amax) amax = av;
    }
    return amax;
}

void vibeasr_i8s_quant_i8(const float * in, int8_t * out, int64_t n,
                          float inv_scale, int relu) {
#if defined(VIBEASR_HAS_AVX512_PATH)
    if (vibeasr_isa() >= VIBEASR_ISA_AVX512) {
        quant_i8_avx512(in, out, n, inv_scale, relu);
        return;
    }
#endif
    for (int64_t i = 0; i < n; i++) {
        float v = in[i] * inv_scale;
        v = v < -127.0f ? -127.0f : (v > 127.0f ? 127.0f : v);
        int8_t q = (int8_t) roundf(v);
        out[i] = relu && q < 0 ? 0 : q;
    }
}

void ggml_vec_dot_i8_i8_1x1(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(VIBEASR_HAS_AVX512_PATH)
    if (vibeasr_isa() >= VIBEASR_ISA_VNNI) {
        // The AVX2 kernel consumes whole QK_I8_S blocks and ignores any tail; match it.
        const int nbytes = (n / QK_I8_S) * QK_I8_S;
        const int8_t * x = (const int8_t *) vx;
        const int8_t * y = (const int8_t *) vy;
        for (int row = 0; row < nrc; row++) {
            s[row] = vae_dot_vnni(x + row * bx, y, nbytes);
        }
        return;
    }
#endif
#if defined(__AVX2__) || defined(__AVX__)
    const int8_t * x = (int8_t *)vx;
    const int8_t * y = (int8_t *)vy;

    const int nb = n / QK_I8_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    const __m256i one16 = _mm256_set1_epi16(1);

    for (int row = 0; row < nrc; row++) {

        __m256i accu = _mm256_setzero_si256();
        const int8_t * x_row = x + row * bx;

        for (int i = 0; i < group32_num; i++) {
            const int8_t * px = x_row + i * 1024;
            const int8_t * py = y + i * 1024;
            __m256i accu32 = _mm256_setzero_si256();

            for (int j = 0; j < 32; j++) {
                __m256i xq8 = _mm256_loadu_si256((const __m256i*)(px));
                __m256i yq8 = _mm256_loadu_si256((const __m256i*)(py));

                const __m256i ax = _mm256_sign_epi8(xq8, xq8);
                const __m256i sy = _mm256_sign_epi8(yq8, xq8);
                __m256i dot = _mm256_maddubs_epi16(ax, sy);

                accu32 = _mm256_add_epi16(accu32, dot);

                px += 32;
                py += 32;
            }
            accu = _mm256_add_epi32(_mm256_madd_epi16(accu32, one16), accu);
        }

        for (int i = 0; i < groupla_num; i++) {
            __m256i accula = _mm256_setzero_si256();
            const int8_t * px = x_row + group32_num * 1024;
            const int8_t * py = y + group32_num * 1024;

            for (int j = 0; j < la_num; j++) {
                __m256i xq8 = _mm256_loadu_si256((const __m256i*)(px));
                __m256i yq8 = _mm256_loadu_si256((const __m256i*)(py));

                const __m256i ax = _mm256_sign_epi8(xq8, xq8);
                const __m256i sy = _mm256_sign_epi8(yq8, xq8);
                __m256i dot = _mm256_maddubs_epi16(ax, sy);

                accula = _mm256_add_epi16(accula, dot);

                px += 32;
                py += 32;
            }
            accu = _mm256_add_epi32(accu, _mm256_madd_epi16(accula, one16));
        }

        int sumi = hsum_i32_8(accu);
        s[row] = sumi;
    }
#elif defined(__ARM_NEON)
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;

    const int nb = n / QK_I8_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = la_num != 0 ? 1 : 0;

    for (int row = 0; row < nrc; row++) {
        int32x4_t accu = vdupq_n_s32(0);
        const int8_t * x_row = x + row * bx;

        for (int i = 0; i < group32_num; i++) {
            const int8_t * px = x_row + i * 32 * QK_I8_S;
            const int8_t * py = y + i * 32 * QK_I8_S;
#if defined(__ARM_FEATURE_DOTPROD)
            for (int j = 0; j < 32; j++) {
                int8x8_t xv = vld1_s8(px);
                int8x8_t yv = vld1_s8(py);
                int32x2_t d = vdot_s32(vdup_n_s32(0), xv, yv);
                accu = vcombine_s32(vadd_s32(vget_low_s32(accu), d), vget_high_s32(accu));
                px += QK_I8_S;
                py += QK_I8_S;
            }
#else
            int16x8_t accu16 = vdupq_n_s16(0);
            for (int j = 0; j < 32; j++) {
                int8x8_t xv = vld1_s8(px);
                int8x8_t yv = vld1_s8(py);
                accu16 = vmlal_s8(accu16, xv, yv);
                px += QK_I8_S;
                py += QK_I8_S;
            }
            accu = vaddq_s32(accu, vmovl_s16(vget_low_s16(accu16)));
            accu = vaddq_s32(accu, vmovl_high_s16(accu16));
#endif
        }

        for (int i = 0; i < groupla_num; i++) {
            const int8_t * px = x_row + group32_num * 32 * QK_I8_S;
            const int8_t * py = y + group32_num * 32 * QK_I8_S;
#if defined(__ARM_FEATURE_DOTPROD)
            for (int j = 0; j < la_num; j++) {
                int8x8_t xv = vld1_s8(px);
                int8x8_t yv = vld1_s8(py);
                int32x2_t d = vdot_s32(vdup_n_s32(0), xv, yv);
                accu = vcombine_s32(vadd_s32(vget_low_s32(accu), d), vget_high_s32(accu));
                px += QK_I8_S;
                py += QK_I8_S;
            }
#else
            int16x8_t accu16la = vdupq_n_s16(0);
            for (int j = 0; j < la_num; j++) {
                int8x8_t xv = vld1_s8(px);
                int8x8_t yv = vld1_s8(py);
                accu16la = vmlal_s8(accu16la, xv, yv);
                px += QK_I8_S;
                py += QK_I8_S;
            }
            accu = vaddq_s32(accu, vmovl_s16(vget_low_s16(accu16la)));
            accu = vaddq_s32(accu, vmovl_high_s16(accu16la));
#endif
        }

        s[row] = vaddlvq_s32(accu);
    }
#else
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;
    for (int row = 0; row < nrc; row++) {
        const int8_t * x_row = x + row * bx;
        int32_t sumi = 0;
        for (int k = 0; k < n; k++) {
            sumi += (int32_t)x_row[k] * (int32_t)y[k];
        }
        s[row] = sumi;
    }
#endif
}

void ggml_vec_dot_i8_i8_1xN(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(VIBEASR_HAS_AVX512_PATH)
    if (vibeasr_isa() >= VIBEASR_ISA_VNNI) {
        const int nbytes = (n / QK_I8_S) * QK_I8_S;
        const int8_t * x = (const int8_t *) vx;
        const int8_t * y = (const int8_t *) vy;
        for (int row = 0; row < nrc; row += VAE_PARALLEL_SIZE) {
            const int8_t * rows[VAE_PARALLEL_SIZE];
            int32_t out[VAE_PARALLEL_SIZE];
            for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) rows[rb] = x + (row + rb) * bx;
            vae_dot_vnni_shared(y, rows, nbytes, VAE_PARALLEL_SIZE, /*sign_from_shared=*/0, out);
            for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) s[row + rb] = out[rb];
        }
        return;
    }
#endif
#if defined(__AVX2__) || defined(__AVX__)
    const int8_t * x = (int8_t *)vx;
    const int8_t * y = (int8_t *)vy;

    const int nb = n / QK_I8_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    const __m256i one16 = _mm256_set1_epi16(1);

    for (int row = 0; row < nrc; row += VAE_PARALLEL_SIZE) {
        __m256i accu[VAE_PARALLEL_SIZE];
        const int8_t * x_row[VAE_PARALLEL_SIZE];
        for(int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
            accu[rb] = _mm256_setzero_si256();
            x_row[rb] = x + (row + rb) * bx;
        }

        for (int i = 0; i < group32_num; i++) {
            const int8_t * px[VAE_PARALLEL_SIZE];
            __m256i accu32[VAE_PARALLEL_SIZE];

            for(int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                px[rb] = x_row[rb] + i * 1024;
                accu32[rb] = _mm256_setzero_si256();
            }

            const int8_t * py = y + i * 1024;

            for (int j = 0; j < 32; j++) {

                __m256i yq8 = _mm256_loadu_si256((const __m256i*)(py));

                for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++)
                {
                    __m256i xq8 = _mm256_loadu_si256((const __m256i*)(px[rb]));

                    const __m256i ax = _mm256_sign_epi8(xq8, xq8);
                    const __m256i sy = _mm256_sign_epi8(yq8, xq8);
                    __m256i dot = _mm256_maddubs_epi16(ax, sy);

                    accu32[rb] = _mm256_add_epi16(accu32[rb], dot);

                    px[rb] += 32;
                }
                py += 32;
            }
            for(int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                accu[rb] = _mm256_add_epi32(_mm256_madd_epi16(accu32[rb], one16), accu[rb]);
            }
        }

        for (int i = 0; i < groupla_num; i++) {

            const int8_t * py = y + group32_num * 1024;
            const int8_t * px[VAE_PARALLEL_SIZE];
            __m256i accula[VAE_PARALLEL_SIZE];

            for(int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                px[rb] = x_row[rb] + group32_num * 1024;
                accula[rb] = _mm256_setzero_si256();
            }

            for (int j = 0; j < la_num; j++) {

                __m256i yq8 = _mm256_loadu_si256((const __m256i*)(py));

                for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {

                    __m256i xq8 = _mm256_loadu_si256((const __m256i*)(px[rb]));

                    const __m256i ax = _mm256_sign_epi8(xq8, xq8);
                    const __m256i sy = _mm256_sign_epi8(yq8, xq8);
                    __m256i dot = _mm256_maddubs_epi16(ax, sy);

                    accula[rb] = _mm256_add_epi16(accula[rb], dot);

                    px[rb] += 32;
                }
                py += 32;
            }
            for(int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                accu[rb] = _mm256_add_epi32(accu[rb], _mm256_madd_epi16(accula[rb], one16));
            }
        }

        for(int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
            int sumi = hsum_i32_8(accu[rb]);
            s[row + rb] = sumi;
        }
    }
#elif defined(__ARM_NEON)
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;

    const int nb = n / QK_I8_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = la_num != 0 ? 1 : 0;

    for (int row = 0; row < nrc; row += VAE_PARALLEL_SIZE) {
        int32x4_t accu[VAE_PARALLEL_SIZE];
        const int8_t * x_row[VAE_PARALLEL_SIZE];
        for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
            accu[rb] = vdupq_n_s32(0);
            x_row[rb] = x + (row + rb) * bx;
        }

        for (int i = 0; i < group32_num; i++) {
            const int8_t * py = y + i * 32 * QK_I8_S;
#if defined(__ARM_FEATURE_DOTPROD)
            const int8_t * px[VAE_PARALLEL_SIZE];
            for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                px[rb] = x_row[rb] + i * 32 * QK_I8_S;
            }
            for (int j = 0; j < 32; j++) {
                int8x8_t yv = vld1_s8(py);
                for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                    int8x8_t xv = vld1_s8(px[rb]);
                    int32x2_t d = vdot_s32(vdup_n_s32(0), xv, yv);
                    accu[rb] = vcombine_s32(vadd_s32(vget_low_s32(accu[rb]), d), vget_high_s32(accu[rb]));
                    px[rb] += QK_I8_S;
                }
                py += QK_I8_S;
            }
#else
            int16x8_t accu16[VAE_PARALLEL_SIZE];
            const int8_t * px[VAE_PARALLEL_SIZE];
            for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                accu16[rb] = vdupq_n_s16(0);
                px[rb] = x_row[rb] + i * 32 * QK_I8_S;
            }
            for (int j = 0; j < 32; j++) {
                int8x8_t yv = vld1_s8(py);
                for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                    int8x8_t xv = vld1_s8(px[rb]);
                    accu16[rb] = vmlal_s8(accu16[rb], xv, yv);
                    px[rb] += QK_I8_S;
                }
                py += QK_I8_S;
            }
            for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                accu[rb] = vaddq_s32(accu[rb], vmovl_s16(vget_low_s16(accu16[rb])));
                accu[rb] = vaddq_s32(accu[rb], vmovl_high_s16(accu16[rb]));
            }
#endif
        }

        for (int i = 0; i < groupla_num; i++) {
            const int8_t * py = y + group32_num * 32 * QK_I8_S;
#if defined(__ARM_FEATURE_DOTPROD)
            const int8_t * px[VAE_PARALLEL_SIZE];
            for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                px[rb] = x_row[rb] + group32_num * 32 * QK_I8_S;
            }
            for (int j = 0; j < la_num; j++) {
                int8x8_t yv = vld1_s8(py);
                for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                    int8x8_t xv = vld1_s8(px[rb]);
                    int32x2_t d = vdot_s32(vdup_n_s32(0), xv, yv);
                    accu[rb] = vcombine_s32(vadd_s32(vget_low_s32(accu[rb]), d), vget_high_s32(accu[rb]));
                    px[rb] += QK_I8_S;
                }
                py += QK_I8_S;
            }
#else
            int16x8_t accu16la[VAE_PARALLEL_SIZE];
            const int8_t * px[VAE_PARALLEL_SIZE];
            for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                accu16la[rb] = vdupq_n_s16(0);
                px[rb] = x_row[rb] + group32_num * 32 * QK_I8_S;
            }
            for (int j = 0; j < la_num; j++) {
                int8x8_t yv = vld1_s8(py);
                for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                    int8x8_t xv = vld1_s8(px[rb]);
                    accu16la[rb] = vmlal_s8(accu16la[rb], xv, yv);
                    px[rb] += QK_I8_S;
                }
                py += QK_I8_S;
            }
            for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                accu[rb] = vaddq_s32(accu[rb], vmovl_s16(vget_low_s16(accu16la[rb])));
                accu[rb] = vaddq_s32(accu[rb], vmovl_high_s16(accu16la[rb]));
            }
#endif
        }

        for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
            s[row + rb] = vaddlvq_s32(accu[rb]);
        }
    }
#else
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;
    for (int row = 0; row < nrc; row++) {
        const int8_t * x_row = x + row * bx;
        int32_t sumi = 0;
        for (int k = 0; k < n; k++) {
            sumi += (int32_t)x_row[k] * (int32_t)y[k];
        }
        s[row] = sumi;
    }
#endif
}

void ggml_vec_dot_i8_i8_Nx1(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(VIBEASR_HAS_AVX512_PATH)
    if (vibeasr_isa() >= VIBEASR_ISA_VNNI) {
        const int nbytes = (n / QK_I8_S) * QK_I8_S;
        const int8_t * x = (const int8_t *) vx;
        const int8_t * y = (const int8_t *) vy;
        for (int col = 0; col < nrc; col += VAE_PARALLEL_SIZE) {
            const int8_t * cols[VAE_PARALLEL_SIZE];
            int32_t out[VAE_PARALLEL_SIZE];
            for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) cols[cb] = y + (col + cb) * by;
            vae_dot_vnni_shared(x, cols, nbytes, VAE_PARALLEL_SIZE, /*sign_from_shared=*/1, out);
            for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) s[(col + cb) * bs] = out[cb];
        }
        return;
    }
#endif
#if defined(__AVX2__) || defined(__AVX__)
    const int8_t * x = (int8_t *)vx;
    const int8_t * y = (int8_t *)vy;

    const int nb = n / QK_I8_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    const __m256i one16 = _mm256_set1_epi16(1);

    for (int col = 0; col < nrc; col += VAE_PARALLEL_SIZE) {

        __m256i accu[VAE_PARALLEL_SIZE];

        for(int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
            accu[cb] = _mm256_setzero_si256();
        }

        for (int i = 0; i < group32_num; i++) {

            __m256i accu32[VAE_PARALLEL_SIZE];

            for(int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                accu32[cb] = _mm256_setzero_si256();
            }

            for (int j = 0; j < 32; j++) {

                const int8_t * px = x + (i * 32 + j) * 32;

                __m256i xq8 = _mm256_loadu_si256((const __m256i*)(px));

                for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {

                    const int8_t * py = y + (col + cb) * by + (i * 32 + j) * 32;

                    __m256i yq8 = _mm256_loadu_si256((const __m256i*)(py));

                    const __m256i ax = _mm256_sign_epi8(xq8, xq8);
                    const __m256i sy = _mm256_sign_epi8(yq8, xq8);
                    __m256i dot = _mm256_maddubs_epi16(ax, sy);

                    accu32[cb] = _mm256_add_epi16(accu32[cb], dot);
                }
            }

            for(int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                accu[cb] = _mm256_add_epi32(_mm256_madd_epi16(accu32[cb], one16), accu[cb]);
            }
        }

        for (int i = 0; i < groupla_num; i++) {

            __m256i accula[VAE_PARALLEL_SIZE];

            for(int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                accula[cb] = _mm256_setzero_si256();
            }

            for (int j = 0; j < la_num; j++) {

                const int8_t * px = x + (group32_num * 32 + j) * 32;

                __m256i xq8 = _mm256_loadu_si256((const __m256i*)(px));

                for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {

                    const int8_t * py = y + (col + cb) * by + (group32_num * 32 + j) * 32;

                    __m256i yq8 = _mm256_loadu_si256((const __m256i*)(py));

                    const __m256i ax = _mm256_sign_epi8(xq8, xq8);
                    const __m256i sy = _mm256_sign_epi8(yq8, xq8);
                    __m256i dot = _mm256_maddubs_epi16(ax, sy);

                    accula[cb] = _mm256_add_epi16(accula[cb], dot);
                }
            }

            for(int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                accu[cb] = _mm256_add_epi32(accu[cb], _mm256_madd_epi16(accula[cb], one16));
            }
        }

        for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
            int sumi = hsum_i32_8(accu[cb]);
            s[(col + cb) * bs] = sumi;
        }
    }
#elif defined(__ARM_NEON)
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;

    const int nb = n / QK_I8_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = la_num != 0 ? 1 : 0;

    for (int col = 0; col < nrc; col += VAE_PARALLEL_SIZE) {
        int32x4_t accu[VAE_PARALLEL_SIZE];
        for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
            accu[cb] = vdupq_n_s32(0);
        }

        for (int i = 0; i < group32_num; i++) {
#if defined(__ARM_FEATURE_DOTPROD)
            for (int j = 0; j < 32; j++) {
                const int8_t * px = x + (i * 32 + j) * QK_I8_S;
                int8x8_t xv = vld1_s8(px);
                for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                    const int8_t * py = y + (col + cb) * by + (i * 32 + j) * QK_I8_S;
                    int8x8_t yv = vld1_s8(py);
                    int32x2_t d = vdot_s32(vdup_n_s32(0), xv, yv);
                    accu[cb] = vcombine_s32(vadd_s32(vget_low_s32(accu[cb]), d), vget_high_s32(accu[cb]));
                }
            }
#else
            int16x8_t accu16[VAE_PARALLEL_SIZE];
            for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                accu16[cb] = vdupq_n_s16(0);
            }
            for (int j = 0; j < 32; j++) {
                const int8_t * px = x + (i * 32 + j) * QK_I8_S;
                int8x8_t xv = vld1_s8(px);
                for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                    const int8_t * py = y + (col + cb) * by + (i * 32 + j) * QK_I8_S;
                    int8x8_t yv = vld1_s8(py);
                    accu16[cb] = vmlal_s8(accu16[cb], xv, yv);
                }
            }
            for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                accu[cb] = vaddq_s32(accu[cb], vmovl_s16(vget_low_s16(accu16[cb])));
                accu[cb] = vaddq_s32(accu[cb], vmovl_high_s16(accu16[cb]));
            }
#endif
        }

        for (int i = 0; i < groupla_num; i++) {
#if defined(__ARM_FEATURE_DOTPROD)
            for (int j = 0; j < la_num; j++) {
                const int8_t * px = x + (group32_num * 32 + j) * QK_I8_S;
                int8x8_t xv = vld1_s8(px);
                for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                    const int8_t * py = y + (col + cb) * by + (group32_num * 32 + j) * QK_I8_S;
                    int8x8_t yv = vld1_s8(py);
                    int32x2_t d = vdot_s32(vdup_n_s32(0), xv, yv);
                    accu[cb] = vcombine_s32(vadd_s32(vget_low_s32(accu[cb]), d), vget_high_s32(accu[cb]));
                }
            }
#else
            int16x8_t accu16la[VAE_PARALLEL_SIZE];
            for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                accu16la[cb] = vdupq_n_s16(0);
            }
            for (int j = 0; j < la_num; j++) {
                const int8_t * px = x + (group32_num * 32 + j) * QK_I8_S;
                int8x8_t xv = vld1_s8(px);
                for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                    const int8_t * py = y + (col + cb) * by + (group32_num * 32 + j) * QK_I8_S;
                    int8x8_t yv = vld1_s8(py);
                    accu16la[cb] = vmlal_s8(accu16la[cb], xv, yv);
                }
            }
            for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                accu[cb] = vaddq_s32(accu[cb], vmovl_s16(vget_low_s16(accu16la[cb])));
                accu[cb] = vaddq_s32(accu[cb], vmovl_high_s16(accu16la[cb]));
            }
#endif
        }

        for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
            s[(col + cb) * bs] = vaddlvq_s32(accu[cb]);
        }
    }
#else
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;
    for (int col = 0; col < nrc; col++) {
        const int8_t * y_col = y + col * by;
        int32_t sumi = 0;
        for (int k = 0; k < n; k++) {
            sumi += (int32_t)x[k] * (int32_t)y_col[k];
        }
        s[col * bs] = sumi;
    }
#endif
}

// ---------------------------------------------------------------------------
// Register-tiled INT8 GEMM. See the header for why this exists and how it avoids
// the per-pair sign fold.
// ---------------------------------------------------------------------------

#define I8_TILE_M 4   // activation rows per tile
#define I8_TILE_N 4   // weight columns per tile
                      // 4x4 int32 accumulators = 16 zmm, leaving room for 4+4 operands

#if defined(VIBEASR_HAS_AVX512_PATH)

VIBEASR_TGT_VNNI static void gemm_i8_tile_vnni(
        int nbytes, int32_t * s, size_t bs,
        const int8_t * vx, const int8_t * vy, int nr, int nc, int n) {
    const __m512i ones  = _mm512_set1_epi8(1);
    const __m512i flip  = _mm512_set1_epi8((char) 0x80);  // +128 on an int8, i.e. to unsigned

    const int nr_t = nr - nr % I8_TILE_M;
    const int nc_t = nc - nc % I8_TILE_N;

    for (int r0 = 0; r0 < nr_t; r0 += I8_TILE_M) {
        // Row sums, once per row block rather than once per tile: the -128*sum(y)
        // correction is the same for every column this row meets.
        int32_t rowsum[I8_TILE_M];
        for (int r = 0; r < I8_TILE_M; r++) {
            __m512i acc = _mm512_setzero_si512();
            const int8_t * py = vy + (size_t)(r0 + r) * n;
            for (int i = 0; i < nbytes; i += 64) {
                acc = _mm512_dpbusd_epi32(acc, ones, _mm512_loadu_si512((const void *)(py + i)));
            }
            rowsum[r] = _mm512_reduce_add_epi32(acc);
        }

        for (int c0 = 0; c0 < nc_t; c0 += I8_TILE_N) {
            __m512i acc[I8_TILE_M][I8_TILE_N];
            for (int r = 0; r < I8_TILE_M; r++)
                for (int c = 0; c < I8_TILE_N; c++)
                    acc[r][c] = _mm512_setzero_si512();

            for (int i = 0; i < nbytes; i += 64) {
                __m512i xu[I8_TILE_N];
                for (int c = 0; c < I8_TILE_N; c++) {
                    // XOR 0x80 turns int8 x into the unsigned byte x+128
                    xu[c] = _mm512_xor_si512(
                        _mm512_loadu_si512((const void *)(vx + (size_t)(c0 + c) * n + i)), flip);
                }
                for (int r = 0; r < I8_TILE_M; r++) {
                    const __m512i yv =
                        _mm512_loadu_si512((const void *)(vy + (size_t)(r0 + r) * n + i));
                    for (int c = 0; c < I8_TILE_N; c++) {
                        acc[r][c] = _mm512_dpbusd_epi32(acc[r][c], xu[c], yv);
                    }
                }
            }

            for (int r = 0; r < I8_TILE_M; r++) {
                for (int c = 0; c < I8_TILE_N; c++) {
                    // sum((x+128)*y) - 128*sum(y) == sum(x*y)
                    s[(size_t)(r0 + r) * bs + c0 + c] =
                        _mm512_reduce_add_epi32(acc[r][c]) - 128 * rowsum[r];
                }
            }
        }

        // Ragged columns
        for (int c = nc_t; c < nc; c++) {
            for (int r = 0; r < I8_TILE_M; r++) {
                ggml_vec_dot_i8_i8(n, s + (size_t)(r0 + r) * bs + c, 0,
                                   vx + (size_t) c * n, 0, vy + (size_t)(r0 + r) * n, 0, 1);
            }
        }
    }

    // Ragged rows
    for (int r = nr_t; r < nr; r++) {
        for (int c = 0; c < nc; c++) {
            ggml_vec_dot_i8_i8(n, s + (size_t) r * bs + c, 0,
                               vx + (size_t) c * n, 0, vy + (size_t) r * n, 0, 1);
        }
    }
}

#endif  // VIBEASR_HAS_AVX512_PATH

void ggml_gemm_i8_i8_tiled(int n, int32_t * s, size_t bs, const void * vx, const void * vy,
                           int nr, int nc) {
    VIBEASR_PROBE(VIBEASR_K_I8_GEMM, (uint64_t) n * (uint64_t) nr * (uint64_t) nc);

#if defined(VIBEASR_HAS_AVX512_PATH)
    // VIBEASR_GEMM_TILE=0 takes the fallback while leaving the vec_dot kernels on
    // their VNNI path, so the tile can be A/B'd against exactly what it replaces.
    // Forcing VIBEASR_ISA down would also demote vec_dot and overstate the tile.
    static const int tile_on = []() {
        const char * e = getenv("VIBEASR_GEMM_TILE");
        return !(e && !strcmp(e, "0"));
    }();

    // The vec_dot kernels consume whole QK_I8_S blocks and ignore any tail; a tile
    // that disagreed would silently change results, so match them exactly. n < one
    // block has no whole blocks at all and goes to the fallback.
    const int nbytes = (n / QK_I8_S) * QK_I8_S;
    if (tile_on && vibeasr_isa() >= VIBEASR_ISA_VNNI && nbytes >= 64 && nbytes % 64 == 0) {
        gemm_i8_tile_vnni(nbytes, s, bs, (const int8_t *) vx, (const int8_t *) vy, nr, nc, n);
        return;
    }
#endif

    // Fallback: the original blocking, one weight column against a block of rows.
    const int64_t row_block = VAE_ROW_BLOCK_SIZE;
    const int64_t col_block = VAE_COL_BLOCK_SIZE;
    for (int64_t c0 = 0; c0 < nc; c0 += col_block) {
        const int64_t cur_c = (c0 + col_block <= nc) ? col_block : (nc - c0);
        for (int64_t r0 = 0; r0 < nr; r0 += row_block) {
            const int64_t cur_r = (r0 + row_block <= nr) ? row_block : (nr - r0);
            const int8_t * vy_r = (const int8_t *) vy + r0 * n;
            for (int64_t c = 0; c < cur_c; ++c) {
                const int64_t col = c0 + c;
                int32_t * s_col = s + col;
                const int8_t * vx_col = (const int8_t *) vx + col * n;
                if (cur_r % VAE_PARALLEL_SIZE == 0) {
                    ggml_vec_dot_i8_i8(n, s_col + r0 * bs, bs, vx_col, n, vy_r, n, (int) cur_r);
                } else {
                    for (int64_t r = 0; r < cur_r; ++r) {
                        ggml_vec_dot_i8_i8(n, s_col + (r0 + r) * bs, 0,
                                           vx_col, 0, vy_r + r * n, 0, 1);
                    }
                }
            }
        }
    }
}

void ggml_vec_dot_i8_i8(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    VIBEASR_PROBE(nrc % VAE_PARALLEL_SIZE == 0
                      ? (VAE_ACT_PARALLEL_SELECTED ? VIBEASR_K_I8_Nx1 : VIBEASR_K_I8_1xN)
                      : VIBEASR_K_I8_1x1,
                  (uint64_t) n * (uint64_t) nrc);
    if (nrc % VAE_PARALLEL_SIZE == 0) {
#if defined(VAE_ACT_PARALLEL)
        ggml_vec_dot_i8_i8_Nx1(n, s, bs, vx, bx, vy, by, nrc);
#else
        ggml_vec_dot_i8_i8_1xN(n, s, bs, vx, bx, vy, by, nrc);
#endif
    } else {
        ggml_vec_dot_i8_i8_1x1(n, s, bs, vx, bx, vy, by, nrc);
    }
}

void ggml_vec_dot_i8_i8_n4_col8(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc) {
    VIBEASR_PROBE(VIBEASR_K_I8_SMALL, (uint64_t) 4 * 8 * (uint64_t) nrc);
    
#if defined(__AVX2__) || defined(__AVX__)
    const __m256i one16 = _mm256_set1_epi16(1);

    for (int row = 0; row < nrc; row++) {

        const int8_t * vy_row = vy + row * 4;
        const int8_t * vx_row = vx;
        
        __m256i qx = _mm256_loadu_si256((const __m256i *)vx_row);
        
        uint32_t vy_32;
        memcpy(&vy_32, vy_row, sizeof(uint32_t));
        __m256i qy = _mm256_set1_epi32(vy_32);
        
        __m256i acc_i32;
        
#if __AVXVNNIINT8__
        acc_i32 = _mm256_setzero_si256();
        acc_i32 = _mm256_dpbssd_epi32(acc_i32, qx, qy);
#else
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        const __m256i sy = _mm256_sign_epi8(qy, qx);
        __m256i dot = _mm256_maddubs_epi16(ax, sy);
        acc_i32 = _mm256_madd_epi16(dot, one16);
#endif
        
        int32_t sums[8];
        _mm256_storeu_si256((__m256i *)sums, acc_i32);
        
        for (int i = 0; i < 8; i++) {
            s[row * bs + i] = sums[i];
        }
    }
#else
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 4;
        for (int col = 0; col < 8; col++) {
            const int8_t * vx_col = vx + col * 4;
            int32_t sumi = 0;
            for (int k = 0; k < 4; k++) {
                sumi += (int32_t)vx_col[k] * (int32_t)vy_row[k];
            }
            s[row * bs + col] = sumi;
        }
    }
#endif
}

void ggml_vec_dot_i8_i8_n8_col4(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc) {
    VIBEASR_PROBE(VIBEASR_K_I8_SMALL, (uint64_t) 8 * 4 * (uint64_t) nrc);
    
#if defined(__AVX2__) || defined(__AVX__)
    const __m256i one16 = _mm256_set1_epi16(1);

    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 8;
        const int8_t * vx_row = vx;
        
        __m256i qx = _mm256_loadu_si256((const __m256i *)vx_row);
        
        uint64_t vy_64;
        memcpy(&vy_64, vy_row, sizeof(uint64_t));
        __m256i qy = _mm256_set_epi64x(vy_64, vy_64, vy_64, vy_64);
        
        __m256i acc_i32;
        
#if __AVXVNNIINT8__
        acc_i32 = _mm256_setzero_si256();
        acc_i32 = _mm256_dpbssd_epi32(acc_i32, qx, qy);
#else
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        const __m256i sy = _mm256_sign_epi8(qy, qx);
        __m256i dot = _mm256_maddubs_epi16(ax, sy);
        acc_i32 = _mm256_madd_epi16(dot, one16);
#endif
        
        __m256i sum_h1 = _mm256_hadd_epi32(acc_i32, acc_i32);
        
        int32_t sums[8];
        _mm256_storeu_si256((__m256i *)sums, sum_h1);
        
        s[row * bs + 0] = sums[0];
        s[row * bs + 1] = sums[1];
        s[row * bs + 2] = sums[4];
        s[row * bs + 3] = sums[5];
    }
#else
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 8;
        for (int col = 0; col < 4; col++) {
            const int8_t * vx_col = vx + col * 8;
            int32_t sumi = 0;
            for (int k = 0; k < 8; k++) {
                sumi += (int32_t)vx_col[k] * (int32_t)vy_row[k];
            }
            s[row * bs + col] = sumi;
        }
    }
#endif
}

void ggml_vec_dot_i8_i8_n16_col2(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc) {
    VIBEASR_PROBE(VIBEASR_K_I8_SMALL, (uint64_t) 16 * 2 * (uint64_t) nrc);
    
#if defined(__AVX2__) || defined(__AVX__)
    const __m256i one16 = _mm256_set1_epi16(1);

    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 16;
        const int8_t * vx_row = vx;
        
        __m256i qx = _mm256_loadu_si256((const __m256i *)vx_row);
        
        __m128i vy_128 = _mm_loadu_si128((const __m128i *)vy_row);
        __m256i qy = _mm256_set_m128i(vy_128, vy_128);
        
        __m256i acc_i32;
        
#if __AVXVNNIINT8__
        acc_i32 = _mm256_setzero_si256();
        acc_i32 = _mm256_dpbssd_epi32(acc_i32, qx, qy);
#else
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        const __m256i sy = _mm256_sign_epi8(qy, qx);
        __m256i dot = _mm256_maddubs_epi16(ax, sy);
        acc_i32 = _mm256_madd_epi16(dot, one16);
#endif
        
        __m256i sum_h1 = _mm256_hadd_epi32(acc_i32, acc_i32);
        __m256i sum_h2 = _mm256_hadd_epi32(sum_h1, sum_h1);
        
        int32_t sums[8];
        _mm256_storeu_si256((__m256i *)sums, sum_h2);
        
        s[row * bs + 0] = sums[0];
        s[row * bs + 1] = sums[4];
    }
#else
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 16;
        for (int col = 0; col < 2; col++) {
            const int8_t * vx_col = vx + col * 16;
            int32_t sumi = 0;
            for (int k = 0; k < 16; k++) {
                sumi += (int32_t)vx_col[k] * (int32_t)vy_row[k];
            }
            s[row * bs + col] = sumi;
        }
    }
#endif
}

void ggml_vec_dot_i8_i8_n2_col16(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc) {
    VIBEASR_PROBE(VIBEASR_K_I8_SMALL, (uint64_t) 2 * 16 * (uint64_t) nrc);

#if defined(__AVX2__) || defined(__AVX__)
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 2;

        __m256i qx = _mm256_loadu_si256((const __m256i *)vx);

        uint16_t vy_16;
        memcpy(&vy_16, vy_row, sizeof(uint16_t));
        __m256i qy = _mm256_set1_epi16(vy_16);

        const __m256i ax = _mm256_sign_epi8(qx, qx);
        const __m256i sy = _mm256_sign_epi8(qy, qx);
        __m256i dot = _mm256_maddubs_epi16(ax, sy);

        __m128i dot_lo = _mm256_castsi256_si128(dot);
        __m128i dot_hi = _mm256_extracti128_si256(dot, 1);
        __m256i ext_lo = _mm256_cvtepi16_epi32(dot_lo);
        __m256i ext_hi = _mm256_cvtepi16_epi32(dot_hi);

        _mm256_storeu_si256((__m256i *)(s + row * bs), ext_lo);
        _mm256_storeu_si256((__m256i *)(s + row * bs + 8), ext_hi);
    }
#else
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 2;
        for (int col = 0; col < 16; col++) {
            const int8_t * vx_col = vx + col * 2;
            int32_t sumi = 0;
            for (int k = 0; k < 2; k++) {
                sumi += (int32_t)vx_col[k] * (int32_t)vy_row[k];
            }
            s[row * bs + col] = sumi;
        }
    }
#endif
}

void ggml_vec_dot_i8_i8_n4_col2(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc) {
    VIBEASR_PROBE(VIBEASR_K_I8_SMALL, (uint64_t) 4 * 2 * (uint64_t) nrc);

#if defined(__ARM_NEON)
    int8x8_t vx_vec = vld1_s8(vx);

    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 4;
        int32_t vy_32;
        memcpy(&vy_32, vy_row, sizeof(int32_t));
        int8x8_t vy_vec = vreinterpret_s8_s32(vdup_n_s32(vy_32));

#if defined(__ARM_FEATURE_DOTPROD)
        int32x2_t dot = vdot_s32(vdup_n_s32(0), vx_vec, vy_vec);
        s[row * bs + 0] = vget_lane_s32(dot, 0);
        s[row * bs + 1] = vget_lane_s32(dot, 1);
#else
        int16x8_t prod = vmull_s8(vx_vec, vy_vec);
        int16x4_t lo = vget_low_s16(prod);
        int16x4_t hi = vget_high_s16(prod);
        s[row * bs + 0] = vaddlv_s16(lo);
        s[row * bs + 1] = vaddlv_s16(hi);
#endif
    }
#else
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 4;
        for (int col = 0; col < 2; col++) {
            const int8_t * vx_col = vx + col * 4;
            int32_t sumi = 0;
            for (int k = 0; k < 4; k++) {
                sumi += (int32_t)vx_col[k] * (int32_t)vy_row[k];
            }
            s[row * bs + col] = sumi;
        }
    }
#endif
}

void ggml_vec_dot_i8_i8_n2_col4(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc) {
    VIBEASR_PROBE(VIBEASR_K_I8_SMALL, (uint64_t) 2 * 4 * (uint64_t) nrc);

#if defined(__ARM_NEON)
    int8x8_t vx_vec = vld1_s8(vx);

    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 2;
        int16_t vy_16;
        memcpy(&vy_16, vy_row, sizeof(int16_t));
        int8x8_t vy_vec = vreinterpret_s8_s16(vdup_n_s16(vy_16));

        int16x8_t prod = vmull_s8(vx_vec, vy_vec);
        int32x4_t psum = vpaddlq_s16(prod);
        s[row * bs + 0] = vgetq_lane_s32(psum, 0);
        s[row * bs + 1] = vgetq_lane_s32(psum, 1);
        s[row * bs + 2] = vgetq_lane_s32(psum, 2);
        s[row * bs + 3] = vgetq_lane_s32(psum, 3);
    }
#else
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 2;
        for (int col = 0; col < 4; col++) {
            const int8_t * vx_col = vx + col * 2;
            int32_t sumi = 0;
            for (int k = 0; k < 2; k++) {
                sumi += (int32_t)vx_col[k] * (int32_t)vy_row[k];
            }
            s[row * bs + col] = sumi;
        }
    }
#endif
}

void ggml_vec_dot_i8_i8_batch_n8(
    int32_t * dst_data,
    const int8_t * weight_data,
    const int8_t * input_data,
    int64_t ne00,
    int64_t ne01,
    int64_t ne02,
    int64_t ne10,
    int64_t ne11) {

    VIBEASR_PROBE(VIBEASR_K_I8_BATCH_N8, (uint64_t) ne00 * (uint64_t) ne02 * (uint64_t) ne11);

#if defined(__AVX2__) || defined(__AVX__)
    const __m256i one16 = _mm256_set1_epi16(1);
    
    for (int64_t batch = 0; batch < ne02; batch += 1) {
        const int8_t * weight = weight_data + batch * ne00;
        const int8_t * input = input_data + batch * ne10 * ne11;
        int32_t * output = dst_data + batch * ne11;

        int64_t weight_i64;
        memcpy(&weight_i64, weight, sizeof(int64_t));
        __m256i w_vec = _mm256_set_epi64x(weight_i64, weight_i64, weight_i64, weight_i64);
        
        int64_t col;
        for (col = 0; col + 3 < ne11; col += 4) {

            __m256i i_vec = _mm256_set_epi64x(
                *(int64_t *)(input + (col + 3) * ne10),
                *(int64_t *)(input + (col + 2) * ne10),
                *(int64_t *)(input + (col + 1) * ne10),
                *(int64_t *)(input + (col + 0) * ne10)
            );

            __m256i acc_i32;
#if __AVXVNNIINT8__
            acc_i32 = _mm256_setzero_si256();
            acc_i32 = _mm256_dpbssd_epi32(acc_i32, i_vec, w_vec);
#else
            const __m256i ax = _mm256_sign_epi8(i_vec, i_vec);
            const __m256i sy = _mm256_sign_epi8(w_vec, i_vec);
            __m256i dot = _mm256_maddubs_epi16(ax, sy);
            acc_i32 = _mm256_madd_epi16(dot, one16);
#endif
            __m256i sum_h1 = _mm256_hadd_epi32(acc_i32, acc_i32);
            
            int32_t sums[8];
            _mm256_storeu_si256((__m256i *)sums, sum_h1);
            
            output[col + 0] = sums[0];
            output[col + 1] = sums[1];
            output[col + 2] = sums[4];
            output[col + 3] = sums[5];
        }
        
        for (; col < ne11; col++) {
            int32_t sum = 0;
            for (int k = 0; k < ne00; k++) {
                sum += (int32_t)weight[k] * (int32_t)input[col * ne10 + k];
            }
            output[col] = sum;
        }
    }
#elif defined(__ARM_NEON)
    for (int64_t batch = 0; batch < ne02; batch += 1) {
        const int8_t * weight = weight_data + batch * ne00;
        const int8_t * input  = input_data + batch * ne10 * ne11;
        int32_t * output      = dst_data + batch * ne11;

        int8x8_t wv = vld1_s8(weight);

        int64_t col;
        for (col = 0; col + 3 < ne11; col += 4) {
            int8x8_t iv0 = vld1_s8(input + (col + 0) * ne10);
            int8x8_t iv1 = vld1_s8(input + (col + 1) * ne10);
            int8x8_t iv2 = vld1_s8(input + (col + 2) * ne10);
            int8x8_t iv3 = vld1_s8(input + (col + 3) * ne10);
#if defined(__ARM_FEATURE_DOTPROD)
            int32x2_t d0 = vdot_s32(vdup_n_s32(0), iv0, wv);
            int32x2_t d1 = vdot_s32(vdup_n_s32(0), iv1, wv);
            int32x2_t d2 = vdot_s32(vdup_n_s32(0), iv2, wv);
            int32x2_t d3 = vdot_s32(vdup_n_s32(0), iv3, wv);
            output[col + 0] = vget_lane_s32(d0, 0) + vget_lane_s32(d0, 1);
            output[col + 1] = vget_lane_s32(d1, 0) + vget_lane_s32(d1, 1);
            output[col + 2] = vget_lane_s32(d2, 0) + vget_lane_s32(d2, 1);
            output[col + 3] = vget_lane_s32(d3, 0) + vget_lane_s32(d3, 1);
#else
            int16x8_t p0 = vmull_s8(iv0, wv);
            int16x8_t p1 = vmull_s8(iv1, wv);
            int16x8_t p2 = vmull_s8(iv2, wv);
            int16x8_t p3 = vmull_s8(iv3, wv);
            output[col + 0] = vaddlvq_s16(p0);
            output[col + 1] = vaddlvq_s16(p1);
            output[col + 2] = vaddlvq_s16(p2);
            output[col + 3] = vaddlvq_s16(p3);
#endif
        }

        for (; col < ne11; col++) {
            int32_t sum = 0;
            for (int k = 0; k < ne00; k++) {
                sum += (int32_t)weight[k] * (int32_t)input[col * ne10 + k];
            }
            output[col] = sum;
        }
    }
#else
    for (int64_t batch = 0; batch < ne02; batch += 1) {
        const int8_t * weight = weight_data + batch * ne00;
        const int8_t * input  = input_data + batch * ne10 * ne11;
        int32_t * output      = dst_data + batch * ne11;

        for (int64_t col = 0; col < ne11; col++) {
            int32_t sum = 0;
            for (int k = 0; k < ne00; k++) {
                sum += (int32_t)weight[k] * (int32_t)input[col * ne10 + k];
            }
            output[col] = sum;
        }
    }
#endif
}
