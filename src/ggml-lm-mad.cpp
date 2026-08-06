#include <vector>
#include <type_traits>
#include <assert.h>
#include "ggml-quants.h"
#include "lm-config.h"
#include "ggml-cpu-impl.h"
#include "vibeasr-cpu.h"
#include "vibeasr-kernel-stats.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Runtime ISA detection and kernel accounting.
//
// These live here rather than in their own translation unit because the ggml fork
// used as a submodule hardcodes the list of VibeASR sources it compiles; adding a
// file would mean patching the submodule.
// ---------------------------------------------------------------------------

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#define VIBEASR_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#endif

extern "C" {

#if defined(VIBEASR_X86)

static void vibeasr_cpuid(unsigned leaf, unsigned sub, unsigned regs[4]) {
#if defined(_MSC_VER)
    int r[4];
    __cpuidex(r, (int) leaf, (int) sub);
    regs[0] = r[0]; regs[1] = r[1]; regs[2] = r[2]; regs[3] = r[3];
#else
    unsigned a, b, c, d;
    __cpuid_count(leaf, sub, a, b, c, d);
    regs[0] = a; regs[1] = b; regs[2] = c; regs[3] = d;
#endif
}

static unsigned long long vibeasr_xgetbv0(void) {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long) hi << 32) | lo;
#endif
}

// AMX tile state is not in the default XCR0 mask; Linux hands it out only on request.
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

static int vibeasr_request_amx_tiles(void) {
#if defined(__linux__) && defined(SYS_arch_prctl)
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
#else
    return 0;
#endif
}

static int vibeasr_detect(void) {
    unsigned r[4];
    vibeasr_cpuid(0, 0, r);
    const unsigned max_leaf = r[0];
    if (max_leaf < 1) return VIBEASR_ISA_SCALAR;

    vibeasr_cpuid(1, 0, r);
    const int has_osxsave = (r[2] >> 27) & 1;
    const int has_avx     = (r[2] >> 28) & 1;
    const int has_fma     = (r[2] >> 12) & 1;
    if (!has_osxsave || !has_avx) return VIBEASR_ISA_SCALAR;

    // XCR0: bit 1 SSE, 2 YMM, 5 opmask, 6 ZMM_hi256, 7 hi16_ZMM, 17 tilecfg, 18 tiledata
    const unsigned long long xcr0 = vibeasr_xgetbv0();
    const int ymm_ok = (xcr0 & 0x6) == 0x6;
    const int zmm_ok = (xcr0 & 0xe6) == 0xe6;
    if (!ymm_ok) return VIBEASR_ISA_SCALAR;

    if (max_leaf < 7) return VIBEASR_ISA_AVX2;
    vibeasr_cpuid(7, 0, r);
    const unsigned ebx7 = r[1], ecx7 = r[2], edx7 = r[3];

    const int has_avx2     = (ebx7 >> 5) & 1;
    const int has_avx512f  = (ebx7 >> 16) & 1;
    const int has_avx512dq = (ebx7 >> 17) & 1;
    const int has_avx512bw = (ebx7 >> 30) & 1;
    const int has_avx512vl = (ebx7 >> 31) & 1;
    const int has_vnni     = (ecx7 >> 11) & 1;
    const int has_amx_tile = (edx7 >> 24) & 1;
    const int has_amx_int8 = (edx7 >> 25) & 1;

    if (!has_avx2 || !has_fma) return VIBEASR_ISA_SCALAR;
    if (!zmm_ok || !(has_avx512f && has_avx512bw && has_avx512vl && has_avx512dq)) {
        return VIBEASR_ISA_AVX2;
    }
    if (!has_vnni) return VIBEASR_ISA_AVX512;
    if (has_amx_tile && has_amx_int8 && vibeasr_request_amx_tiles()) {
        // Re-read XCR0: the kernel only sets the tile bits once the request succeeds.
        const unsigned long long xcr0_amx = vibeasr_xgetbv0();
        if ((xcr0_amx & 0x60000) == 0x60000) return VIBEASR_ISA_AMX;
    }
    return VIBEASR_ISA_VNNI;
}

#else  // !VIBEASR_X86

static int vibeasr_detect(void) { return VIBEASR_ISA_SCALAR; }

#endif

const char * vibeasr_isa_name(int isa) {
    switch (isa) {
        case VIBEASR_ISA_AMX:    return "amx-int8";
        case VIBEASR_ISA_VNNI:   return "avx512-vnni";
        case VIBEASR_ISA_AVX512: return "avx512";
        case VIBEASR_ISA_AVX2:   return "avx2";
        default:                 return "scalar";
    }
}

static int vibeasr_isa_cached = -1;
static int vibeasr_isa_detected_cached = -1;

static int vibeasr_isa_from_name(const char * s) {
    if (!s) return -1;
    if (!strcmp(s, "amx"))    return VIBEASR_ISA_AMX;
    if (!strcmp(s, "vnni"))   return VIBEASR_ISA_VNNI;
    if (!strcmp(s, "avx512")) return VIBEASR_ISA_AVX512;
    if (!strcmp(s, "avx2"))   return VIBEASR_ISA_AVX2;
    if (!strcmp(s, "scalar")) return VIBEASR_ISA_SCALAR;
    return -1;
}

int vibeasr_isa_detected(void) {
    if (vibeasr_isa_detected_cached < 0) {
        vibeasr_isa_detected_cached = vibeasr_detect();
    }
    return vibeasr_isa_detected_cached;
}

int vibeasr_isa(void) {
    int isa = vibeasr_isa_cached;
    if (isa < 0) {
        isa = vibeasr_isa_detected();
        const int forced = vibeasr_isa_from_name(getenv("VIBEASR_ISA"));
        if (forced >= 0) {
            // Never let the override select something the CPU cannot execute.
            isa = forced < isa ? forced : isa;
        }
        if (getenv("VIBEASR_ISA_VERBOSE")) {
            fprintf(stderr, "[vibeasr] cpu supports %s, using %s\n",
                    vibeasr_isa_name(vibeasr_isa_detected()), vibeasr_isa_name(isa));
        }
        vibeasr_isa_cached = isa;
    }
    return isa;
}

// --- kernel accounting -----------------------------------------------------

// Off until the constructor below runs, so a probe that somehow fires during static
// init reads a defined value rather than tripping on a sentinel.
int vibeasr_kernel_stats_on = 0;
static int vibeasr_kernel_stats_ready = 0;

#define VIBEASR_STAT_SLOTS 64

struct vibeasr_stat_slot {
    uint64_t ns[VIBEASR_K_COUNT];
    uint64_t calls[VIBEASR_K_COUNT];
    uint64_t macs[VIBEASR_K_COUNT];
    char pad[64];
};

static struct vibeasr_stat_slot vibeasr_stats[VIBEASR_STAT_SLOTS];
static int vibeasr_stat_next = 0;
static __thread int vibeasr_stat_slot_id = -1;

static const char * vibeasr_kernel_name(int id) {
    switch (id) {
        case VIBEASR_K_I2_1x1:      return "i2_s  1x1 (gemv)";
        case VIBEASR_K_I2_1xN:      return "i2_s  1xN";
        case VIBEASR_K_I2_Nx1:      return "i2_s  Nx1";
        case VIBEASR_K_I8_1x1:      return "i8_s  1x1 (gemv)";
        case VIBEASR_K_I8_1xN:      return "i8_s  1xN";
        case VIBEASR_K_I8_Nx1:      return "i8_s  Nx1";
        case VIBEASR_K_I8_SMALL:    return "i8_s  small-n";
        case VIBEASR_K_I8_BATCH_N8: return "i8_s  dwconv n8";
        case VIBEASR_K_I8_GEMM:     return "i8_s  gemm (tiled)";
        default:                    return "?";
    }
}

static void vibeasr_kernel_stats_dump(void) {
    uint64_t ns[VIBEASR_K_COUNT] = {0}, calls[VIBEASR_K_COUNT] = {0}, macs[VIBEASR_K_COUNT] = {0};
    for (int s = 0; s < VIBEASR_STAT_SLOTS; s++) {
        for (int k = 0; k < VIBEASR_K_COUNT; k++) {
            ns[k]    += vibeasr_stats[s].ns[k];
            calls[k] += vibeasr_stats[s].calls[k];
            macs[k]  += vibeasr_stats[s].macs[k];
        }
    }
    uint64_t tot_ns = 0, tot_macs = 0;
    for (int k = 0; k < VIBEASR_K_COUNT; k++) { tot_ns += ns[k]; tot_macs += macs[k]; }
    if (tot_ns == 0) return;

    fprintf(stderr, "\n[vibeasr] kernel time (%s, summed over threads)\n", vibeasr_isa_name(vibeasr_isa()));
    fprintf(stderr, "  %-18s %12s %12s %10s %10s\n", "kernel", "cpu-ms", "calls", "GMAC", "GMAC/s");
    for (int k = 0; k < VIBEASR_K_COUNT; k++) {
        if (!calls[k]) continue;
        fprintf(stderr, "  %-18s %12.1f %12llu %10.2f %10.1f\n",
                vibeasr_kernel_name(k), ns[k] / 1e6, (unsigned long long) calls[k],
                macs[k] / 1e9, ns[k] ? macs[k] / (double) ns[k] : 0.0);
    }
    fprintf(stderr, "  %-18s %12.1f %12s %10.2f %10.1f\n",
            "total", tot_ns / 1e6, "", tot_macs / 1e9, tot_ns ? tot_macs / (double) tot_ns : 0.0);
}

__attribute__((constructor))
void vibeasr_kernel_stats_init(void) {
    if (vibeasr_kernel_stats_ready) return;
    vibeasr_kernel_stats_ready = 1;
    const char * e = getenv("VIBEASR_KERNEL_STATS");
    vibeasr_kernel_stats_on = (e && *e && strcmp(e, "0")) ? 1 : 0;
    if (vibeasr_kernel_stats_on) atexit(vibeasr_kernel_stats_dump);
}

void vibeasr_kernel_stats_add(int id, uint64_t ns, uint64_t macs) {
    int slot = vibeasr_stat_slot_id;
    if (slot < 0) {
        slot = __atomic_fetch_add(&vibeasr_stat_next, 1, __ATOMIC_RELAXED) % VIBEASR_STAT_SLOTS;
        vibeasr_stat_slot_id = slot;
    }
    vibeasr_stats[slot].ns[id]    += ns;
    vibeasr_stats[slot].calls[id] += 1;
    vibeasr_stats[slot].macs[id]  += macs;
}

}  // extern "C"

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#define QK_I2_S 128
#elif defined(__ARM_NEON)
#define QK_I2_S 64
#endif

#if defined(__ARM_NEON)
#define I2S_Y_BASE(u) ((((u) >> 1) * 128) + (((u) & 1) * 16))
#define I2S_Y_GROUP 32
#endif

#if defined(ACT_PARALLEL)
#define ACT_PARALLEL_SELECTED 1
#else
#define ACT_PARALLEL_SELECTED 0
#endif

// ---------------------------------------------------------------------------
// AVX-512 / AVX512-VNNI path for I2_S x I8 (the BitNet ternary LM weights)
//
// Weight layout: 32 bytes hold one QK_I2_S = 128-weight block as four 32-wide groups
// packed into the bit pairs of each byte -- group g of weight j occupies bits
// [7-2g : 6-2g]. Values are stored biased by one (0/1/2 for -1/0/+1), which is what
// lets the unsigned side of vpmaddubsw / vpdpbusd take them directly. The matching
// activations for one 32-byte weight step are the next 128 int8, group-major.
//
// The AVX2 kernel therefore reads 32 weight bytes and 128 activation bytes per step,
// and has to flush its int16 accumulator every 32 steps. Here we take two steps at a
// time in 512-bit lanes: one 64-byte weight load covers both, and the four activation
// loads are stitched into group order with vpermi2q. vpdpbusd accumulates straight to
// int32, so the flush and the group32/leftover split it forced both disappear.
// ---------------------------------------------------------------------------

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#define VIBEASR_HAS_AVX512_PATH 1
#define VIBEASR_TGT_VNNI __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")))

// Gathers the qwords of two 64-byte activation loads into the order the unpacked
// weight groups sit in: low half from `a`, high half from `b`.
static const int64_t vibeasr_perm_lo[8] = {0, 1, 2, 3, 8, 9, 10, 11};
static const int64_t vibeasr_perm_hi[8] = {4, 5, 6, 7, 12, 13, 14, 15};

// Columns handled per pass in the Nx1 (prefill) kernel. Wider than PARALLEL_SIZE on
// purpose: the weight unpack and permute are per-pass, so spreading them over more
// columns is what makes the AVX-512 path beat the AVX2 one here.
#define I2S_COL_BLOCK 4

struct i2s_groups { __m512i g[4]; };

VIBEASR_TGT_VNNI static inline struct i2s_groups i2s_unpack(const uint8_t * px) {
    const __m512i mask = _mm512_set1_epi8(0x03);
    const __m512i raw  = _mm512_loadu_si512((const void *) px);
    struct i2s_groups out;
    out.g[0] = _mm512_and_si512(_mm512_srli_epi16(raw, 6), mask);
    out.g[1] = _mm512_and_si512(_mm512_srli_epi16(raw, 4), mask);
    out.g[2] = _mm512_and_si512(_mm512_srli_epi16(raw, 2), mask);
    out.g[3] = _mm512_and_si512(raw, mask);
    return out;
}

// Activations for two consecutive 32-byte weight steps, reordered to match i2s_unpack.
VIBEASR_TGT_VNNI static inline void i2s_load_acts(const int8_t * py, __m512i y[4]) {
    const __m512i lo = _mm512_loadu_si512((const void *) vibeasr_perm_lo);
    const __m512i hi = _mm512_loadu_si512((const void *) vibeasr_perm_hi);
    const __m512i ya = _mm512_loadu_si512((const void *) (py +   0));
    const __m512i yb = _mm512_loadu_si512((const void *) (py +  64));
    const __m512i yc = _mm512_loadu_si512((const void *) (py + 128));
    const __m512i yd = _mm512_loadu_si512((const void *) (py + 192));
    y[0] = _mm512_permutex2var_epi64(ya, lo, yc);
    y[1] = _mm512_permutex2var_epi64(ya, hi, yc);
    y[2] = _mm512_permutex2var_epi64(yb, lo, yd);
    y[3] = _mm512_permutex2var_epi64(yb, hi, yd);
}

// The mirror image of i2s_load_acts: instead of shuffling activations into weight
// order, shuffle the unpacked weights into the order plain 64-byte activation loads
// already have. When the weights are the shared operand (the Nx1 / prefill shape)
// this pays for the permutes once and leaves each column with four loads and four
// vpdpbusd -- no shuffles on the per-column inner path at all.
VIBEASR_TGT_VNNI static inline void i2s_weights_in_act_order(const uint8_t * px, __m512i w[4]) {
    const struct i2s_groups g = i2s_unpack(px);
    const __m512i lo = _mm512_loadu_si512((const void *) vibeasr_perm_lo);
    const __m512i hi = _mm512_loadu_si512((const void *) vibeasr_perm_hi);
    w[0] = _mm512_permutex2var_epi64(g.g[0], lo, g.g[1]);  // pairs with py[  0: 64]
    w[1] = _mm512_permutex2var_epi64(g.g[2], lo, g.g[3]);  // pairs with py[ 64:128]
    w[2] = _mm512_permutex2var_epi64(g.g[0], hi, g.g[1]);  // pairs with py[128:192]
    w[3] = _mm512_permutex2var_epi64(g.g[2], hi, g.g[3]);  // pairs with py[192:256]
}

// Single 32-byte weight step, 256-bit lanes -- used for an odd trailing block.
VIBEASR_TGT_VNNI static inline __m256i i2s_step256(__m256i acc, const uint8_t * px, const int8_t * py) {
    const __m256i mask = _mm256_set1_epi8(0x03);
    const __m256i raw  = _mm256_loadu_si256((const __m256i *) px);
    acc = _mm256_dpbusd_epi32(acc, _mm256_and_si256(_mm256_srli_epi16(raw, 6), mask),
                              _mm256_loadu_si256((const __m256i *) (py + 0)));
    acc = _mm256_dpbusd_epi32(acc, _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask),
                              _mm256_loadu_si256((const __m256i *) (py + 32)));
    acc = _mm256_dpbusd_epi32(acc, _mm256_and_si256(_mm256_srli_epi16(raw, 2), mask),
                              _mm256_loadu_si256((const __m256i *) (py + 64)));
    acc = _mm256_dpbusd_epi32(acc, _mm256_and_si256(raw, mask),
                              _mm256_loadu_si256((const __m256i *) (py + 96)));
    return acc;
}

VIBEASR_TGT_VNNI static inline int32_t hsum256(__m256i a) {
    const __m128i s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    const __m128i h = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
    return _mm_cvtsi128_si32(_mm_add_epi32(h, _mm_shuffle_epi32(h, _MM_SHUFFLE(1, 1, 1, 1))));
}

// One row: nb weight blocks at px against nb*128 activations at py.
VIBEASR_TGT_VNNI static int32_t i2s_dot_vnni(const uint8_t * px, const int8_t * py, int nb) {
    __m512i acc[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(),
                      _mm512_setzero_si512(), _mm512_setzero_si512()};
    int b = 0;
    for (; b + 2 <= nb; b += 2) {
        const struct i2s_groups w = i2s_unpack(px + b * 32);
        __m512i y[4];
        i2s_load_acts(py + b * 128, y);
        acc[0] = _mm512_dpbusd_epi32(acc[0], w.g[0], y[0]);
        acc[1] = _mm512_dpbusd_epi32(acc[1], w.g[1], y[1]);
        acc[2] = _mm512_dpbusd_epi32(acc[2], w.g[2], y[2]);
        acc[3] = _mm512_dpbusd_epi32(acc[3], w.g[3], y[3]);
    }
    int32_t sum = _mm512_reduce_add_epi32(
        _mm512_add_epi32(_mm512_add_epi32(acc[0], acc[1]), _mm512_add_epi32(acc[2], acc[3])));
    if (b < nb) {
        sum += hsum256(i2s_step256(_mm256_setzero_si256(), px + b * 32, py + b * 128));
    }
    return sum;
}

// nblk rows sharing one activation vector: the weight loads stay per-row while the
// four activation registers are hoisted out of the block loop.
VIBEASR_TGT_VNNI static void i2s_dot_vnni_rows(
        const uint8_t * const * rows, const int8_t * py, int nb, int nblk, int32_t * out) {
    __m512i acc0[PARALLEL_SIZE], acc1[PARALLEL_SIZE];
    __m256i tail[PARALLEL_SIZE];
    for (int r = 0; r < nblk; r++) {
        acc0[r] = _mm512_setzero_si512();
        acc1[r] = _mm512_setzero_si512();
        tail[r] = _mm256_setzero_si256();
    }

    int b = 0;
    for (; b + 2 <= nb; b += 2) {
        __m512i y[4];
        i2s_load_acts(py + b * 128, y);
        for (int r = 0; r < nblk; r++) {
            const struct i2s_groups w = i2s_unpack(rows[r] + b * 32);
            acc0[r] = _mm512_dpbusd_epi32(acc0[r], w.g[0], y[0]);
            acc1[r] = _mm512_dpbusd_epi32(acc1[r], w.g[1], y[1]);
            acc0[r] = _mm512_dpbusd_epi32(acc0[r], w.g[2], y[2]);
            acc1[r] = _mm512_dpbusd_epi32(acc1[r], w.g[3], y[3]);
        }
    }
    if (b < nb) {
        for (int r = 0; r < nblk; r++) {
            tail[r] = i2s_step256(tail[r], rows[r] + b * 32, py + b * 128);
        }
    }
    for (int r = 0; r < nblk; r++) {
        out[r] = _mm512_reduce_add_epi32(_mm512_add_epi32(acc0[r], acc1[r])) + hsum256(tail[r]);
    }
}

// nblk activation columns sharing one weight row -- the prefill/batched shape. Here
// the expensive part (unpacking the 2-bit weights) is done once per block and reused.
VIBEASR_TGT_VNNI static void i2s_dot_vnni_cols(
        const uint8_t * px, const int8_t * const * cols, int nb, int nblk, int32_t * out) {
    // Two accumulators per column rather than four: with I2S_COL_BLOCK columns live
    // that is 16 zmm accumulators, leaving room for the weights and permute indices,
    // and two-long dependency chains still give the vpdpbusd port plenty to chew on
    // across independent columns.
    __m512i acc0[I2S_COL_BLOCK], acc1[I2S_COL_BLOCK];
    __m256i tail[I2S_COL_BLOCK];
    for (int c = 0; c < nblk; c++) {
        acc0[c] = _mm512_setzero_si512();
        acc1[c] = _mm512_setzero_si512();
        tail[c] = _mm256_setzero_si256();
    }

    int b = 0;
    for (; b + 2 <= nb; b += 2) {
        __m512i w[4];
        i2s_weights_in_act_order(px + b * 32, w);
        for (int c = 0; c < nblk; c++) {
            const int8_t * py = cols[c] + b * 128;
            acc0[c] = _mm512_dpbusd_epi32(acc0[c], w[0], _mm512_loadu_si512((const void *) (py +   0)));
            acc1[c] = _mm512_dpbusd_epi32(acc1[c], w[1], _mm512_loadu_si512((const void *) (py +  64)));
            acc0[c] = _mm512_dpbusd_epi32(acc0[c], w[2], _mm512_loadu_si512((const void *) (py + 128)));
            acc1[c] = _mm512_dpbusd_epi32(acc1[c], w[3], _mm512_loadu_si512((const void *) (py + 192)));
        }
    }
    if (b < nb) {
        for (int c = 0; c < nblk; c++) {
            tail[c] = i2s_step256(tail[c], px + b * 32, cols[c] + b * 128);
        }
    }
    for (int c = 0; c < nblk; c++) {
        out[c] = _mm512_reduce_add_epi32(_mm512_add_epi32(acc0[c], acc1[c])) + hsum256(tail[c]);
    }
}
#endif  // x86

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#include <immintrin.h>
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
#elif defined(__loongarch_asx)
static inline int hsum_i32_8(const __m256i a) {

    __m256i tmp1 = __lasx_xvpermi_q(a, a, 0x11);
    __m256i tmp2 = __lasx_xvpermi_q(a, a, 0x00);

    __m128i  tmp1_128 = lasx_extracti128_lo(tmp1);
    __m128i  tmp2_128 = lasx_extracti128_lo(tmp2);

    __m128i sum128 = __lsx_vadd_w(tmp1_128, tmp2_128);

    __m128i ev = __lsx_vpickev_w(sum128, sum128);
    __m128i od = __lsx_vpickod_w(sum128, sum128);
    __m128i sum64 = __lsx_vadd_w(ev, od);

    int sum64_1, sum64_2;
    sum64_1 = __lsx_vpickve2gr_w(sum64, 0);
    sum64_2 = __lsx_vpickve2gr_w(sum64, 1);

    return  sum64_1 + sum64_2;
}
#endif

size_t quantize_i2_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#if defined(ACT_PARALLEL)
    size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);

    int n = nrow * n_per_row;

    double max = 0;
    for (int i = 0; i < n; ++i) {
        max = fmax(max, (double)fabs((double)src[i]));
    }
    double i2_scale = max;

    uint8_t* q8 = (uint8_t*)malloc(n * sizeof(uint8_t));
    for (int i=0; i<n; i++) {
        if (fabs((double)(src[i])) < 1e-6) {
            q8[i] = 1;
            continue;
        }
        q8[i] = (double)src[i] * i2_scale > 0 ? 2 : 0;
    }

    memset(dst, 0, n * sizeof(uint8_t) / 4);


    uint8_t* i2_weight = (uint8_t*)dst;
    for (int i = 0; i < n / QK_I2_S; i++) {
        for (int j = 0; j < QK_I2_S; j++) {
            int group_idx = j / 32;
            int group_pos = j % 32;
            uint8_t temp = (q8[i * QK_I2_S + j] << (6 - 2 * group_idx));
            i2_weight[i * 32 + group_pos] |= temp;            
        }
    }

    float* scale_ptr = (float*)((char*)i2_weight + n / 4);
    scale_ptr[0] = i2_scale;

    free(q8);

    return n / 4 + 32;
#else
    assert((nrow % 4) == 0 && "quantize_i2_s_1x4 requires nrow % 4 == 0");

    size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);
    int64_t n = nrow * n_per_row;

    double max = 0;
    for (int64_t i = 0; i < n; ++i) {
        max = fmax(max, (double)fabs((double)src[i]));
    }
    double i2_scale = max;

    uint8_t* q8 = (uint8_t*)malloc(n * sizeof(uint8_t));
    for (int64_t i=0; i<n; i++) {
        if (fabs((double)(src[i])) < 1e-6) {
            q8[i] = 1;
            continue;
        }
        q8[i] = (double)src[i] * i2_scale > 0 ? 2 : 0;
    }

    uint8_t* out = (uint8_t*)dst;
    memset(out, 0, (size_t)(n / 4));

    int64_t nrow4 = nrow / 4;
    for (int64_t rg = 0; rg < nrow4; rg++) {
        int64_t r0 = rg * 4 + 0;
        int64_t r1 = rg * 4 + 1;
        int64_t r2 = rg * 4 + 2;
        int64_t r3 = rg * 4 + 3;

        int64_t base = rg * n_per_row;

        for (int64_t col = 0; col < n_per_row; col++) {
            uint8_t q0 = q8[r0 * n_per_row + col];
            uint8_t q1 = q8[r1 * n_per_row + col];
            uint8_t q2 = q8[r2 * n_per_row + col];
            uint8_t q3 = q8[r3 * n_per_row + col];

            uint8_t packed = (uint8_t)((q0 << 6) | (q1 << 4) | (q2 << 2) | (q3 << 0));
            out[base + col] = packed;
        }
    }

    float* scale_ptr = (float*)((char*)out + n / 4);
    scale_ptr[0] = (float)i2_scale;

    free(q8);

    return n / 4 + 32;
#endif
#elif defined(__ARM_NEON)
    size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);

    int n = nrow * n_per_row;

    double max = 0;
    for (int i = 0; i < n; ++i) {
        max = fmax(max, (double)fabs((double)src[i]));
    }
    double i2_scale = max;

    uint8_t* q8 = (uint8_t*)malloc(n * sizeof(uint8_t));
    for (int i=0; i<n; i++) {
        if (fabs((double)(src[i])) < 1e-6) {
            q8[i] = 1;
            continue;
        }
        q8[i] = (double)src[i] * i2_scale > 0 ? 2 : 0;
    }

    memset(dst, 0, n * sizeof(uint8_t) / 4);

    uint8_t* i2_weight = (uint8_t*)dst;
    for (int i = 0; i < n / 128; i++) {
        for (int j = 0; j < 128; j++) {
            int group_idx = j / 32;
            int group_pos = j % 32;
            uint8_t temp = (q8[i * 128 + j] << (6 - 2 * group_idx));
            i2_weight[i * 32 + group_pos] |= temp;
        }
    }

    float* scale_ptr = (float*)((char*)i2_weight + n / 4);
    scale_ptr[0] = i2_scale;

    free(q8);

    return n / 4 + 32;
#endif
}

void ggml_vec_dot_i2_i8_s_1x1(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(VIBEASR_HAS_AVX512_PATH)
    if (vibeasr_isa() >= VIBEASR_ISA_VNNI) {
        const uint8_t * x = (const uint8_t *) vx;
        const int8_t  * y = (const int8_t *) vy;
        const int nb = n / QK_I2_S;
        for (int row = 0; row < nrc; row++) {
            s[row] = (float) i2s_dot_vnni(x + row * bx / 4, y, nb);
        }
        return;
    }
#endif
#if defined(__AVX2__)
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;
    
    __m256i mask = _mm256_set1_epi8(0x03);
    __m256i one16 = _mm256_set1_epi16(1);

    for (int row = 0; row < nrc; row++) {
        __m256i accu = _mm256_setzero_si256();
        
        const uint8_t * x_row = x + row * bx / 4;
        
        for (int i = 0; i < group32_num; i++) {
            const uint8_t *px = x_row + i * 1024;
            const int8_t  *py = y + i * 4096;
            __m256i accu32 = _mm256_setzero_si256();
            
            for (int j = 0; j < 32; j++) {
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(px));
                __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
                __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
                __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

                xq8_3 = _mm256_and_si256(xq8_3, mask);
                xq8_2 = _mm256_and_si256(xq8_2, mask);
                xq8_1 = _mm256_and_si256(xq8_1, mask);
                xq8_0 = _mm256_and_si256(xq8_0, mask);

                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(py));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(py + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(py + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(py + 96));

                xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_0, xq8_1));
                accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_2, xq8_3));

                px += 32;
                py += 128;
            }
            accu = _mm256_add_epi32(_mm256_madd_epi16(accu32, one16), accu);
        }

        for (int i = 0; i < groupla_num; i++) {
            __m256i accula = _mm256_setzero_si256();
            const uint8_t *px = x_row + group32_num * 1024;
            const int8_t  *py = y + group32_num * 4096;
            
            for (int j = 0; j < la_num; j++) {
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(px));
                __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
                __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
                __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

                xq8_3 = _mm256_and_si256(xq8_3, mask);
                xq8_2 = _mm256_and_si256(xq8_2, mask);
                xq8_1 = _mm256_and_si256(xq8_1, mask);
                xq8_0 = _mm256_and_si256(xq8_0, mask);

                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(py));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(py + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(py + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(py + 96));

                xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_0, xq8_1));
                accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_2, xq8_3));

                px += 32;
                py += 128;
            }
            accu = _mm256_add_epi32(accu, _mm256_madd_epi16(accula, one16));
        }
        
        int sumi = hsum_i32_8(accu);
        s[row] = (float)sumi;
    }
#elif defined(__ARM_NEON)
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    const uint8x16_t mask = vdupq_n_u8(3);

    for (int row = 0; row < nrc; row++) {
        int32x4_t accu = vdupq_n_s32(0);

        const uint8_t * x_row = x + row * bx / 4;

        for (int i=0; i < group32_num; i++) {

#if defined(__ARM_FEATURE_DOTPROD)

#else
            int16x8_t accu32 = vdupq_n_s16(0);
#endif
            for (int j=0; j < 32; j++) {
                uint8x16_t xq8_3 = vld1q_u8(x_row + i * 32 * 16 + j * 16);
                uint8x16_t xq8_2 = vshrq_n_u8(xq8_3, 2);
                uint8x16_t xq8_1 = vshrq_n_u8(xq8_3, 4);
                uint8x16_t xq8_0 = vshrq_n_u8(xq8_3, 6);

                int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
                int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));

                const int8_t * py = y + I2S_Y_BASE(i * 32 + j);
                const int8x16_t yq8_0 = vld1q_s8(py + 0 * I2S_Y_GROUP);
                const int8x16_t yq8_1 = vld1q_s8(py + 1 * I2S_Y_GROUP);
                const int8x16_t yq8_2 = vld1q_s8(py + 2 * I2S_Y_GROUP);
                const int8x16_t yq8_3 = vld1q_s8(py + 3 * I2S_Y_GROUP);

#if defined(__ARM_FEATURE_DOTPROD)
                accu = vdotq_s32(accu, q8_0, yq8_0);
                accu = vdotq_s32(accu, q8_1, yq8_1);
                accu = vdotq_s32(accu, q8_2, yq8_2);
                accu = vdotq_s32(accu, q8_3, yq8_3);
#else
                accu32 = vmlal_s8(accu32, vget_low_s8(q8_0), vget_low_s8(yq8_0));
                accu32 = vmlal_s8(accu32, vget_high_s8(q8_0), vget_high_s8(yq8_0));
                accu32 = vmlal_s8(accu32, vget_low_s8(q8_1), vget_low_s8(yq8_1));
                accu32 = vmlal_s8(accu32, vget_high_s8(q8_1), vget_high_s8(yq8_1));
                accu32 = vmlal_s8(accu32, vget_low_s8(q8_2), vget_low_s8(yq8_2));
                accu32 = vmlal_s8(accu32, vget_high_s8(q8_2), vget_high_s8(yq8_2));
                accu32 = vmlal_s8(accu32, vget_low_s8(q8_3), vget_low_s8(yq8_3));
                accu32 = vmlal_s8(accu32, vget_high_s8(q8_3), vget_high_s8(yq8_3));
#endif
            }

#if defined(__ARM_FEATURE_DOTPROD)

#else
            accu = vaddq_s32(accu, vmovl_s16(vget_low_s16(accu32)));
            accu = vaddq_s32(accu, vmovl_high_s16(accu32));
#endif
        }

        for (int i = 0; i < groupla_num; i++){
#if defined(__ARM_FEATURE_DOTPROD)

#else
            int16x8_t accula = vdupq_n_s16(0);
#endif
            for (int j = 0; j < la_num; j++) {
                uint8x16_t xq8_3 = vld1q_u8(x_row + group32_num * 32 * 16 + j * 16);
                uint8x16_t xq8_2 = vshrq_n_u8(xq8_3, 2);
                uint8x16_t xq8_1 = vshrq_n_u8(xq8_3, 4);
                uint8x16_t xq8_0 = vshrq_n_u8(xq8_3, 6);

                int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
                int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));

                const int8_t * py = y + I2S_Y_BASE(group32_num * 32 + j);
                const int8x16_t yq8_0 = vld1q_s8(py + 0 * I2S_Y_GROUP);
                const int8x16_t yq8_1 = vld1q_s8(py + 1 * I2S_Y_GROUP);
                const int8x16_t yq8_2 = vld1q_s8(py + 2 * I2S_Y_GROUP);
                const int8x16_t yq8_3 = vld1q_s8(py + 3 * I2S_Y_GROUP);

#if defined(__ARM_FEATURE_DOTPROD)
                accu = vdotq_s32(accu, q8_0, yq8_0);
                accu = vdotq_s32(accu, q8_1, yq8_1);
                accu = vdotq_s32(accu, q8_2, yq8_2);
                accu = vdotq_s32(accu, q8_3, yq8_3);
#else
                accula = vmlal_s8(accula, vget_low_s8(q8_0), vget_low_s8(yq8_0));
                accula = vmlal_s8(accula, vget_high_s8(q8_0), vget_high_s8(yq8_0));
                accula = vmlal_s8(accula, vget_low_s8(q8_1), vget_low_s8(yq8_1));
                accula = vmlal_s8(accula, vget_high_s8(q8_1), vget_high_s8(yq8_1));
                accula = vmlal_s8(accula, vget_low_s8(q8_2), vget_low_s8(yq8_2));
                accula = vmlal_s8(accula, vget_high_s8(q8_2), vget_high_s8(yq8_2));
                accula = vmlal_s8(accula, vget_low_s8(q8_3), vget_low_s8(yq8_3));
                accula = vmlal_s8(accula, vget_high_s8(q8_3), vget_high_s8(yq8_3));
#endif
            }
#if defined(__ARM_FEATURE_DOTPROD)

#else
            accu = vaddq_s32(accu, vmovl_s16(vget_low_s16(accula)));
            accu = vaddq_s32(accu, vmovl_high_s16(accula));
#endif
        }
        int sumi = vaddlvq_s32(accu);
        s[row] = (float)sumi;
    }
#endif
}

void ggml_vec_dot_i2_i8_s_1xN(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(VIBEASR_HAS_AVX512_PATH)
    if (vibeasr_isa() >= VIBEASR_ISA_VNNI) {
        const uint8_t * x = (const uint8_t *) vx;
        const int8_t  * y = (const int8_t *) vy;
        const int nb = n / QK_I2_S;
        for (int row = 0; row < nrc; row += PARALLEL_SIZE) {
            const uint8_t * rows[PARALLEL_SIZE];
            int32_t out[PARALLEL_SIZE];
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) rows[rb] = x + (row + rb) * bx / 4;
            i2s_dot_vnni_rows(rows, y, nb, PARALLEL_SIZE, out);
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) s[row + rb] = (float) out[rb];
        }
        return;
    }
#endif
#if defined(__AVX2__)
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    const __m256i mask = _mm256_set1_epi8(0x03);
    const __m256i one16 = _mm256_set1_epi16(1);

    for (int row = 0; row < nrc; row+=PARALLEL_SIZE) {
        __m256i accu[PARALLEL_SIZE];
        const uint8_t * x_row[PARALLEL_SIZE];
        for(int rb = 0; rb < PARALLEL_SIZE; rb++) {
            accu[rb] = _mm256_setzero_si256();
            x_row[rb] = x + (row + rb) * bx / 4;
        }
        
        for (int i = 0; i < group32_num; i++) {
            const uint8_t * px[PARALLEL_SIZE];
            __m256i accu32[PARALLEL_SIZE];
            for(int rb = 0; rb < PARALLEL_SIZE; rb++) {
                px[rb] = x_row[rb] + i * 1024;
                accu32[rb] = _mm256_setzero_si256();
            }
            const int8_t  *py = y + i * 4096;
            
            for (int j = 0; j < 32; j++) {
                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(py));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(py + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(py + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(py + 96));
                for (int rb = 0; rb < PARALLEL_SIZE; rb++)
                {
                    __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(px[rb]));
                    __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
                    __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
                    __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

                    xq8_3 = _mm256_and_si256(xq8_3, mask);
                    xq8_2 = _mm256_and_si256(xq8_2, mask);
                    xq8_1 = _mm256_and_si256(xq8_1, mask);
                    xq8_0 = _mm256_and_si256(xq8_0, mask);

                    xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                    xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                    xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                    xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                    accu32[rb] = _mm256_add_epi16(accu32[rb], _mm256_add_epi16(xq8_0, xq8_1));
                    accu32[rb] = _mm256_add_epi16(accu32[rb], _mm256_add_epi16(xq8_2, xq8_3));

                    px[rb] += 32;
                }
                py += 128;
            }
            for(int rb = 0; rb < PARALLEL_SIZE; rb++) {
                accu[rb] = _mm256_add_epi32(_mm256_madd_epi16(accu32[rb], one16), accu[rb]);
            } 
        }

        for (int i = 0; i < groupla_num; i++) {
            const int8_t  *py = y + group32_num * 4096;
            const uint8_t * px[PARALLEL_SIZE];
            __m256i accula[PARALLEL_SIZE];
            for(int rb = 0; rb < PARALLEL_SIZE; rb++) {
                px[rb] = x_row[rb] + group32_num * 1024;
                accula[rb] = _mm256_setzero_si256();
            }
            
            for (int j = 0; j < la_num; j++) {
                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(py));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(py + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(py + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(py + 96));

                for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                    __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(px[rb]));
                    __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
                    __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
                    __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

                    xq8_3 = _mm256_and_si256(xq8_3, mask);
                    xq8_2 = _mm256_and_si256(xq8_2, mask);
                    xq8_1 = _mm256_and_si256(xq8_1, mask);
                    xq8_0 = _mm256_and_si256(xq8_0, mask);

                    

                    xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                    xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                    xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                    xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                    accula[rb] = _mm256_add_epi16(accula[rb], _mm256_add_epi16(xq8_0, xq8_1));
                    accula[rb] = _mm256_add_epi16(accula[rb], _mm256_add_epi16(xq8_2, xq8_3));

                    px[rb] += 32;
                }
                py += 128;
            }
            for(int rb = 0; rb < PARALLEL_SIZE; rb++) {
                accu[rb] = _mm256_add_epi32(accu[rb], _mm256_madd_epi16(accula[rb], one16));
            } 
        }
        
        for(int rb = 0; rb < PARALLEL_SIZE; rb++) {
            int sumi = hsum_i32_8(accu[rb]);
            s[row + rb] = (float)sumi;
        }
    }
#elif defined(__ARM_NEON)
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;
    
    const uint8x16_t mask = vdupq_n_u8(3);

    for (int row = 0; row < nrc; row += PARALLEL_SIZE) {

        int32x4_t accu[PARALLEL_SIZE];
        const uint8_t * x_row[PARALLEL_SIZE];
        
        for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
            accu[rb] = vdupq_n_s32(0);
            x_row[rb] = x + (row + rb) * bx / 4;
        }

        for (int i = 0; i < group32_num; i++) {
#if defined(__ARM_FEATURE_DOTPROD)

#else
            int16x8_t accu32[PARALLEL_SIZE];
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                accu32[rb] = vdupq_n_s16(0);
            }
#endif
            const uint8_t * px[PARALLEL_SIZE];
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                px[rb] = x_row[rb] + i * 32 * 16;
            }

            for (int j = 0; j < 32; j++) {
                const int8_t * py = y + I2S_Y_BASE(i * 32 + j);
                const int8x16_t yq8_0 = vld1q_s8(py + 0 * I2S_Y_GROUP);
                const int8x16_t yq8_1 = vld1q_s8(py + 1 * I2S_Y_GROUP);
                const int8x16_t yq8_2 = vld1q_s8(py + 2 * I2S_Y_GROUP);
                const int8x16_t yq8_3 = vld1q_s8(py + 3 * I2S_Y_GROUP);

                for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                    uint8x16_t xq8_3 = vld1q_u8(px[rb] + 0);
                    uint8x16_t xq8_2 = vshrq_n_u8(xq8_3, 2);
                    uint8x16_t xq8_1 = vshrq_n_u8(xq8_3, 4);
                    uint8x16_t xq8_0 = vshrq_n_u8(xq8_3, 6);

                    int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
                    int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                    int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                    int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));

#if defined(__ARM_FEATURE_DOTPROD)
                    accu[rb] = vdotq_s32(accu[rb], q8_0, yq8_0);
                    accu[rb] = vdotq_s32(accu[rb], q8_1, yq8_1);
                    accu[rb] = vdotq_s32(accu[rb], q8_2, yq8_2);
                    accu[rb] = vdotq_s32(accu[rb], q8_3, yq8_3);
#else
                    accu32[rb] = vmlal_s8(accu32[rb], vget_low_s8(q8_3), vget_low_s8(yq8_3));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_high_s8(q8_3), vget_high_s8(yq8_3));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_low_s8(q8_2), vget_low_s8(yq8_2));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_high_s8(q8_2), vget_high_s8(yq8_2));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_low_s8(q8_1), vget_low_s8(yq8_1));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_high_s8(q8_1), vget_high_s8(yq8_1));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_low_s8(q8_0), vget_low_s8(yq8_0));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_high_s8(q8_0), vget_high_s8(yq8_0));
                    
#endif
                    px[rb] += 16;
                }
            }

#if defined(__ARM_FEATURE_DOTPROD)

#else
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                accu[rb] = vaddq_s32(accu[rb], vmovl_s16(vget_low_s16(accu32[rb])));
                accu[rb] = vaddq_s32(accu[rb], vmovl_high_s16(accu32[rb]));
            }
#endif
        }

        for (int i = 0; i < groupla_num; i++) {
#if defined(__ARM_FEATURE_DOTPROD)

#else
            int16x8_t accula[PARALLEL_SIZE];
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                accula[rb] = vdupq_n_s16(0);
            }
#endif
            const uint8_t * px[PARALLEL_SIZE];
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                px[rb] = x_row[rb] + group32_num * 32 * 16;
            }

            for (int j = 0; j < la_num; j++) {
                const int8_t * py = y + I2S_Y_BASE(group32_num * 32 + j);
                const int8x16_t yq8_0 = vld1q_s8(py + 0 * I2S_Y_GROUP);
                const int8x16_t yq8_1 = vld1q_s8(py + 1 * I2S_Y_GROUP);
                const int8x16_t yq8_2 = vld1q_s8(py + 2 * I2S_Y_GROUP);
                const int8x16_t yq8_3 = vld1q_s8(py + 3 * I2S_Y_GROUP);

                for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                    uint8x16_t xq8_3 = vld1q_u8(px[rb] + 0);
                    uint8x16_t xq8_2 = vshrq_n_u8(xq8_3, 2);
                    uint8x16_t xq8_1 = vshrq_n_u8(xq8_3, 4);
                    uint8x16_t xq8_0 = vshrq_n_u8(xq8_3, 6);

                    int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
                    int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                    int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                    int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
                    
#if defined(__ARM_FEATURE_DOTPROD)
                    accu[rb] = vdotq_s32(accu[rb], q8_0, yq8_0);
                    accu[rb] = vdotq_s32(accu[rb], q8_1, yq8_1);
                    accu[rb] = vdotq_s32(accu[rb], q8_2, yq8_2);
                    accu[rb] = vdotq_s32(accu[rb], q8_3, yq8_3);
#else
                    accula[rb] = vmlal_s8(accula[rb], vget_low_s8(q8_3), vget_low_s8(yq8_3));
                    accula[rb] = vmlal_s8(accula[rb], vget_high_s8(q8_3), vget_high_s8(yq8_3));
                    accula[rb] = vmlal_s8(accula[rb], vget_low_s8(q8_2), vget_low_s8(yq8_2));
                    accula[rb] = vmlal_s8(accula[rb], vget_high_s8(q8_2), vget_high_s8(yq8_2));
                    accula[rb] = vmlal_s8(accula[rb], vget_low_s8(q8_1), vget_low_s8(yq8_1));
                    accula[rb] = vmlal_s8(accula[rb], vget_high_s8(q8_1), vget_high_s8(yq8_1));
                    accula[rb] = vmlal_s8(accula[rb], vget_low_s8(q8_0), vget_low_s8(yq8_0));
                    accula[rb] = vmlal_s8(accula[rb], vget_high_s8(q8_0), vget_high_s8(yq8_0));

#endif
                    px[rb] += 16;
                }
            }

#if defined(__ARM_FEATURE_DOTPROD)

#else
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                accu[rb] = vaddq_s32(accu[rb], vmovl_s16(vget_low_s16(accula[rb])));
                accu[rb] = vaddq_s32(accu[rb], vmovl_high_s16(accula[rb]));
            }
#endif
        }

        for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
            int sumi = vaddlvq_s32(accu[rb]);
            s[row + rb] = (float)sumi;
        }
    }
#endif
}

void ggml_vec_dot_i2_i8_s_Nx1(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(VIBEASR_HAS_AVX512_PATH)
    // Measured choice, not an oversight: on a 1-FMA-unit AVX-512 Xeon (Cascade Lake
    // and the Silver/Bronze SKUs) zmm vpdpbusd retires one per cycle while ymm
    // vpmaddubsw retires two, so both move 64 int8 products per cycle -- and this
    // kernel's weights are shared across columns, which is exactly the case where the
    // AVX2 code already amortises its unpack well. Measured on Xeon @2.8GHz,
    // n=1536/8960: AVX2 89-105 GMAC/s vs VNNI 52-75. The other I2_S shapes and all
    // I8_S shapes do win on VNNI, so only this one falls through.
    //
    // Unlike the I8_S kernels, the AVX2 I2_S path is numerically exact here: ternary
    // weights keep the int16 partial sums small enough not to wrap (kernel_bench
    // checks this in the full +-127 regime). Set VIBEASR_I2_NX1_VNNI=1 to take the
    // AVX-512 path anyway -- worth re-measuring on 2-FMA-unit and AMX parts.
    static const int nx1_vnni = getenv("VIBEASR_I2_NX1_VNNI") != NULL;
    if (nx1_vnni && vibeasr_isa() >= VIBEASR_ISA_VNNI) {
        const uint8_t * x = (const uint8_t *) vx;
        const int8_t  * y = (const int8_t *) vy;
        const int nb = n / QK_I2_S;
        for (int col = 0; col < nrc; col += I2S_COL_BLOCK) {
            const int nblk = (col + I2S_COL_BLOCK <= nrc) ? I2S_COL_BLOCK : (nrc - col);
            const int8_t * cols[I2S_COL_BLOCK];
            int32_t out[I2S_COL_BLOCK];
            for (int cb = 0; cb < nblk; cb++) cols[cb] = y + (col + cb) * by;
            i2s_dot_vnni_cols(x, cols, nb, nblk, out);
            for (int cb = 0; cb < nblk; cb++) s[(col + cb) * bs] = (float) out[cb];
        }
        return;
    }
#endif
#if defined(__AVX2__)
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    __m256i mask = _mm256_set1_epi8(0x03);
    __m256i one16 = _mm256_set1_epi16(1);

    for (int col = 0; col < nrc; col += PARALLEL_SIZE) {
        __m256i accu[PARALLEL_SIZE];

        for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
            accu[iy] = _mm256_setzero_si256();
        }

        const int8_t * y_col = y + col * by;
        
        for (int i = 0; i < group32_num; i++) {
            const uint8_t *px = x + i * 1024;
            const int8_t  *py = y_col + i * 4096;
            __m256i accu32[PARALLEL_SIZE];

            for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                accu32[iy] = _mm256_setzero_si256();
            }

            for (int j = 0; j < 32; j++) {

                __m256i xq8   = _mm256_loadu_si256((const __m256i*)(px));
                __m256i xq8_3 = _mm256_and_si256(xq8, mask);
                __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8, 2), mask);
                __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8, 4), mask);
                __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8, 6), mask);

                for (int iy = 0; iy < PARALLEL_SIZE; iy++)
                {
                    accu32[iy] = _mm256_add_epi16(accu32[iy], _mm256_add_epi16(
                                    _mm256_add_epi16(_mm256_maddubs_epi16(xq8_0, _mm256_loadu_si256((const __m256i*)(py + 0 * 32 + iy * by))),
                                                    _mm256_maddubs_epi16(xq8_1, _mm256_loadu_si256((const __m256i*)(py + 1 * 32 + iy * by)))),
                                    _mm256_add_epi16(_mm256_maddubs_epi16(xq8_2, _mm256_loadu_si256((const __m256i*)(py + 2 * 32 + iy * by))),
                                                    _mm256_maddubs_epi16(xq8_3, _mm256_loadu_si256((const __m256i*)(py + 3 * 32 + iy * by))))));
                }

                px += 32;
                py += 128;
            }

            for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                accu[iy] = _mm256_add_epi32(_mm256_madd_epi16(accu32[iy], one16), accu[iy]);
            }
        }

        for (int i = 0; i < groupla_num; i++) {
            const uint8_t *px = x + group32_num * 1024;
            const int8_t  *py = y_col + group32_num * 4096;
            __m256i accula[PARALLEL_SIZE];

            for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                accula[iy] = _mm256_setzero_si256();
            }
            
            for (int j = 0; j < la_num; j++) {
                
                __m256i xq8   = _mm256_loadu_si256((const __m256i*)(px));
                __m256i xq8_3 = _mm256_and_si256(xq8, mask);
                __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8, 2), mask);
                __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8, 4), mask);
                __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8, 6), mask);

                for (int iy = 0; iy < PARALLEL_SIZE; iy++)
                {
                    accula[iy] = _mm256_add_epi16(accula[iy], _mm256_add_epi16(
                                    _mm256_add_epi16(_mm256_maddubs_epi16(xq8_0, _mm256_loadu_si256((const __m256i*)(py + 0 * 32 + iy * by))),
                                                    _mm256_maddubs_epi16(xq8_1, _mm256_loadu_si256((const __m256i*)(py + 1 * 32 + iy * by)))),
                                    _mm256_add_epi16(_mm256_maddubs_epi16(xq8_2, _mm256_loadu_si256((const __m256i*)(py + 2 * 32 + iy * by))),
                                                    _mm256_maddubs_epi16(xq8_3, _mm256_loadu_si256((const __m256i*)(py + 3 * 32 + iy * by))))));
                }

                px += 32;
                py += 128;
            }

            for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                accu[iy] = _mm256_add_epi32(_mm256_madd_epi16(accula[iy], one16), accu[iy]);
            }
        }

        for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
            int sumi = hsum_i32_8(accu[iy]);
            s[(col + iy) * bs] = (float)sumi;
        }
    }
#elif defined(__ARM_NEON)
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    const uint8x16_t mask = vdupq_n_u8(3);

    for (int col = 0; col < nrc; col += PARALLEL_SIZE) {
        int32x4_t accu[PARALLEL_SIZE];

        for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
            accu[iy] = vdupq_n_s32(0);
        }

        const int8_t * y_col = y + col * by;
        
        for (int i = 0; i < group32_num; i++) {
            const uint8_t *px = x + i * 512;
            const int8_t  *py = y_col + I2S_Y_BASE(i * 32);

#if defined(__ARM_FEATURE_DOTPROD)

#else
            int16x8_t accu32[PARALLEL_SIZE];

            for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                accu32[iy] = vdupq_n_s16(0);
            }
#endif
            for (int j = 0; j < 32; j++) {
                uint8x16_t xq8_3 = vld1q_u8(px + 0);
                uint8x16_t xq8_2 = vshrq_n_u8(xq8_3, 2);
                uint8x16_t xq8_1 = vshrq_n_u8(xq8_3, 4);
                uint8x16_t xq8_0 = vshrq_n_u8(xq8_3, 6);

                int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
                int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));

                for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                    const int8x16_t yq8_0 = vld1q_s8(py + 0 * I2S_Y_GROUP + iy * by);
                    const int8x16_t yq8_1 = vld1q_s8(py + 1 * I2S_Y_GROUP + iy * by);
                    const int8x16_t yq8_2 = vld1q_s8(py + 2 * I2S_Y_GROUP + iy * by);
                    const int8x16_t yq8_3 = vld1q_s8(py + 3 * I2S_Y_GROUP + iy * by);

#if defined(__ARM_FEATURE_DOTPROD)
                    accu[iy] = vdotq_s32(accu[iy], q8_0, yq8_0);
                    accu[iy] = vdotq_s32(accu[iy], q8_1, yq8_1);
                    accu[iy] = vdotq_s32(accu[iy], q8_2, yq8_2);
                    accu[iy] = vdotq_s32(accu[iy], q8_3, yq8_3);
#else
                    accu32[iy] = vmlal_s8(accu32[iy], vget_low_s8(q8_0), vget_low_s8(yq8_0));
                    accu32[iy] = vmlal_s8(accu32[iy], vget_high_s8(q8_0), vget_high_s8(yq8_0));
                    accu32[iy] = vmlal_s8(accu32[iy], vget_low_s8(q8_1), vget_low_s8(yq8_1));
                    accu32[iy] = vmlal_s8(accu32[iy], vget_high_s8(q8_1), vget_high_s8(yq8_1));
                    accu32[iy] = vmlal_s8(accu32[iy], vget_low_s8(q8_2), vget_low_s8(yq8_2));
                    accu32[iy] = vmlal_s8(accu32[iy], vget_high_s8(q8_2), vget_high_s8(yq8_2));
                    accu32[iy] = vmlal_s8(accu32[iy], vget_low_s8(q8_3), vget_low_s8(yq8_3));
                    accu32[iy] = vmlal_s8(accu32[iy], vget_high_s8(q8_3), vget_high_s8(yq8_3));
#endif
                }

                px += 16;
                py += (j & 1) ? 112 : 16;
            }

#if defined(__ARM_FEATURE_DOTPROD)

#else
            for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                accu[iy] = vaddq_s32(accu[iy], vaddq_s32(vmovl_high_s16(accu32[iy]), vmovl_s16(vget_low_s16(accu32[iy]))));
            }
#endif
        }

        for (int i = 0; i < groupla_num; i++) {
            const uint8_t *px = x + group32_num * 512;
            const int8_t  *py = y_col + I2S_Y_BASE(group32_num * 32);

#if defined(__ARM_FEATURE_DOTPROD)

#else
            int16x8_t accula[PARALLEL_SIZE];

            for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                accula[iy] = vdupq_n_s16(0);
            }
#endif

            for (int j = 0; j < la_num; j++) {
                uint8x16_t xq8_3 = vld1q_u8(px + 0);
                uint8x16_t xq8_2 = vshrq_n_u8(xq8_3, 2);
                uint8x16_t xq8_1 = vshrq_n_u8(xq8_3, 4);
                uint8x16_t xq8_0 = vshrq_n_u8(xq8_3, 6);

                int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
                int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));

                for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                    const int8x16_t yq8_0 = vld1q_s8(py + 0 * I2S_Y_GROUP + iy * by);
                    const int8x16_t yq8_1 = vld1q_s8(py + 1 * I2S_Y_GROUP + iy * by);
                    const int8x16_t yq8_2 = vld1q_s8(py + 2 * I2S_Y_GROUP + iy * by);
                    const int8x16_t yq8_3 = vld1q_s8(py + 3 * I2S_Y_GROUP + iy * by);

#if defined(__ARM_FEATURE_DOTPROD)
                    accu[iy] = vdotq_s32(accu[iy], q8_0, yq8_0);
                    accu[iy] = vdotq_s32(accu[iy], q8_1, yq8_1);
                    accu[iy] = vdotq_s32(accu[iy], q8_2, yq8_2);
                    accu[iy] = vdotq_s32(accu[iy], q8_3, yq8_3);
#else
                    accula[iy] = vmlal_s8(accula[iy], vget_low_s8(q8_0), vget_low_s8(yq8_0));
                    accula[iy] = vmlal_s8(accula[iy], vget_high_s8(q8_0), vget_high_s8(yq8_0));
                    accula[iy] = vmlal_s8(accula[iy], vget_low_s8(q8_1), vget_low_s8(yq8_1));
                    accula[iy] = vmlal_s8(accula[iy], vget_high_s8(q8_1), vget_high_s8(yq8_1));
                    accula[iy] = vmlal_s8(accula[iy], vget_low_s8(q8_2), vget_low_s8(yq8_2));
                    accula[iy] = vmlal_s8(accula[iy], vget_high_s8(q8_2), vget_high_s8(yq8_2));
                    accula[iy] = vmlal_s8(accula[iy], vget_low_s8(q8_3), vget_low_s8(yq8_3));
                    accula[iy] = vmlal_s8(accula[iy], vget_high_s8(q8_3), vget_high_s8(yq8_3));
#endif
                }

                px += 16;
                py += (j & 1) ? 112 : 16;
            }

#if defined(__ARM_FEATURE_DOTPROD)

#else
            for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                accu[iy] = vaddq_s32(accu[iy], vaddq_s32(vmovl_high_s16(accula[iy]), vmovl_s16(vget_low_s16(accula[iy]))));
            }
#endif
        }

        for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
            int sumi = vaddlvq_s32(accu[iy]);
            s[(col + iy) * bs] = (float)sumi;
        }
    }
#endif
}


void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    VIBEASR_PROBE(nrc % PARALLEL_SIZE == 0
                      ? (ACT_PARALLEL_SELECTED ? VIBEASR_K_I2_Nx1 : VIBEASR_K_I2_1xN)
                      : VIBEASR_K_I2_1x1,
                  (uint64_t) n * (uint64_t) nrc);
    if (nrc % PARALLEL_SIZE == 0)
    {
#if defined(ACT_PARALLEL)
        ggml_vec_dot_i2_i8_s_Nx1(n, s, bs, vx, bx, vy, by, nrc);
#else
        ggml_vec_dot_i2_i8_s_1xN(n, s, bs, vx, bx, vy, by, nrc);
#endif
    }
    else
    {
        ggml_vec_dot_i2_i8_s_1x1(n, s, bs, vx, bx, vy, by, nrc);
    }
}

size_t quantize_i8_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights;
    
    int n = nrow * n_per_row;
    
    float max_val = 0.0f;
    for (int i = 0; i < n; ++i) {
        float abs_val = fabsf(src[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    
    const float scale = (max_val > 0.0f) ? (127.0f / max_val) : 1.0f;
    
    int8_t * i8_weight = (int8_t *)dst;
    for (int i = 0; i < n; ++i) {
        float val = src[i] * scale;
        int32_t qval = (int32_t)roundf(val);
        if (qval > 127) qval = 127;
        if (qval < -127) qval = -127;
        i8_weight[i] = (int8_t)qval;
    }
    
    float * scale_ptr = (float *)((char *)i8_weight + nrow * n_per_row);
    scale_ptr[0] = 1.0f / scale;
    
    return nrow * n_per_row + 32;
}

