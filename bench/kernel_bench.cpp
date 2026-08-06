// Correctness and throughput harness for the VibeASR I2_S / I8_S kernels.
//
//   ./build/bin/kernel_bench --check      # correctness only
//   ./build/bin/kernel_bench              # correctness + throughput
//   VIBEASR_ISA=avx2 ./build/bin/kernel_bench   # force the baseline path
//
// Every SIMD path is checked against an exact int32 scalar reference rather than
// against the other SIMD paths: the three blocked variants compute genuinely
// different things (1x1/1xN walk rows of x against one y; Nx1 walks columns of y
// against one x), so cross-comparing them proves nothing.
//
// Two magnitude regimes are checked. "narrow" keeps operands small enough that the
// AVX2 kernels' int16 accumulator cannot overflow; "full" uses the whole +-127 range
// that quantize_i8_s can actually emit. The AVX2 kernels accumulate vpmaddubsw
// results in int16 and only flush to int32 every 32 blocks, so they can wrap in the
// full regime -- the AVX-512 VNNI paths accumulate in int32 throughout and cannot.
// Reporting both regimes separately keeps that distinction visible.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

#include "ggml-vae-i8_s-mad.h"
#include "lm-config.h"
#include "vae-config.h"
#include "vibeasr-cpu.h"

extern "C" {
void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
}

// Only the dispatch entry points are declared in the C headers; the blocked variants
// are defined in the .cpp files with no C declaration, so they keep C++ linkage.
void ggml_vec_dot_i8_i8_1x1(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_i8_i8_1xN(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_i8_i8_Nx1(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_i2_i8_s_1x1(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_i2_i8_s_1xN(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_i2_i8_s_Nx1(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

static double now_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

static uint64_t rng_state;
static void rng_seed(uint64_t s) { rng_state = s ? s : 1; }
static uint32_t rng() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t) (rng_state >> 32);
}

// amp <= 127; quantize_i8_s clamps to +-127 so -128 never reaches these kernels.
static void fill_i8(int8_t * p, size_t n, int amp) {
    for (size_t i = 0; i < n; i++) p[i] = (int8_t) ((int) (rng() % (2 * amp + 1)) - amp);
}

static void fill_i2_packed(uint8_t * p, size_t nbytes) {
    for (size_t i = 0; i < nbytes; i++) {
        uint8_t b = 0;
        for (int g = 0; g < 4; g++) b |= (uint8_t) ((rng() % 3) << (6 - 2 * g));
        p[i] = b;
    }
}

// --- scalar references -----------------------------------------------------

static int32_t ref_i8(const int8_t * x, const int8_t * y, int nbytes) {
    int32_t s = 0;
    for (int k = 0; k < nbytes; k++) s += (int32_t) x[k] * (int32_t) y[k];
    return s;
}

// Unpacks the 2-bit block layout: group g of weight j lives in bits [7-2g : 6-2g] of
// byte (block*32 + j), and its activation is y[block*128 + g*32 + j].
static int32_t ref_i2(const uint8_t * x, const int8_t * y, int nb) {
    int32_t s = 0;
    for (int b = 0; b < nb; b++) {
        for (int g = 0; g < 4; g++) {
            for (int j = 0; j < 32; j++) {
                const uint8_t w = (uint8_t) ((x[b * 32 + j] >> (6 - 2 * g)) & 0x03);
                s += (int32_t) w * (int32_t) y[b * 128 + g * 32 + j];
            }
        }
    }
    return s;
}

struct Tally {
    int checked = 0;
    int wrong = 0;
    void note(bool ok) { checked++; if (!ok) wrong++; }
};

// --- correctness -----------------------------------------------------------

static void check_i8(int n, int nrc, int amp, Tally & t, bool verbose) {
    const int nbytes = (n / 32) * 32;  // the kernels consume whole QK_I8_S blocks
    std::vector<int8_t> x((size_t) n * nrc), y((size_t) n * nrc);
    fill_i8(x.data(), x.size(), amp);
    fill_i8(y.data(), y.size(), amp);

    std::vector<int32_t> got(nrc);

    // 1x1 and 1xN: s[r] = dot(x_row r, y)
    for (int variant = 0; variant < 2; variant++) {
        std::fill(got.begin(), got.end(), 0);
        (variant ? ggml_vec_dot_i8_i8_1xN : ggml_vec_dot_i8_i8_1x1)(
            n, got.data(), 1, x.data(), n, y.data(), n, nrc);
        for (int r = 0; r < nrc; r++) {
            const int32_t want = ref_i8(x.data() + (size_t) r * n, y.data(), nbytes);
            const bool ok = got[r] == want;
            t.note(ok);
            if (!ok && verbose) {
                printf("    i8 %s n=%d row=%d: got %d want %d\n",
                       variant ? "1xN" : "1x1", n, r, got[r], want);
                verbose = false;  // one line per shape is enough
            }
        }
    }

    // Nx1: s[c] = dot(x, y_col c)
    std::fill(got.begin(), got.end(), 0);
    ggml_vec_dot_i8_i8_Nx1(n, got.data(), 1, x.data(), n, y.data(), n, nrc);
    for (int c = 0; c < nrc; c++) {
        const int32_t want = ref_i8(x.data(), y.data() + (size_t) c * n, nbytes);
        const bool ok = got[c] == want;
        t.note(ok);
        if (!ok && verbose) {
            printf("    i8 Nx1 n=%d col=%d: got %d want %d\n", n, c, got[c], want);
            verbose = false;
        }
    }
}

static void check_i2(int n, int nrc, int amp, Tally & t, bool verbose) {
    const int nb = n / 128;  // QK_I2_S
    std::vector<uint8_t> x((size_t) n / 4 * nrc);
    std::vector<int8_t>  y((size_t) n * nrc);
    fill_i2_packed(x.data(), x.size());
    fill_i8(y.data(), y.size(), amp);

    std::vector<float> got(nrc);

    for (int variant = 0; variant < 2; variant++) {
        std::fill(got.begin(), got.end(), 0.f);
        (variant ? ggml_vec_dot_i2_i8_s_1xN : ggml_vec_dot_i2_i8_s_1x1)(
            n, got.data(), 1, x.data(), n, y.data(), n, nrc);
        for (int r = 0; r < nrc; r++) {
            const int32_t want = ref_i2(x.data() + (size_t) r * (n / 4), y.data(), nb);
            const bool ok = got[r] == (float) want;
            t.note(ok);
            if (!ok && verbose) {
                printf("    i2 %s n=%d row=%d: got %.0f want %d\n",
                       variant ? "1xN" : "1x1", n, r, got[r], want);
                verbose = false;
            }
        }
    }

    std::fill(got.begin(), got.end(), 0.f);
    ggml_vec_dot_i2_i8_s_Nx1(n, got.data(), 1, x.data(), n, y.data(), n, nrc);
    for (int c = 0; c < nrc; c++) {
        const int32_t want = ref_i2(x.data(), y.data() + (size_t) c * n, nb);
        const bool ok = got[c] == (float) want;
        t.note(ok);
        if (!ok && verbose) {
            printf("    i2 Nx1 n=%d col=%d: got %.0f want %d\n", n, c, got[c], want);
            verbose = false;
        }
    }
}

// --- throughput ------------------------------------------------------------

int main(int argc, char ** argv) {
    const bool check_only = argc > 1 && !strcmp(argv[1], "--check");

    printf("cpu supports: %-12s   using: %s\n\n",
           vibeasr_isa_name(vibeasr_isa_detected()), vibeasr_isa_name(vibeasr_isa()));

    // Shapes from the model: VAE stage widths 32..8192, LM hidden 1536 / FFN 8960.
    // nrc values follow the GEMM row blocks: 32 is what prefill now issues, 4 and 16
    // are kept because the earlier routing decision was made at those widths.
    const int i8_shapes[][2] = {{32, 32}, {128, 32}, {512, 32}, {2048, 32}, {8192, 32}};
    const int i2_shapes[][2] = {{1536, 4}, {1536, 16}, {1536, 32}, {8960, 16}, {8960, 32}};

    int rc = 0;
    struct { const char * label; int amp; bool fatal; } regimes[] = {
        {"narrow (|v| <= 8, no int16 overflow possible)", 8,   true},
        {"full   (|v| <= 127, as quantize_i8_s emits)",   127, false},
    };

    for (auto & reg : regimes) {
        Tally t;
        printf("%s\n", reg.label);
        for (auto & sh : i8_shapes) { rng_seed(0x243f6a88); check_i8(sh[0], sh[1], reg.amp, t, true); }
        for (auto & sh : i2_shapes) { rng_seed(0x85a308d3); check_i2(sh[0], sh[1], reg.amp, t, true); }
        printf("  %d/%d dot products match the scalar reference%s\n\n",
               t.checked - t.wrong, t.checked, t.wrong ? "" : "  [ok]");
        if (t.wrong && reg.fatal) rc = 1;
    }

    if (check_only) return rc;

    // --- throughput ---
    struct Timing { std::string name; double gmacs; };
    std::vector<Timing> out;

    for (auto & sh : i8_shapes) {
        const int n = sh[0], nrc = sh[1];
        const int iters = (int) (4e8 / ((double) n * nrc)) + 1;
        std::vector<int8_t> x((size_t) n * nrc), y((size_t) n * nrc);
        rng_seed(0x243f6a88);
        fill_i8(x.data(), x.size(), 127);
        fill_i8(y.data(), y.size(), 127);
        std::vector<int32_t> s(nrc);
        struct { const char * nm; void (*fn)(int, int32_t *, size_t, const void *, size_t, const void *, size_t, int); }
            ks[] = {{"i8_s 1x1", ggml_vec_dot_i8_i8_1x1},
                    {"i8_s 1xN", ggml_vec_dot_i8_i8_1xN},
                    {"i8_s Nx1", ggml_vec_dot_i8_i8_Nx1}};
        for (auto & k : ks) {
            k.fn(n, s.data(), 1, x.data(), n, y.data(), n, nrc);
            const double t0 = now_s();
            for (int it = 0; it < iters; it++) k.fn(n, s.data(), 1, x.data(), n, y.data(), n, nrc);
            const double dt = now_s() - t0;
            char buf[64];
            snprintf(buf, sizeof buf, "%s n=%-5d nrc=%-3d", k.nm, n, nrc);
            out.push_back({buf, (double) n * nrc * iters / dt / 1e9});
        }
    }

    for (auto & sh : i2_shapes) {
        const int n = sh[0], nrc = sh[1];
        const int iters = (int) (4e8 / ((double) n * nrc)) + 1;
        std::vector<uint8_t> x((size_t) n / 4 * nrc);
        std::vector<int8_t>  y((size_t) n * nrc);
        rng_seed(0x85a308d3);
        fill_i2_packed(x.data(), x.size());
        fill_i8(y.data(), y.size(), 127);
        std::vector<float> s(nrc);
        struct { const char * nm; void (*fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int); }
            ks[] = {{"i2_s 1x1", ggml_vec_dot_i2_i8_s_1x1},
                    {"i2_s 1xN", ggml_vec_dot_i2_i8_s_1xN},
                    {"i2_s Nx1", ggml_vec_dot_i2_i8_s_Nx1}};
        for (auto & k : ks) {
            k.fn(n, s.data(), 1, x.data(), n, y.data(), n, nrc);
            const double t0 = now_s();
            for (int it = 0; it < iters; it++) k.fn(n, s.data(), 1, x.data(), n, y.data(), n, nrc);
            const double dt = now_s() - t0;
            char buf[64];
            snprintf(buf, sizeof buf, "%s n=%-5d nrc=%-3d", k.nm, n, nrc);
            out.push_back({buf, (double) n * nrc * iters / dt / 1e9});
        }
    }

    printf("throughput, single thread (%s)\n", vibeasr_isa_name(vibeasr_isa()));
    printf("  %-30s %10s\n", "kernel / shape", "GMAC/s");
    for (auto & r : out) printf("  %-30s %10.1f\n", r.name.c_str(), r.gmacs);
    return rc;
}
