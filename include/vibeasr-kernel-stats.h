// Opt-in accounting for time spent inside the hand-written I2_S / I8_S kernels.
//
// Set VIBEASR_KERNEL_STATS=1 to get a per-kernel breakdown (wall time, calls, MACs)
// on exit. It answers the question that decides where optimisation effort belongs:
// how much of a run is actually inside these kernels versus the surrounding ggml
// graph machinery (im2col, activation quantisation, norms, copies).
//
// Disabled builds cost nothing: the probe compiles to a load of a global bool.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum vibeasr_kernel_id {
    VIBEASR_K_I2_1x1 = 0,
    VIBEASR_K_I2_1xN,
    VIBEASR_K_I2_Nx1,
    VIBEASR_K_I8_1x1,
    VIBEASR_K_I8_1xN,
    VIBEASR_K_I8_Nx1,
    VIBEASR_K_I8_SMALL,
    VIBEASR_K_I8_BATCH_N8,
    VIBEASR_K_COUNT
};

extern int vibeasr_kernel_stats_on;

void vibeasr_kernel_stats_init(void);
void vibeasr_kernel_stats_add(int id, uint64_t ns, uint64_t macs);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <chrono>

namespace vibeasr {

// Scoped probe. Construction/destruction is skipped entirely when stats are off.
struct kernel_probe {
    int id;
    uint64_t macs;
    std::chrono::steady_clock::time_point t0;

    kernel_probe(int id_, uint64_t macs_) : id(id_), macs(macs_) {
        if (vibeasr_kernel_stats_on) {
            t0 = std::chrono::steady_clock::now();
        }
    }

    ~kernel_probe() {
        if (vibeasr_kernel_stats_on) {
            const auto dt = std::chrono::steady_clock::now() - t0;
            vibeasr_kernel_stats_add(
                id, (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count(), macs);
        }
    }
};

}  // namespace vibeasr

// One probe per scope; the name is fixed so a nested probe is a compile error rather
// than silent double counting.
#define VIBEASR_PROBE(id, macs) ::vibeasr::kernel_probe vibeasr_scoped_probe((id), (uint64_t)(macs))

#endif  // __cplusplus
