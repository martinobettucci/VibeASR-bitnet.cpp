// Runtime x86 ISA detection for the VibeASR kernels.
//
// The kernels ship several implementations of the same dot product and pick one on
// first use, so a single binary runs the widest path the host actually supports:
//
//   AMX-INT8     tile matmul          (Sapphire Rapids and later)
//   AVX-512 VNNI vpdpbusd, 64B lanes  (Cascade Lake and later)
//   AVX-512 F/BW 64B lanes            (Skylake-X and later)
//   AVX2                              (baseline for this project on x86)
//
// Selection can be overridden for benchmarking with VIBEASR_ISA=avx2|avx512|vnni|amx.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

enum vibeasr_isa {
    VIBEASR_ISA_SCALAR = 0,
    VIBEASR_ISA_AVX2   = 1,
    VIBEASR_ISA_AVX512 = 2,   // AVX512F + BW + VL + DQ
    VIBEASR_ISA_VNNI   = 3,   // the above + AVX512_VNNI
    VIBEASR_ISA_AMX    = 4,   // the above + AMX-TILE + AMX-INT8, tiles enabled by the OS
};

// Highest ISA usable on this process, honouring VIBEASR_ISA. Cheap after the first
// call (the result is cached in a global initialised on first use).
int vibeasr_isa(void);

// Highest ISA the CPU reports, ignoring the VIBEASR_ISA override. For diagnostics.
int vibeasr_isa_detected(void);

const char * vibeasr_isa_name(int isa);

#ifdef __cplusplus
}
#endif
