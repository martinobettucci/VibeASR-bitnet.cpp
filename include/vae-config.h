// VAE_ROW_BLOCK_SIZE is the number of activation rows handed to one vec_dot call in
// the I8_S GEMM, and it is the dominant cost knob for the VAE encoder -- which is
// about 60% of total inference time.
//
// At the original value of 4 a single 8.4 s clip issues ~194 million vec_dot calls
// averaging ~1200 MACs each, running at roughly 2% of this CPU's int8 peak. The work
// is not the problem; the per-call prologue, epilogue and horizontal reduction are.
// Raising the block amortises all three over more rows without changing any result.
//
// Measured with bench/row_block_sweep.sh on Xeon @2.8 GHz (Cascade Lake, AVX-512
// VNNI, 4 cores), 8.4 s clip, VAE encode acoustic + semantic:
//
//   block    4 (upstream)   9718 ms   194.8M vec_dot calls
//   block   16              6524 ms    52.9M
//   block   32              6289 ms    31.9M     <- default
//   block   64              6504 ms    25.1M
//
// Past 32 the activation rows stop fitting in L1 and the win from fewer calls is
// spent again on cache misses. Override at configure time with
// -DVAE_ROW_BLOCK_SIZE=N to re-tune -- note it must reach the C compiler as well as
// the C++ one, since the blocking loop lives in ggml-aarch64.c.
#define VAE_ACT_PARALLEL
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#if defined(VAE_ACT_PARALLEL)
    #ifndef VAE_ROW_BLOCK_SIZE
        #define VAE_ROW_BLOCK_SIZE 32
    #endif
    // Also sweepable via row_block_sweep.sh --macro; both are still at upstream's
    // values because nothing has measured them yet.
    #ifndef VAE_COL_BLOCK_SIZE
        #define VAE_COL_BLOCK_SIZE 16
    #endif
    #ifndef VAE_PARALLEL_SIZE
        #define VAE_PARALLEL_SIZE 4
    #endif
#else
    #define VAE_ROW_BLOCK_SIZE 16
    #define VAE_COL_BLOCK_SIZE 4
    #define VAE_PARALLEL_SIZE 4
#endif
#elif defined(__ARM_NEON)
#if defined(VAE_ACT_PARALLEL)
    #define VAE_ROW_BLOCK_SIZE 4
    #define VAE_COL_BLOCK_SIZE 16
    #define VAE_PARALLEL_SIZE 4
#else
    #define VAE_ROW_BLOCK_SIZE 16
    #define VAE_COL_BLOCK_SIZE 4
    #define VAE_PARALLEL_SIZE 4
#endif
#endif
