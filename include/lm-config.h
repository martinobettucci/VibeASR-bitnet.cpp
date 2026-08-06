// ROW_BLOCK_SIZE is the LM's counterpart to VAE_ROW_BLOCK_SIZE: how many activation
// rows go into one vec_dot call in ggml_gemm_i2_i8_s, which is the prefill path
// (~30% of inference). Same shape of problem, same knob; see vae-config.h for the
// measurement behind the VAE value and bench/row_block_sweep.sh to re-run it.
#define ACT_PARALLEL
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#if defined(ACT_PARALLEL)
    // Measured with bench/row_block_sweep.sh --macro ROW_BLOCK_SIZE on Xeon @2.8 GHz
    // (Cascade Lake, AVX-512 VNNI, 4 cores), 8.4 s clip, LM prefill:
    //
    //   block    4 (upstream)   2727 ms
    //   block   16              1025 ms
    //   block   32               925 ms   <- default
    //   block   64               935 ms
    //
    // Decode is unaffected: it is a GEMV with nrc=1, so there are no rows to block.
    #ifndef ROW_BLOCK_SIZE
        #define ROW_BLOCK_SIZE 32
    #endif
    #ifndef COL_BLOCK_SIZE
        #define COL_BLOCK_SIZE 32
    #endif
    #ifndef PARALLEL_SIZE
        #define PARALLEL_SIZE 4
    #endif
#else
    #define ROW_BLOCK_SIZE 32
    #define COL_BLOCK_SIZE 4
    #define PARALLEL_SIZE 4
#endif
#elif defined(__ARM_NEON)
#if defined(ACT_PARALLEL)
    #define ROW_BLOCK_SIZE 4
    #define COL_BLOCK_SIZE 32
    #define PARALLEL_SIZE 4
#else
    #define ROW_BLOCK_SIZE 32
    #define COL_BLOCK_SIZE 4
    #define PARALLEL_SIZE 4
#endif
#endif

