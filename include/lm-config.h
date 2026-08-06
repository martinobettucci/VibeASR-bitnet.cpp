// ROW_BLOCK_SIZE is the LM's counterpart to VAE_ROW_BLOCK_SIZE: how many activation
// rows go into one vec_dot call in ggml_gemm_i2_i8_s, which is the prefill path
// (~30% of inference). Same shape of problem, same knob; see vae-config.h for the
// measurement behind the VAE value and bench/row_block_sweep.sh to re-run it.
#define ACT_PARALLEL
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#if defined(ACT_PARALLEL)
    #ifndef ROW_BLOCK_SIZE
        #define ROW_BLOCK_SIZE 4
    #endif
    #define COL_BLOCK_SIZE 32
    #define PARALLEL_SIZE 4
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

