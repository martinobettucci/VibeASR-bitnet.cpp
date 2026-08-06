// Re-quantise a single tensor of an already-quantised VibeASR LM GGUF, copying
// everything else through untouched.
//
// Why this exists: the released vibeasr-lm-i2_s-embed-q6_k.gguf carries
// output.weight as F16 -- 151936 x 1536 x 2 B = 467 MB, 47% of the 993 MB file --
// while token_embd.weight beside it is already Q6_K at 191 MB. The transformer body
// is I2_S at roughly 330 MB. So nearly half the model, and nearly half the memory
// traffic of every decoded token, is one unquantised matrix.
//
// llama-quantize cannot fix this: it has no "leave these tensors alone" mode, so it
// would dequantise the I2_S body and destroy the BitNet weights. Hence a tool that
// touches exactly one tensor.
//
//   ./build/bin/requant_lm_head in.gguf out.gguf [--type q6_k|q5_k|q4_k|q8_0]
//                               [--tensor output.weight] [--drop]
//
// --drop removes the tensor instead of re-quantising it. For output.weight that is
// the better move here: the source checkpoint has tie_word_embeddings=true and
// lm_head.weight is bit-identical to embed_tokens.weight, so the F16 copy is a
// duplicate of a matrix the file already stores as Q6_K. llama.cpp loads
// LLM_TENSOR_OUTPUT as TENSOR_NOT_REQUIRED and falls back to token_embd when it is
// missing, so dropping it is numerically the same as re-quantising to Q6_K while
// keeping only one copy resident.
//
// Rows are quantised in parallel across the requested thread count.

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static ggml_type parse_type(const std::string & s) {
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q6_k") return GGML_TYPE_Q6_K;
    if (s == "q5_k") return GGML_TYPE_Q5_K;
    if (s == "q4_k") return GGML_TYPE_Q4_K;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    return GGML_TYPE_COUNT;
}

static void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s <in.gguf> <out.gguf> [--type q6_k | --drop] [--tensor output.weight] [-t N]\n"
            "  --type    target quantisation (q8_0 q6_k q5_k q4_k q4_0), default q6_k\n"
            "  --drop    remove the tensor instead of re-quantising it\n"
            "  --tensor  tensor to act on, default output.weight\n"
            "  -t        threads for quantisation, default hardware concurrency\n", prog);
}

int main(int argc, char ** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }

    const std::string fin = argv[1];
    const std::string fout = argv[2];
    std::string target_name = "output.weight";
    ggml_type target_type = GGML_TYPE_Q6_K;
    bool drop = false;
    int nthreads = (int) std::thread::hardware_concurrency();

    for (int i = 3; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--drop") {
            drop = true;
        } else if (a == "--type" && i + 1 < argc) {
            target_type = parse_type(argv[++i]);
            if (target_type == GGML_TYPE_COUNT) { fprintf(stderr, "unknown --type\n"); return 1; }
        } else if (a == "--tensor" && i + 1 < argc) {
            target_name = argv[++i];
        } else if (a == "-t" && i + 1 < argc) {
            nthreads = atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }
    if (nthreads < 1) nthreads = 1;

    struct ggml_context * ctx_data = NULL;
    struct gguf_init_params gp = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_data };
    struct gguf_context * gin = gguf_init_from_file(fin.c_str(), gp);
    if (!gin) { fprintf(stderr, "failed to open %s\n", fin.c_str()); return 1; }

    struct gguf_context * gout = gguf_init_empty();
    gguf_set_kv(gout, gin);  // architecture, tokenizer, hyperparameters -- all of it

    struct ggml_tensor * src = ggml_get_tensor(ctx_data, target_name.c_str());
    if (!src) {
        fprintf(stderr, "tensor '%s' not found in %s\n", target_name.c_str(), fin.c_str());
        return 1;
    }
    if (!ggml_is_contiguous(src)) {
        fprintf(stderr, "tensor '%s' is not contiguous\n", target_name.c_str());
        return 1;
    }

    const int64_t n_per_row = src->ne[0];
    const int64_t nrows     = ggml_nrows(src);

    if (drop) {
        size_t bytes_in = 0, bytes_out = 0;
        for (int i = 0; i < gguf_get_n_tensors(gin); i++) {
            const char * name = gguf_get_tensor_name(gin, i);
            struct ggml_tensor * t = ggml_get_tensor(ctx_data, name);
            bytes_in += ggml_nbytes(t);
            if (target_name == name) continue;
            gguf_add_tensor(gout, t);
            bytes_out += ggml_nbytes(t);
        }
        gguf_write_to_file(gout, fout.c_str(), /*only_meta =*/ false);
        printf("dropped %-18s %8.1f MB\n", target_name.c_str(), ggml_nbytes(src) / 1e6);
        printf("model   %8.1f MB -> %8.1f MB  (%.1f%% smaller)\n",
               bytes_in / 1e6, bytes_out / 1e6, 100.0 * (1.0 - (double) bytes_out / bytes_in));
        printf("wrote %s\n", fout.c_str());
        ggml_free(ctx_data);
        gguf_free(gin);
        gguf_free(gout);
        return 0;
    }

    if (n_per_row % ggml_blck_size(target_type) != 0) {
        fprintf(stderr, "row length %lld is not a multiple of the %s block size %d\n",
                (long long) n_per_row, ggml_type_name(target_type), ggml_blck_size(target_type));
        return 1;
    }

    printf("re-quantising %-18s %5lld x %-7lld  %s -> %s\n", target_name.c_str(),
           (long long) n_per_row, (long long) nrows,
           ggml_type_name(src->type), ggml_type_name(target_type));

    // Dequantise to F32 once; ggml_quantize_chunk always takes float input.
    std::vector<float> f32((size_t) nrows * n_per_row);
    if (src->type == GGML_TYPE_F32) {
        memcpy(f32.data(), src->data, f32.size() * sizeof(float));
    } else {
        const struct ggml_type_traits * tr = ggml_get_type_traits(src->type);
        if (!tr || !tr->to_float) {
            fprintf(stderr, "no dequantiser for source type %s\n", ggml_type_name(src->type));
            return 1;
        }
        const size_t row_bytes = ggml_row_size(src->type, n_per_row);
        for (int64_t r = 0; r < nrows; r++) {
            tr->to_float((const char *) src->data + r * row_bytes,
                        f32.data() + r * n_per_row, (int) n_per_row);
        }
    }

    // Quantise row blocks in parallel. ggml_quantize_chunk writes each block of rows
    // to its own slice of the destination, so the threads never overlap.
    std::vector<char> qdata(ggml_row_size(target_type, n_per_row) * nrows);
    const int64_t chunk = (nrows + nthreads - 1) / nthreads;
    std::vector<std::thread> pool;
    for (int t = 0; t < nthreads; t++) {
        const int64_t r0 = t * chunk;
        const int64_t r1 = r0 + chunk < nrows ? r0 + chunk : nrows;
        if (r0 >= r1) break;
        pool.emplace_back([&, r0, r1]() {
            ggml_quantize_chunk(target_type, f32.data() + r0 * n_per_row,
                                qdata.data() + ggml_row_size(target_type, n_per_row) * r0,
                                r0, r1 - r0, n_per_row, /*imatrix =*/ NULL);
        });
    }
    for (auto & th : pool) th.join();

    // Rebuild the tensor list in the original order, swapping in the new tensor.
    struct ggml_init_params mp = { /*.mem_size =*/ ggml_tensor_overhead() * (gguf_get_n_tensors(gin) + 1),
                                   /*.mem_buffer =*/ NULL, /*.no_alloc =*/ true };
    struct ggml_context * ctx_meta = ggml_init(mp);
    struct ggml_tensor * newt = ggml_new_tensor_2d(ctx_meta, target_type, n_per_row, nrows);
    ggml_set_name(newt, target_name.c_str());

    size_t bytes_in = 0, bytes_out = 0;
    for (int i = 0; i < gguf_get_n_tensors(gin); i++) {
        const char * name = gguf_get_tensor_name(gin, i);
        struct ggml_tensor * t = ggml_get_tensor(ctx_data, name);
        bytes_in += ggml_nbytes(t);
        if (target_name == name) {
            gguf_add_tensor(gout, newt);
            gguf_set_tensor_data(gout, name, qdata.data(), qdata.size());
            bytes_out += qdata.size();
        } else {
            gguf_add_tensor(gout, t);
            bytes_out += ggml_nbytes(t);
        }
    }

    gguf_write_to_file(gout, fout.c_str(), /*only_meta =*/ false);

    printf("tensor  %8.1f MB -> %8.1f MB\n", ggml_nbytes(src) / 1e6, qdata.size() / 1e6);
    printf("model   %8.1f MB -> %8.1f MB  (%.1f%% smaller)\n",
           bytes_in / 1e6, bytes_out / 1e6, 100.0 * (1.0 - (double) bytes_out / bytes_in));
    printf("wrote %s\n", fout.c_str());

    ggml_free(ctx_meta);
    ggml_free(ctx_data);
    gguf_free(gin);
    gguf_free(gout);
    return 0;
}
