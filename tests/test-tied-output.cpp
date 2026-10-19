#include "llama-model.h"
#include "ggml-alloc.h"
#include "ggml-quants.h"
#include "iqk/iqk_mul_mat.h"
#include "iqk/iqk_quantize.h"
#include "iqk/iqk_cpu_ops.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::abort();
    }
}

static void test_format(ggml_type type, int group_rows, bool same_tensor = false) {
    constexpr int width = 256, rows = 16;
    llama_model model{};
    ggml_init_params params = { 4 * ggml_tensor_overhead(), nullptr, true };
    auto * ctx = ggml_init(params);
    model.ctxs.push_back(ctx);
    model.tok_embd = ggml_new_tensor_2d(ctx, type, width, rows);
    ggml_set_name(model.tok_embd, "token_embd.weight");
    model.output = ggml_dup_tensor(ctx, model.tok_embd);
    ggml_set_name(model.output, model.tok_embd->name);
    auto * buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    check(buf != nullptr, "allocation failed");
    model.bufs.push_back(buf);
    std::vector<float> weights(width * rows);
    for (int i = 0; i < width * rows; ++i) {
        weights[i] = float((i * 17 + i / 11) % 3 - 1);
    }
    auto traits = ggml_internal_get_type_traits(type);
    if (type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(model.tok_embd, weights.data(), 0, weights.size() * sizeof(float));
    } else {
        traits.from_float_ref(weights.data(), model.tok_embd->data, weights.size());
    }
    std::memcpy(model.output->data, model.tok_embd->data, ggml_nbytes(model.output));
    std::vector<char> saved(ggml_nbytes(model.tok_embd));
    std::memcpy(saved.data(), model.tok_embd->data, saved.size());

    // Distinct outputs must not be treated as tied merely because the shapes match.
    ggml_set_name(model.output, "output.weight");
    check(!model.repack_tied_output(), "untied output was repacked");
    ggml_set_name(model.output, model.tok_embd->name);
    if (same_tensor) {
        model.output = model.tok_embd;
    }
    auto * original = model.output;
    model.output_mtp = original;
    model.arch = LLM_ARCH_QWEN35;
    model.hadamard_map[original->name].block_size = 128;
    // Packed kernels require complete row groups; unsupported shapes stay as-is.
    const int64_t original_rows = original->ne[1];
    original->ne[1] = 15;
    check(!model.repack_tied_output(), "incomplete row group was repacked");
    original->ne[1] = original_rows;
    bool packed = model.repack_tied_output();
    check(packed == (type == GGML_TYPE_PQ2_0 || type == GGML_TYPE_PTQ1_0), "unexpected repack eligibility");
    check(model.tok_embd->type == type, "embedding format changed");
    check(std::memcmp(saved.data(), model.tok_embd->data, saved.size()) == 0, "embedding bytes changed");
    check(std::memcmp(saved.data(), original->data, saved.size()) == 0, "original output bytes changed");
    check(model.hadamard_rotation(model.output) == model.hadamard_rotation(model.tok_embd), "Hadamard identity lost");
    if (!packed) {
        return;
    }
    check(model.output != original, "output not detached");
    check(model.output_mtp == model.output, "MTP still uses old output");
    check(!model.repack_tied_output(), "output repacked twice");
    std::vector<float> reference(width * rows), actual(width * rows);
    traits.to_float(original->data, reference.data(), reference.size());
    const auto packed_traits = ggml_internal_get_type_traits(model.output->type);
    const auto group_bytes = ggml_row_size(type, width) * group_rows;
    for (int row = 0; row < rows; row += group_rows) {
        packed_traits.to_float((char *) model.output->data + row / group_rows * group_bytes,
                actual.data() + row * width, width * group_rows);
    }
    for (size_t i = 0; i < reference.size(); ++i) {
        check(std::fabs(reference[i] - actual[i]) < 1e-6f, "repacking changed weight values");
    }
    std::printf("%s: weights, embeddings, Hadamard identity and eligibility OK\n", ggml_type_name(type));
}

static void test_pq2_repack_layout() {
    // Compare every packed byte with a scalar reference, including all two-bit
    // codes and multiple row groups/blocks. Prefill output to detect unwritten bytes.
    for (int width : { 128, 256, 5120 }) {
        for (int rows : { 8, 24, 264 }) {
            const int blocks = width / QK_PQ2_0;
            std::vector<block_pq2_0> input(rows * blocks);
            uint32_t state = 42;
            for (auto & block : input) {
                block.d = ggml_fp32_to_fp16(0.25f + float(state % 16));
                for (auto & byte : block.qs) {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    byte = uint8_t(state);
                }
            }
            std::vector<block_pq2_0_r8> expected(rows / 8 * blocks), actual(expected.size());
            std::memset(actual.data(), 0xa5, actual.size() * sizeof(actual[0]));
            for (int group = 0; group < rows / 8; ++group) {
                for (int block = 0; block < blocks; ++block) {
                    auto & dst = expected[group * blocks + block];
                    for (int k = 0; k < 8; ++k) {
                        const auto & src = input[(group * 8 + k) * blocks + block];
                        dst.d[k] = src.d;
                        for (int p = 0; p < QK_PQ2_0; ++p) {
                            const int ib32 = p / 32;
                            const int l = (p % 32) / 4;
                            const int i = p % 4;
                            const int q = (src.qs[p / 4] >> (2 * (p % 4))) & 3;
                            dst.qs[64 * ib32 + 32 * (l / 4) + 4 * k + i] |= q << (2 * (l % 4));
                        }
                    }
                }
            }
            repack_pq2_0_r8(rows, width, input.data(), actual.data(), false);
            check(std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(actual[0])) == 0,
                    "PQ2 repacking differs from scalar reference");
        }
    }
}

static void test_ternary_matmul(ggml_type type, int rows, int tokens) {
    constexpr int width = 256;
    ggml_init_params params = { 1024 * 1024, nullptr, false };
    auto * ctx = ggml_init(params);
    auto * w = ggml_new_tensor_2d(ctx, type, width, rows);
    ggml_set_name(w, "output.weight");
    auto wt = ggml_internal_get_type_traits(type);
    std::vector<float> wf(width * rows), wref(width * rows);
    for (size_t i = 0; i < wf.size(); ++i) {
        wf[i] = float((i * 13 + i / 7) % 3) - 1.0f;
    }
    wt.from_float_ref(wf.data(), w->data, wf.size());
    wt.to_float(w->data, wref.data(), wref.size());
    iqk_repack_tensor(w);
    const auto qt = GGML_TYPE_Q8_K128;
    auto bt = ggml_internal_get_type_traits(qt);
    auto * b = ggml_new_tensor_2d(ctx, qt, width, tokens);
    std::vector<float> bf(width * tokens), bref(width * tokens);
    for (size_t i = 0; i < bf.size(); ++i) {
        bf[i] = std::sin(float(i) * 0.37f);
    }
    bt.from_float(bf.data(), b->data, bf.size());
    const auto * blocks = static_cast<const block_q8_K128 *>(b->data);
    for (size_t i = 0; i < bref.size(); ++i) {
        bref[i] = blocks[i / 128].d * blocks[i / 128].qs[i % 128];
    }
    std::vector<float> actual(rows * tokens, NAN);
    check(iqk_mul_mat(rows, tokens, width, w->type, w->data, ggml_row_size(w->type, width),
            qt, b->data, ggml_row_size(qt, width), actual.data(), rows, 0, 1), "ternary kernel unavailable");
    for (int t = 0; t < tokens; ++t) {
        for (int row = 0; row < rows; ++row) {
            float ref = 0;
            for (int k = 0; k < width; ++k) {
                ref += wref[row * width + k] * bref[t * width + k];
            }
            check(std::isfinite(actual[t * rows + row]) &&
                    std::fabs(actual[t * rows + row] - ref) < 1e-3f, "ternary matmul differs from reference");
        }
    }
    ggml_free(ctx);
}

static void test_hadamard(ggml_type type, int block_size) {
    const int width = block_size * 3;
    ggml_init_params params = { 8 * 1024 * 1024, nullptr, false };
    auto * ctx = ggml_init(params);
    auto * src = ggml_new_tensor_3d(ctx, type, width, 3, 2);
    std::vector<float> input(ggml_nelements(src)), expected(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(float(i) * 0.37f) * float(int(i % 7) - 3);
    }
    if (type == GGML_TYPE_F32) {
        std::memcpy(src->data, input.data(), input.size() * sizeof(float));
        expected = input;
    } else {
        auto traits = ggml_internal_get_type_traits(type);
        traits.from_float_ref(input.data(), src->data, input.size());
        traits.to_float(src->data, expected.data(), expected.size());
    }
    std::vector<char> saved(ggml_nbytes(src));
    std::memcpy(saved.data(), src->data, saved.size());
    for (size_t offset = 0; offset < expected.size(); offset += block_size) {
        float scale = 1;
        for (int h = 1; h < block_size; h *= 2) {
            for (int i = 0; i < block_size; i += 2*h) {
                for (int j = i; j < i + h; ++j) {
                    float x = expected[offset + j], y = expected[offset + j + h];
                    expected[offset + j] = x + y;
                    expected[offset + j + h] = x - y;
                }
            }
            scale *= 0.707106781f;
        }
        for (int i = 0; i < block_size; ++i) expected[offset + i] *= scale;
    }
    auto * dst = ggml_hadamard(ctx, src, block_size);
    for (int threads : { 1, 4, 7 }) {
        std::memset(dst->data, 0xff, ggml_nbytes(dst));
        for (int thread = 0; thread < threads; ++thread) {
            iqk_hadamard(dst, thread, threads);
        }
        check(std::memcmp(dst->data, expected.data(), expected.size() * sizeof(float)) == 0,
                "Hadamard differs from scalar reference");
        check(std::memcmp(src->data, saved.data(), saved.size()) == 0, "Hadamard changed its input");
    }
    ggml_free(ctx);
}

int main() {
    ggml_time_init();
    test_pq2_repack_layout();
    test_format(GGML_TYPE_PQ2_0, 8);
    test_format(GGML_TYPE_PQ2_0, 8, true);
    test_format(GGML_TYPE_PTQ1_0, 8);
    test_format(GGML_TYPE_Q4_0, 8);
    test_format(GGML_TYPE_Q6_K, 4);
    test_format(GGML_TYPE_F32, 1);
    for (auto type : { GGML_TYPE_PQ2_0, GGML_TYPE_PTQ1_0 }) {
        // Include several 64-row repacking chunks and an incomplete last chunk
        // to exercise both multiple workers and uneven work distribution.
        for (int rows : { 8, 16, 24, 40, 72, 264 }) {
            for (int tokens : { 1, 2, 3 }) {
                test_ternary_matmul(type, rows, tokens);
            }
        }
    }
    std::puts("ternary matmul: decode, batches and row tails OK");
    for (int block_size : { 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048 }) {
        test_hadamard(GGML_TYPE_F32, block_size);
        test_hadamard(GGML_TYPE_F16, block_size);
        if (block_size >= 32) test_hadamard(GGML_TYPE_Q8_0, block_size);
    }
    std::puts("Hadamard: scalar equality, dequantization and worker tails OK");
    return 0;
}
