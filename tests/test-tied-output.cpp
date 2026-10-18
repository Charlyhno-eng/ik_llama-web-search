#include "llama-model.h"
#include "ggml-alloc.h"
#include "ggml-quants.h"
#include "iqk/iqk_mul_mat.h"
#include "iqk/iqk_quantize.h"

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

int main() {
    ggml_time_init();
    test_format(GGML_TYPE_PQ2_0, 8);
    test_format(GGML_TYPE_PQ2_0, 8, true);
    test_format(GGML_TYPE_PTQ1_0, 8);
    test_format(GGML_TYPE_Q4_0, 8);
    test_format(GGML_TYPE_Q6_K, 4);
    test_format(GGML_TYPE_F32, 1);
    for (auto type : { GGML_TYPE_PQ2_0, GGML_TYPE_PTQ1_0 }) {
        for (int rows : { 8, 16, 24, 40 }) {
            for (int tokens : { 1, 2, 3 }) {
                test_ternary_matmul(type, rows, tokens);
            }
        }
    }
    std::puts("ternary matmul: decode, batches and row tails OK");
    return 0;
}
