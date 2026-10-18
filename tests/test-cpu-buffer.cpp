#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::abort();
    }
}

int main() {
    auto * buft = ggml_backend_cpu_buffer_type();
    constexpr size_t page = 2 * 1024 * 1024;
    // Exercise both sides of alignment/rounding boundaries on all platforms.
    const size_t sizes[] = { 0, 1, 33, page - 33, page - 32, page, page + 1, 3 * page + 17 };
    for (size_t size : sizes) {
        auto * buffer = ggml_backend_buft_alloc_buffer(buft, size);
        check(buffer != nullptr, "CPU allocation failed");
        auto * data = static_cast<unsigned char *>(ggml_backend_buffer_get_base(buffer));
        check(data != nullptr, "missing CPU buffer base");
        check(reinterpret_cast<uintptr_t>(data) % ggml_backend_buft_get_alignment(buft) == 0,
                "CPU buffer is not aligned");
        check(ggml_backend_buffer_get_size(buffer) >= size, "CPU buffer is too small");
        std::memset(data, 0x3c, size);
        ggml_backend_buffer_clear(buffer, 0xa5);
        for (size_t i = 0; i < size; ++i) {
            check(data[i] == 0xa5, "CPU buffer clear missed allocated bytes");
        }
        if (size) {
            data[0] = 0x12;
            data[size - 1] = 0x34;
            check(data[size - 1] == 0x34, "last requested byte is not writable");
        }
        ggml_backend_buffer_free(buffer);
    }
    check(ggml_backend_buft_alloc_buffer(buft, SIZE_MAX) == nullptr,
            "overflowing CPU allocation was accepted");
    check(ggml_backend_buft_alloc_buffer(buft, SIZE_MAX - 1) == nullptr,
            "overflowing alignment padding was accepted");
    std::puts("CPU buffers: allocation, alignment, clear, boundaries and overflow OK");
}
