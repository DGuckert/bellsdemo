#include "ggml.h"

#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
    for (int bits = 1; bits <= 8; ++bits) {
        const ggml_type type = (ggml_type) ((int) GGML_TYPE_EXL3_1 + bits - 1);

        assert(ggml_is_exl3(type));
        assert(ggml_exl3_n_bits(type) == bits);
        assert(ggml_blck_size(type) == 256);
        assert(ggml_type_size(type) == (size_t) (32 * bits));

        char expected[16];
        std::snprintf(expected, sizeof(expected), "exl3_%d", bits);
        assert(std::strcmp(ggml_type_name(type), expected) == 0);
    }

    assert(!ggml_is_exl3(GGML_TYPE_F32));
    assert(!ggml_is_exl3(GGML_TYPE_Q4_0));
    assert(!ggml_is_exl3(GGML_TYPE_Q2_0));

    std::printf("EXL3 type registration OK\n");
    return 0;
}
