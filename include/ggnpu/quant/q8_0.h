#ifndef GGNPU_Q8_0_H
#define GGNPU_Q8_0_H

#include <cstdint>
#include <vector>

namespace ggnpu {

struct Q8_0Block {
    int16_t d;
    int8_t qs[32];
};

void decode_q8_0_for_npu(const uint8_t* gguf_data, size_t data_size,
                         std::vector<int8_t>& int8_output,
                         std::vector<float>& scales_output);

} // namespace ggnpu

#endif // GGNPU_Q8_0_H
