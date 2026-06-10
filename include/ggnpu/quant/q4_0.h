#ifndef GGNPU_Q4_0_H
#define GGNPU_Q4_0_H

#include <cstdint>
#include <vector>

namespace ggnpu {

struct Q4_0Block {
    int16_t d;
    uint8_t qs[16];  // 16 bytes for 32 Q4_0 values (2 values per byte)
};

void decode_q4_0_for_npu(const uint8_t* gguf_data, size_t data_size,
                         std::vector<int8_t>& int8_output,
                         std::vector<float>& scales_output);

} // namespace ggnpu

#endif // GGNPU_Q4_0_H
