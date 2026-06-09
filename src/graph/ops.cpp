#include "graph.h"
#include "backend.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <iostream>

namespace ggnpu {

// Simplified operations for the compute graph
// In production, these would dispatch to the NPU backend

namespace {

// CPU reference implementations for testing

void rms_norm_cpu(const float* input, float* output, int size, float eps) {
    float variance = 0.0f;
    for (int i = 0; i < size; i++) {
        variance += input[i] * input[i];
    }
    variance /= size;
    variance += eps;
    float inv_std = 1.0f / std::sqrt(variance);

    for (int i = 0; i < size; i++) {
        output[i] = input[i] * inv_std;
    }
}

void rope_cpu(float* data, int n_dims, int64_t offset, float freq_scale,
              float freq_base, int64_t rope_dims) {
    for (int64_t i = 0; i < rope_dims; i += 2) {
        float ratio = 1.0f / std::pow(freq_base, static_cast<float>(i) / n_dims);
        float val = offset * ratio;
        float ftheta = val * freq_scale;
        float cos_val = std::cos(ftheta);
        float sin_val = std::sin(ftheta);

        float v0 = data[i];
        float v1 = data[i + 1];
        data[i] = v0 * cos_val - v1 * sin_val;
        data[i + 1] = v0 * sin_val + v1 * cos_val;
    }
}

void softmax_cpu(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float max_val = -INFINITY;
        for (int c = 0; c < cols; c++) {
            if (input[r * cols + c] > max_val) {
                max_val = input[r * cols + c];
            }
        }

        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            output[r * cols + c] = std::exp(input[r * cols + c] - max_val);
            sum += output[r * cols + c];
        }

        for (int c = 0; c < cols; c++) {
            output[r * cols + c] /= sum;
        }
    }
}

void silu_cpu(const float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        float x = input[i];
        output[i] = x / (1.0f + std::exp(-x));
    }
}

} // namespace

} // namespace ggnpu
