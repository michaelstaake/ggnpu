#ifndef GGNPU_GRAPH_H
#define GGNPU_GRAPH_H

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include "backend.h"
#include "tensor.h"

namespace ggnpu {

enum class OpType {
    MUL_MAT_Q,
    RMS_NORM,
    ROPE,
    SOFTMAX,
    ADD,
    MUL,
    SILU,
    FLASH_ATTN,
    COPY,
    VIEW,
};

struct OpNode {
    OpType type;
    std::string name;
    std::vector<std::shared_ptr<OpNode>> inputs;
    std::shared_ptr<OpNode> output;
    void* cpu_buffer = nullptr;
    bool needs_sync = false;

    // Parameters for operations
    int M = 0, N = 0, K = 0;
    int lda = 0, ldb = 0, ldc = 0;
    int32_t n_batches = 1;
    GgmlType B_type = GgmlType::F32;
    int size = 0;
    float eps = 0.0f;
    int64_t offset = 0;
    float freq_scale = 1.0f;
    float freq_base = 10000.0f;
    int64_t rope_dims = 0;
    int rows = 0, cols = 0;
    int batch_size = 0;
    int n_head = 0;
    int head_dim = 0;
    int64_t ctx_len = 0;
    const float* freq_factors = nullptr;
};

class ComputeGraph {
public:
    ComputeGraph();
    ~ComputeGraph();

    void set_backend(std::shared_ptr<Backend> backend);
    Status compile();
    Status execute();
    void reset();

    std::shared_ptr<OpNode> add_node(OpType type, const std::string& name);
    void connect(std::shared_ptr<OpNode> from, std::shared_ptr<OpNode> to);

    const std::vector<std::shared_ptr<OpNode>>& nodes() const { return nodes_; }

private:
    std::vector<std::shared_ptr<OpNode>> nodes_;
    std::shared_ptr<Backend> backend_;
    bool compiled_ = false;
};

} // namespace ggnpu

#endif // GGNPU_GRAPH_H
