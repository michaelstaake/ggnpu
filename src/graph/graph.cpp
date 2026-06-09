#include "graph.h"
#include "backend.h"
#include <algorithm>
#include <iostream>

namespace ggnpu {

ComputeGraph::ComputeGraph() = default;
ComputeGraph::~ComputeGraph() = default;

void ComputeGraph::set_backend(std::shared_ptr<Backend> backend) {
    backend_ = backend;
    compiled_ = false;
}

Status ComputeGraph::compile() {
    if (!backend_) {
        return Status::ERROR;
    }
    compiled_ = true;
    return Status::OK;
}

Status ComputeGraph::execute() {
    if (!compiled_) {
        return Status::ERROR;
    }

    for (auto& node : nodes_) {
        Status status = Status::OK;

        switch (node->type) {
        case OpType::MUL_MAT_Q:
            if (node->cpu_buffer) {
                MulMatParams params;
                params.A = node->cpu_buffer;
                params.B = node->inputs.empty() ? nullptr : node->inputs[0]->cpu_buffer;
                params.C = node->output ? node->output->cpu_buffer : nullptr;
                params.M = node->M;
                params.N = node->N;
                params.K = node->K;
                params.lda = node->lda;
                params.ldb = node->ldb;
                params.ldc = node->ldc;
                params.n_batches = node->n_batches;
                params.B_type = node->B_type;
                status = backend_->mul_mat_q(params);
            }
            break;

        case OpType::RMS_NORM:
            if (node->cpu_buffer && node->inputs.size() > 0) {
                RmsNormParams params;
                params.input = static_cast<const float*>(node->inputs[0]->cpu_buffer);
                params.output = static_cast<float*>(node->cpu_buffer);
                params.size = node->size;
                params.eps = node->eps;
                status = backend_->rms_norm(params);
            }
            break;

        case OpType::ROPE:
            if (node->cpu_buffer) {
                RopeParams params;
                params.data = static_cast<float*>(node->cpu_buffer);
                params.n_dims = node->size;
                params.offset = node->offset;
                params.freq_scale = node->freq_scale;
                params.freq_base = node->freq_base;
                params.rope_dims = node->rope_dims;
                status = backend_->rope(params);
            }
            break;

        case OpType::SOFTMAX:
            if (node->cpu_buffer && node->inputs.size() > 0) {
                SoftmaxParams params;
                params.input = static_cast<const float*>(node->inputs[0]->cpu_buffer);
                params.output = static_cast<float*>(node->cpu_buffer);
                params.rows = node->rows;
                params.cols = node->cols;
                status = backend_->softmax(params);
            }
            break;

        case OpType::SILU:
            if (node->cpu_buffer && node->inputs.size() > 0) {
                SiluParams params;
                params.input = static_cast<const float*>(node->inputs[0]->cpu_buffer);
                params.output = static_cast<float*>(node->cpu_buffer);
                params.size = node->size;
                status = backend_->silu(params);
            }
            break;

        case OpType::FLASH_ATTN:
            if (node->cpu_buffer && node->inputs.size() >= 3) {
                AttnParams params;
                params.Q = static_cast<const float*>(node->inputs[0]->cpu_buffer);
                params.K = static_cast<const float*>(node->inputs[1]->cpu_buffer);
                params.V = static_cast<const float*>(node->inputs[2]->cpu_buffer);
                params.output = static_cast<float*>(node->cpu_buffer);
                params.batch_size = node->batch_size;
                params.n_head = node->n_head;
                params.head_dim = node->head_dim;
                params.ctx_len = node->ctx_len;
                params.freq_factors = node->freq_factors;
                status = backend_->flash_attn(params);
            }
            break;

        case OpType::ADD:
        case OpType::MUL:
        case OpType::COPY:
        case OpType::VIEW:
            break;

        default:
            break;
        }

        if (status != Status::OK) {
            return status;
        }
    }

    if (backend_) {
        backend_->sync();
    }
    return Status::OK;
}

void ComputeGraph::reset() {
    nodes_.clear();
    compiled_ = false;
}

std::shared_ptr<OpNode> ComputeGraph::add_node(OpType type, const std::string& name) {
    auto node = std::make_shared<OpNode>();
    node->type = type;
    node->name = name;
    nodes_.push_back(node);
    return node;
}

void ComputeGraph::connect(std::shared_ptr<OpNode> from, std::shared_ptr<OpNode> to) {
    from->output = to;
    to->inputs.push_back(from);
}

} // namespace ggnpu
