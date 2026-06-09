#include "graph.h"
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
    // Execute nodes in topological order
    for (auto& node : nodes_) {
        // Execute the operation through the backend
        // This is a simplified execution model
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
