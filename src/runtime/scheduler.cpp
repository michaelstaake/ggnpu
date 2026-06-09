#include "graph.h"
#include "backend.h"
#include <vector>
#include <queue>
#include <map>
#include <functional>

namespace ggnpu {

// Scheduler for compute graph execution
// Manages topological ordering and dependency tracking

class Scheduler {
public:
    Scheduler() = default;

    void schedule(ComputeGraph& graph) {
        // Topological sort of graph nodes
        std::map<std::shared_ptr<OpNode>, int> in_degree;
        for (auto& node : graph.nodes()) {
            in_degree[node] = static_cast<int>(node->inputs.size());
        }

        std::queue<std::shared_ptr<OpNode>> ready;
        for (auto& [node, deg] : in_degree) {
            if (deg == 0) ready.push(node);
        }

        scheduled_order_.clear();
        while (!ready.empty()) {
            auto node = ready.front();
            ready.pop();
            scheduled_order_.push_back(node);

            // Find nodes that depend on this one
            for (auto& other : graph.nodes()) {
                for (auto& inp : other->inputs) {
                    if (inp == node) {
                        in_degree[other]--;
                        if (in_degree[other] == 0) {
                            ready.push(other);
                        }
                    }
                }
            }
        }
    }

    const std::vector<std::shared_ptr<OpNode>>& order() const {
        return scheduled_order_;
    }

private:
    std::vector<std::shared_ptr<OpNode>> scheduled_order_;
};

} // namespace ggnpu
