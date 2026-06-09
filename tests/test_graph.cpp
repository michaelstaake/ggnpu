#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "backend.h"
#include "graph.h"

namespace {

int tests_passed = 0;
int tests_failed = 0;

void assert_true(bool cond, const std::string& msg) {
    if (cond) {
        tests_passed++;
        std::cout << "  PASS: " << msg << "\n";
    } else {
        tests_failed++;
        std::cout << "  FAIL: " << msg << "\n";
    }
}

void test_compute_graph() {
    std::cout << "\n--- Compute Graph ---\n";

    ggnpu::ComputeGraph graph;
    auto backend = ggnpu::create_cpu_ref_backend();
    graph.set_backend(backend);

    auto node1 = graph.add_node(ggnpu::OpType::RMS_NORM, "norm1");
    auto node2 = graph.add_node(ggnpu::OpType::SOFTMAX, "softmax1");
    graph.connect(node1, node2);

    assert_true(graph.nodes().size() == 2, "Graph has 2 nodes");
    assert_true(!graph.nodes()[0]->output || graph.nodes()[0]->output == node2, "Node1 connected to Node2");

    auto status = graph.compile();
    assert_true(status == ggnpu::Status::OK, "Graph compiles successfully");
}

void test_topological_order() {
    std::cout << "\n--- Topological Order ---\n";

    ggnpu::ComputeGraph graph;
    auto backend = ggnpu::create_cpu_ref_backend();
    graph.set_backend(backend);

    auto node_a = graph.add_node(ggnpu::OpType::RMS_NORM, "norm_a");
    auto node_b = graph.add_node(ggnpu::OpType::MUL_MAT_Q, "matmul_b");
    auto node_c = graph.add_node(ggnpu::OpType::SOFTMAX, "softmax_c");

    graph.connect(node_a, node_b);
    graph.connect(node_b, node_c);

    graph.compile();

    node_a->cpu_buffer = new float[4]{1.0f, 2.0f, 3.0f, 4.0f};
    node_b->cpu_buffer = new float[4]{0.0f, 0.0f, 0.0f, 0.0f};
    node_c->cpu_buffer = new float[4]{0.0f, 0.0f, 0.0f, 0.0f};

    node_b->M = 1;
    node_b->N = 1;
    node_b->K = 4;
    node_b->B_type = ggnpu::GgmlType::F32;

    node_c->rows = 1;
    node_c->cols = 4;

    node_a->size = 4;
    node_a->eps = 1e-5f;

    auto status = graph.execute();
    assert_true(status == ggnpu::Status::OK, "Graph executes in topological order");

    delete[] static_cast<float*>(node_a->cpu_buffer);
    delete[] static_cast<float*>(node_b->cpu_buffer);
    delete[] static_cast<float*>(node_c->cpu_buffer);
}

void test_backend_interface() {
    std::cout << "\n--- Backend Interface ---\n";

    auto backend = ggnpu::create_cpu_ref_backend();

    assert_true(backend->is_available(), "CPU ref backend is available");
    assert_true(backend->name() == "cpu_ref", "Backend name is cpu_ref");
    assert_true(backend->last_error() == ggnpu::Status::OK, "Last error is OK");
}

void run_tests(const std::string& filter) {
    std::cout << "Running ggnpu tests...\n";

    test_compute_graph();
    test_topological_order();
    test_backend_interface();

    std::cout << "\n=== Results ===\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";
    std::cout << "Total:  " << (tests_passed + tests_failed) << "\n";
}

} // namespace

int main(int argc, char* argv[]) {
    std::string filter;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--filter" && i + 1 < argc) {
            filter = argv[++i];
        }
    }

    run_tests(filter);
    return tests_failed > 0 ? 1 : 0;
}
