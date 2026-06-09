#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <type_traits>
#include <sstream>
#include "ggnpu/graph.h"
#include "ggnpu/backend.h"
#include "ggnpu/tensor.h"

namespace {

template<typename T>
typename std::enable_if<std::is_pointer<T>::value, std::string>::type
to_str(const T& v) {
    std::ostringstream oss;
    oss << v;
    return oss.str();
}

template<typename T>
typename std::enable_if<std::is_enum<T>::value, std::string>::type
to_str(const T& v) { return std::to_string(static_cast<int>(v)); }

template<typename T>
typename std::enable_if<!std::is_pointer<T>::value && !std::is_enum<T>::value && !std::is_same<T, std::string>::value, std::string>::type
to_str(const T& v) { return std::to_string(v); }

std::string to_str(const std::string& v) { return "\"" + v + "\""; }

std::string to_str(const char* v) { return std::string(v); }

}

int tests_passed = 0;
int tests_failed = 0;

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { \
        std::cerr << "  FAIL: " << msg << " (expected " << to_str(b) << ", got " << to_str(a) << ")\n"; \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "  FAIL: " << msg << "\n"; \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps, msg) do { \
    if (std::fabs((a) - (b)) > (eps)) { \
        std::cerr << "  FAIL: " << msg << " (expected ~" << (b) << ", got " << (a) << ")\n"; \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

void test_graph_creation() {
    std::cout << "  test_graph_creation\n";

    ggnpu::ComputeGraph graph;
    ASSERT_TRUE(graph.nodes().empty(), "New graph has no nodes");

    auto node = graph.add_node(ggnpu::OpType::MUL_MAT_Q, "test_matmul");
    ASSERT_TRUE(node != nullptr, "add_node returns non-null");
    ASSERT_EQ(graph.nodes().size(), 1, "Graph has 1 node after add");
    ASSERT_EQ(node->type, ggnpu::OpType::MUL_MAT_Q, "Node type is MUL_MAT_Q");
    ASSERT_EQ(node->name, "test_matmul", "Node name is test_matmul");
}

void test_graph_node_parameters() {
    std::cout << "  test_graph_node_parameters\n";

    ggnpu::ComputeGraph graph;
    auto node = graph.add_node(ggnpu::OpType::MUL_MAT_Q, "matmul_256x256");

    node->M = 256;
    node->N = 256;
    node->K = 256;
    node->lda = 256;
    node->ldb = 256;
    node->ldc = 256;
    node->n_batches = 1;
    node->B_type = ggnpu::GgmlType::Q4_0;

    ASSERT_EQ(node->M, 256, "M = 256");
    ASSERT_EQ(node->N, 256, "N = 256");
    ASSERT_EQ(node->K, 256, "K = 256");
    ASSERT_EQ(node->B_type, ggnpu::GgmlType::Q4_0, "B_type is Q4_0");
}

void test_graph_connect() {
    std::cout << "  test_graph_connect\n";

    ggnpu::ComputeGraph graph;
    auto node_a = graph.add_node(ggnpu::OpType::MUL_MAT_Q, "matmul_a");
    auto node_b = graph.add_node(ggnpu::OpType::RMS_NORM, "rmsnorm_b");

    graph.connect(node_a, node_b);

    ASSERT_EQ(node_a->output.get(), node_b.get(), "node_a output points to node_b");
    ASSERT_EQ(node_b->inputs.size(), 1, "node_b has 1 input");
    ASSERT_EQ(node_b->inputs[0].get(), node_a.get(), "node_b input[0] is node_a");
}

void test_graph_multiple_inputs() {
    std::cout << "  test_graph_multiple_inputs\n";

    ggnpu::ComputeGraph graph;
    auto node_a = graph.add_node(ggnpu::OpType::MUL_MAT_Q, "matmul_a");
    auto node_b = graph.add_node(ggnpu::OpType::MUL_MAT_Q, "matmul_b");
    auto node_c = graph.add_node(ggnpu::OpType::ADD, "add_c");

    graph.connect(node_a, node_c);
    graph.connect(node_b, node_c);

    ASSERT_EQ(node_c->inputs.size(), 2, "node_c has 2 inputs");
    ASSERT_TRUE(node_c->inputs[0] == node_a, "node_c input[0] is node_a");
    ASSERT_TRUE(node_c->inputs[1] == node_b, "node_c input[1] is node_b");
}

void test_graph_with_cpu_backend() {
    std::cout << "  test_graph_with_cpu_backend\n";

    ggnpu::ComputeGraph graph;
    auto backend = ggnpu::create_cpu_ref_backend();
    graph.set_backend(backend);

    std::vector<float> input(4, 2.0f);
    std::vector<float> output(4, 0.0f);

    auto input_node = graph.add_node(ggnpu::OpType::VIEW, "input_view");
    input_node->cpu_buffer = input.data();
    input_node->size = 4;

    auto rms_node = graph.add_node(ggnpu::OpType::RMS_NORM, "rmsnorm");
    rms_node->cpu_buffer = output.data();
    rms_node->size = 4;
    rms_node->eps = 1e-5f;

    graph.connect(input_node, rms_node);

    ggnpu::Status st = graph.compile();
    ASSERT_EQ(st, ggnpu::Status::OK, "Graph compiles successfully");

    st = graph.execute();
    ASSERT_EQ(st, ggnpu::Status::OK, "Graph executes successfully");

    ASSERT_NEAR(output[0], 1.0f, 0.01f, "RMSNorm output ≈ 1.0 for input=2.0");
    ASSERT_NEAR(output[1], 1.0f, 0.01f, "RMSNorm output ≈ 1.0 for input=2.0");
}

void test_graph_topological_sort() {
    std::cout << "  test_graph_topological_sort\n";

    ggnpu::ComputeGraph graph;
    auto backend = ggnpu::create_cpu_ref_backend();
    graph.set_backend(backend);

    auto c = graph.add_node(ggnpu::OpType::VIEW, "c");
    auto b = graph.add_node(ggnpu::OpType::VIEW, "b");
    auto a = graph.add_node(ggnpu::OpType::VIEW, "a");

    graph.connect(a, b);
    graph.connect(b, c);

    ggnpu::Status st = graph.compile();
    ASSERT_EQ(st, ggnpu::Status::OK, "Graph with dependencies compiles");

    st = graph.execute();
    ASSERT_EQ(st, ggnpu::Status::OK, "Graph with dependencies executes");

    const auto& nodes = graph.nodes();
    ASSERT_EQ(nodes.size(), 3, "Graph has 3 nodes");
}

void test_graph_reset() {
    std::cout << "  test_graph_reset\n";

    ggnpu::ComputeGraph graph;
    graph.add_node(ggnpu::OpType::MUL_MAT_Q, "node1");
    graph.add_node(ggnpu::OpType::RMS_NORM, "node2");
    ASSERT_EQ(graph.nodes().size(), 2, "Graph has 2 nodes");

    graph.reset();
    ASSERT_TRUE(graph.nodes().empty(), "Graph is empty after reset");
}

void test_graph_all_op_types() {
    std::cout << "  test_graph_all_op_types\n";

    std::vector<ggnpu::OpType> all_ops = {
        ggnpu::OpType::MUL_MAT_Q,
        ggnpu::OpType::RMS_NORM,
        ggnpu::OpType::ROPE,
        ggnpu::OpType::SOFTMAX,
        ggnpu::OpType::ADD,
        ggnpu::OpType::MUL,
        ggnpu::OpType::SILU,
        ggnpu::OpType::FLASH_ATTN,
        ggnpu::OpType::COPY,
        ggnpu::OpType::VIEW,
    };

    ggnpu::ComputeGraph graph;
    for (size_t i = 0; i < all_ops.size(); i++) {
        auto node = graph.add_node(all_ops[i], "op_" + std::to_string(i));
        ASSERT_TRUE(node != nullptr, "Can create node for all op types");
    }

    ASSERT_EQ(graph.nodes().size(), all_ops.size(), "Created nodes for all op types");
}

void test_graph_status_values() {
    std::cout << "  test_graph_status_values\n";

    ASSERT_EQ(static_cast<int>(ggnpu::Status::OK), 0, "Status::OK = 0");
    ASSERT_EQ(static_cast<int>(ggnpu::Status::ERROR), 1, "Status::ERROR = 1");
    ASSERT_EQ(static_cast<int>(ggnpu::Status::NOT_FOUND), 2, "Status::NOT_FOUND = 2");
    ASSERT_EQ(static_cast<int>(ggnpu::Status::INVALID_PARAM), 3, "Status::INVALID_PARAM = 3");
    ASSERT_EQ(static_cast<int>(ggnpu::Status::NPU_UNAVAILABLE), 4, "Status::NPU_UNAVAILABLE = 4");
    ASSERT_EQ(static_cast<int>(ggnpu::Status::OUT_OF_MEMORY), 5, "Status::OUT_OF_MEMORY = 5");
}

void test_add_op() {
    std::cout << "  test_add_op\n";

    ggnpu::ComputeGraph graph;
    auto backend = ggnpu::create_cpu_ref_backend();
    graph.set_backend(backend);

    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};
    std::vector<float> result(3, 0.0f);

    auto a_node = graph.add_node(ggnpu::OpType::VIEW, "a");
    a_node->cpu_buffer = a.data();
    a_node->size = 3;

    auto b_node = graph.add_node(ggnpu::OpType::VIEW, "b");
    b_node->cpu_buffer = b.data();
    b_node->size = 3;

    auto add_node = graph.add_node(ggnpu::OpType::ADD, "add");
    add_node->cpu_buffer = result.data();
    add_node->size = 3;

    graph.connect(a_node, add_node);
    graph.connect(b_node, add_node);

    graph.compile();
    graph.execute();

    ASSERT_NEAR(result[0], 5.0f, 0.001f, "1+4=5");
    ASSERT_NEAR(result[1], 7.0f, 0.001f, "2+5=7");
    ASSERT_NEAR(result[2], 9.0f, 0.001f, "3+6=9");
}

void test_mul_op() {
    std::cout << "  test_mul_op\n";

    ggnpu::ComputeGraph graph;
    auto backend = ggnpu::create_cpu_ref_backend();
    graph.set_backend(backend);

    std::vector<float> a = {2.0f, 3.0f, 4.0f};
    std::vector<float> b = {5.0f, 6.0f, 7.0f};
    std::vector<float> result(3, 0.0f);

    auto a_node = graph.add_node(ggnpu::OpType::VIEW, "a");
    a_node->cpu_buffer = a.data();
    a_node->size = 3;

    auto b_node = graph.add_node(ggnpu::OpType::VIEW, "b");
    b_node->cpu_buffer = b.data();
    b_node->size = 3;

    auto mul_node = graph.add_node(ggnpu::OpType::MUL, "mul");
    mul_node->cpu_buffer = result.data();
    mul_node->size = 3;

    graph.connect(a_node, mul_node);
    graph.connect(b_node, mul_node);

    graph.compile();
    graph.execute();

    ASSERT_NEAR(result[0], 10.0f, 0.001f, "2*5=10");
    ASSERT_NEAR(result[1], 18.0f, 0.001f, "3*6=18");
    ASSERT_NEAR(result[2], 28.0f, 0.001f, "4*7=28");
}

void test_copy_op() {
    std::cout << "  test_copy_op\n";

    ggnpu::ComputeGraph graph;
    auto backend = ggnpu::create_cpu_ref_backend();
    graph.set_backend(backend);

    std::vector<float> src = {1.5f, 2.5f, 3.5f};
    std::vector<float> dst(3, 0.0f);

    auto src_node = graph.add_node(ggnpu::OpType::VIEW, "src");
    src_node->cpu_buffer = src.data();
    src_node->size = 3;

    auto copy_node = graph.add_node(ggnpu::OpType::COPY, "copy");
    copy_node->cpu_buffer = dst.data();
    copy_node->size = 3;

    graph.connect(src_node, copy_node);

    graph.compile();
    graph.execute();

    ASSERT_NEAR(dst[0], 1.5f, 0.001f, "copy[0]");
    ASSERT_NEAR(dst[1], 2.5f, 0.001f, "copy[1]");
    ASSERT_NEAR(dst[2], 3.5f, 0.001f, "copy[2]");
}

void run_tests() {
    std::cout << "=== Graph Tests ===\n\n";

    test_graph_creation();
    test_graph_node_parameters();
    test_graph_connect();
    test_graph_multiple_inputs();
    test_graph_with_cpu_backend();
    test_graph_topological_sort();
    test_graph_reset();
    test_graph_all_op_types();
    test_graph_status_values();
    test_add_op();
    test_mul_op();
    test_copy_op();

    std::cout << "\n--- Results ---\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";

    if (tests_failed > 0) {
        std::cout << "\nSOME TESTS FAILED\n";
    } else {
        std::cout << "\nALL TESTS PASSED\n";
    }
}

int main() {
    run_tests();
    return tests_failed > 0 ? 1 : 0;
}
