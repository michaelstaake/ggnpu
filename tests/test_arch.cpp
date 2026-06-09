#include <iostream>
#include <string>

#include "model.h"
#include "gguf.h"

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

void test_model_creation() {
    std::cout << "\n--- Model Creation ---\n";

    ggnpu::Model model;
    assert_true(!model.is_loaded(), "New model is not loaded");
}

void test_model_unload() {
    std::cout << "\n--- Model Unload ---\n";

    ggnpu::Model model;
    model.unload();  // Should not crash
    assert_true(!model.is_loaded(), "Model is not loaded after unload");
}

void run_tests(const std::string& filter) {
    std::cout << "Running ggnpu tests...\n";

    test_model_creation();
    test_model_unload();

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
