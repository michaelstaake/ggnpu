cmake_minimum_required(VERSION 3.20)
project(ggnpu VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CMake module path for FindXRT
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Options
option(GGNPU_BUILD_TESTS "Build unit tests" ON)
option(GGNPU_NPU_BACKEND "Enable AMD NPU backend (requires XRT)" OFF)
option(GGNPU_TEST_CPU "Enable CPU reference backend for testing" ON)
option(GGNPU_BUILD_DOCS "Build documentation" OFF)
option(GGNPU_BUILD_KERNELS "Build NPU kernels at build time (requires Triton-XDNA)" OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Threading
find_package(Threads REQUIRED)

# XRT (optional, for NPU backend)
if(GGNPU_NPU_BACKEND)
    find_package(XRT)
    if(NOT XRT_FOUND)
        message(WARNING "XRT not found. NPU backend will be disabled.")
        message(WARNING "  Install: sudo apt install libxrt2 libxrt-npu2 libxrt-dev")
        set(GGNPU_NPU_BACKEND OFF)
    else()
        message(STATUS "XRT found: ${XRT_LIBRARIES} (headers: ${XRT_INCLUDE_DIRS})")
    endif()
endif()

# Check for Triton-XDNA (for kernel building)
set(TRITON_XDNA_FOUND OFF)
if(GGNPU_BUILD_KERNELS)
    find_program(PYTHON3_BIN python3)
    if(PYTHON3_BIN)
        execute_process(
            COMMAND ${PYTHON3_BIN} -c "from triton.backends.amd_triton_npu.driver import NPUDriver; print('ok')"
            RESULT_VARIABLE TRITON_RESULT
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(TRITON_RESULT EQUAL 0)
            set(TRITON_XDNA_FOUND ON)
            message(STATUS "Triton-XDNA found: ${PYTHON3_BIN}")
        else()
            message(WARNING "python3 found but triton module not available")
            message(WARNING "  Install Triton-XDNA: pip install triton-xdna")
        endif()
    endif()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Library source files
set(LIB_SOURCES
    src/gguf/gguf.cpp
    src/gguf/tensor.cpp
    src/quant/q4_0.cpp
    src/quant/q8_0.cpp
    src/quant/q4_k.cpp
    src/quant/q6_k.cpp
    src/quant/quant.cpp
    src/arch/llama/llama.cpp
    src/graph/graph.cpp
    src/runtime/kv_cache.cpp
    src/cache/cache.cpp
    src/backends/cpu_ref/cpu_ref.cpp
    src/cli/tokenizer.cpp
    src/cli/unicode.cpp
    src/cli/unicode-data.cpp
)

if(GGNPU_NPU_BACKEND)
    list(APPEND LIB_SOURCES
        src/backends/amd_xdna/amd_xdna.cpp
        src/backends/amd_xdna/buffer.cpp
        src/backends/amd_xdna/kernels.cpp
    )
endif()

add_library(ggnpu_lib STATIC ${LIB_SOURCES})
target_include_directories(ggnpu_lib PUBLIC ${CMAKE_SOURCE_DIR}/include/ggnpu)
target_link_libraries(ggnpu_lib PRIVATE Threads::Threads)

if(GGNPU_NPU_BACKEND)
    target_link_libraries(ggnpu_lib PRIVATE XRT::XRT dl pthread rt)
    target_compile_definitions(ggnpu_lib PRIVATE GGNPU_HAS_NPU_BACKEND=1)
endif()

if(GGNPU_TEST_CPU)
    target_compile_definitions(ggnpu_lib PRIVATE GGNPU_TEST_CPU=1)
endif()

# Main binary
add_executable(ggnpu src/cli/main.cpp)
target_link_libraries(ggnpu PRIVATE ggnpu_lib Threads::Threads)

if(GGNPU_NPU_BACKEND)
    target_compile_definitions(ggnpu PRIVATE GGNPU_HAS_NPU_BACKEND=1)
endif()

if(GGNPU_TEST_CPU)
    target_compile_definitions(ggnpu PRIVATE GGNPU_TEST_CPU=1)
endif()

# Tests
if(GGNPU_BUILD_TESTS)
    enable_testing()

    set(TEST_FILES
        test_gguf
        test_quant
        test_graph
        test_arch
        test_tokenizer
    )

    foreach(TEST_NAME ${TEST_FILES})
        add_executable(${TEST_NAME} tests/${TEST_NAME}.cpp)
        target_link_libraries(${TEST_NAME} PRIVATE ggnpu_lib Threads::Threads)
        target_compile_definitions(${TEST_NAME} PRIVATE GGNPU_TEST_CPU=1)
        if(TEST_NAME STREQUAL "test_tokenizer")
            add_test(NAME ${TEST_NAME}
                COMMAND ${TEST_NAME} ${CMAKE_SOURCE_DIR}/models/llama-3.2-1b-q4_k_m.gguf)
        else()
            add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
        endif()
    endforeach()

    # E2E: France prompt → greedy top-1 is Paris (requires local model.gguf)
    add_test(
        NAME test_e2e_logits
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/test-e2e-logits.sh
                $<TARGET_FILE:ggnpu>
                ${CMAKE_SOURCE_DIR}/models/llama-3.2-1b-q4_k_m.gguf)
    set_tests_properties(test_e2e_logits PROPERTIES
        SKIP_RETURN_CODE 77
        LABELS "e2e")
endif()

# Install
install(TARGETS ggnpu DESTINATION bin)

#====//
# NPU Kernel Building (optional, requires Triton-XDNA)
# Uses scripts/build-kernels.sh to compile all kernel types via Triton-XDNA
#====//
if(GGNPU_BUILD_KERNELS)
    if(TRITON_XDNA_FOUND)
        message(STATUS "NPU kernel building enabled (via Triton-XDNA)")

        # Create kernel output directory
        set(KERNEL_OUTPUT_DIR "${CMAKE_BINARY_DIR}/xclbin")
        file(MAKE_DIRECTORY "${KERNEL_OUTPUT_DIR}")

        # Build all kernels using the shell script
        find_program(BASH_BIN bash)
        add_custom_target(build_npu_kernels
            COMMAND ${BASH_BIN} ${CMAKE_SOURCE_DIR}/scripts/build-kernels.sh
            COMMENT "Building all NPU kernels (matmul, rmsnorm, rope, softmax, silu, flash_attn) via Triton-XDNA"
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            VERBATIM
        )

        # Install kernels to prefix
        install(DIRECTORY "${KERNEL_OUTPUT_DIR}/"
            DESTINATION share/ggnpu/xclbin
            FILES_MATCHING PATTERN "*.xclbin"
        )

    else()
        message(STATUS "NPU kernel building disabled (Triton-XDNA not found)")
        message(STATUS "  Install Triton-XDNA: pip install triton-xdna")
        message(STATUS "  Or use prebuilt xclbins in ~/.cache/ggnpu/xclbin/")
    endif()
endif()

# Print build summary
message(STATUS "")
message(STATUS "=== GGNPU Build Configuration ===")
message(STATUS "  NPU backend:        ${GGNPU_NPU_BACKEND}")
message(STATUS "  CPU ref backend:    ${GGNPU_TEST_CPU}")
message(STATUS "  Tests:              ${GGNPU_BUILD_TESTS}")
message(STATUS "  Kernel building:    ${GGNPU_BUILD_KERNELS}")
if(TRITON_XDNA_FOUND)
    message(STATUS "  Triton-XDNA:       available")
else()
    message(STATUS "  Triton-XDNA:       not available (requires pip install triton-xdna)")
endif()
message(STATUS "=================================")
message(STATUS "")
