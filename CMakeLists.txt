cmake_minimum_required(VERSION 3.20)
project(ggnpu VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CMake module path for FindPeano, FindXRT
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Options
option(GGNPU_BUILD_TESTS "Build unit tests" ON)
option(GGNPU_NPU_BACKEND "Enable AMD NPU backend (requires XRT)" OFF)
option(GGNPU_TEST_CPU "Enable CPU reference backend for testing" ON)
option(GGNPU_BUILD_DOCS "Build documentation" OFF)
option(GGNPU_BUILD_KERNELS "Build NPU kernels at build time (requires mlir-aie)" OFF)

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
    )

    foreach(TEST_NAME ${TEST_FILES})
        add_executable(${TEST_NAME} tests/${TEST_NAME}.cpp)
        target_link_libraries(${TEST_NAME} PRIVATE ggnpu_lib Threads::Threads)
        target_compile_definitions(${TEST_NAME} PRIVATE GGNPU_TEST_CPU=1)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
endif()

# Install
install(TARGETS ggnpu DESTINATION bin)

#====//
# NPU Kernel Building (optional, requires mlir-aie + Peano)
# Uses scripts/build-kernels.sh to compile all kernel types
#====//
if(GGNPU_BUILD_KERNELS)
    set(AIECC_FOUND OFF)
    set(AIE_HOME "" CACHE PATH "mlir-aie build directory")

    if(AIE_HOME)
        if(EXISTS "${AIE_HOME}/bin/aiecc.py")
            set(AIECC_FOUND ON)
            message(STATUS "mlir-aie found: ${AIE_HOME}/bin/aiecc.py")
        else()
            message(WARNING "AIE_HOME set to ${AIE_HOME} but aiecc.py not found")
        endif()
    else()
        find_program(AIECC_BIN aiecc.py)
        if(AIECC_BIN)
            set(AIECC_FOUND ON)
            message(STATUS "mlir-aie (aiecc.py) found: ${AIECC_BIN}")
        endif()
    endif()

    if(AIECC_FOUND)
        message(STATUS "NPU kernel building enabled (via scripts/build-kernels.sh)")

        # Create kernel output directory
        set(KERNEL_OUTPUT_DIR "${CMAKE_BINARY_DIR}/xclbin")
        file(MAKE_DIRECTORY "${KERNEL_OUTPUT_DIR}")

        # Build all kernels using the shell script
        add_custom_target(build_npu_kernels
            COMMAND ${CMAKE_COMMAND} -E env
                AIE_HOME=${AIE_HOME}
                PEANO_HOME=${PEANO_HOME}
                GGNPU_CACHE_DIR=${KERNEL_OUTPUT_DIR}
            ${CMAKE_SOURCE_DIR}/scripts/build-kernels.sh
            COMMENT "Building all NPU kernels (matmul, rmsnorm, rope, softmax, silu, flash_attn)"
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            VERBATIM
        )

        # Install kernels to prefix
        install(DIRECTORY "${KERNEL_OUTPUT_DIR}/"
            DESTINATION share/ggnpu/xclbin
            FILES_MATCHING PATTERN "*.xclbin"
        )

    else()
        message(STATUS "NPU kernel building disabled (aiecc.py not found)")
        message(STATUS "  Install mlir-aie and set AIE_HOME, or enable prebuilt xclbins")
    endif()
endif()

# Print build summary
message(STATUS "")
message(STATUS "=== GGNPU Build Configuration ===")
message(STATUS "  NPU backend:        ${GGNPU_NPU_BACKEND}")
message(STATUS "  CPU ref backend:    ${GGNPU_TEST_CPU}")
message(STATUS "  Tests:              ${GGNPU_BUILD_TESTS}")
message(STATUS "  Kernel building:    ${GGNPU_BUILD_KERNELS}")
if(AIECC_FOUND)
    message(STATUS "  JIT compilation:    available")
else()
    message(STATUS "  JIT compilation:    not available (requires mlir-aie)")
endif()
message(STATUS "=================================")
message(STATUS "")
