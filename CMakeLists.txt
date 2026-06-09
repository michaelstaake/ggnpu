cmake_minimum_required(VERSION 3.20)
project(ggnpu VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(GGNPU_BUILD_TESTS "Build unit tests" ON)
option(GGNPU_NPU_BACKEND "Enable AMD NPU backend (requires XRT)" OFF)
option(GGNPU_TEST_CPU "Enable CPU reference backend for testing" ON)
option(GGNPU_BUILD_DOCS "Build documentation" OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Threading
find_package(Threads REQUIRED)

# XRT (optional, for NPU backend)
set(XRT_INCLUDE_DIRS "" CACHE PATH "XRT include directory")
set(XRT_LIBRARIES "" CACHE FILEPATH "XRT library")

if(GGNPU_NPU_BACKEND)
    if(XRT_INCLUDE_DIRS AND XRT_LIBRARIES)
        message(STATUS "XRT found: ${XRT_LIBRARIES}")
    else()
        find_path(XRT_INCLUDE_DIRS xrt.h
            PATHS /opt/xilinx/include /usr/include
            PATH_SUFFIXES xrt
        )
        find_library(XRT_LIBRARIES xrt_coreutil
            PATHS /opt/xilinx/lib /usr/lib
        )
        if(NOT XRT_INCLUDE_DIRS OR NOT XRT_LIBRARIES)
            message(WARNING "XRT not found. NPU backend will be disabled. Set GGNPU_NPU_BACKEND=OFF.")
            set(GGNPU_NPU_BACKEND OFF)
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
    src/arch/llama/llama.cpp
    src/graph/graph.cpp
    src/graph/ops.cpp
    src/runtime/scheduler.cpp
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
    target_include_directories(ggnpu_lib PUBLIC ${XRT_INCLUDE_DIRS})
    target_link_libraries(ggnpu_lib PRIVATE ${XRT_LIBRARIES})
    target_compile_definitions(ggnpu_lib PRIVATE GGNPU_HAS_NPU_BACKEND=1)
endif()

if(GGNPU_TEST_CPU)
    target_compile_definitions(ggnpu_lib PRIVATE GGNPU_TEST_CPU=1)
endif()

# Main binary
add_executable(ggnpu src/cli/main.cpp)
target_link_libraries(ggnpu PRIVATE ggnpu_lib Threads::Threads)

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
