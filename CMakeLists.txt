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
        # Try XRT setup script first (sets XRT_PREFIX)
        find_program(XRT_SETUP_SCRIPT setup.sh PATHS /opt/xilinx/xrt)
        if(XRT_SETUP_SCRIPT)
            execute_process(
                COMMAND bash -c "source ${XRT_SETUP_SCRIPT} 2>/dev/null && echo \$XRT_PREFIX"
                OUTPUT_VARIABLE XRT_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if(XRT_PREFIX)
                message(STATUS "XRT found via setup.sh: ${XRT_PREFIX}")
                set(XRT_INCLUDE_DIRS "${XRT_PREFIX}/include")
                set(XRT_LIB_SEARCH_PATHS "${XRT_PREFIX}/lib" "${XRT_PREFIX}/lib/x86_64-linux-gnu")
            endif()
        endif()

        if(NOT XRT_INCLUDE_DIRS)
            find_path(XRT_INCLUDE_DIRS xrt.h
                PATHS
                    /opt/xilinx/include
                    /opt/xilinx/include/xrt
                    /usr/include
                    /usr/include/xrt
                    ${XRT_LIB_SEARCH_PATHS}
                PATH_SUFFIXES
                    xrt
            )
        endif()

        if(NOT XRT_LIBRARIES)
            find_library(XRT_LIBRARIES NAMES xrt_core xrt_coreutil xilinxopencl
                PATHS
                    /opt/xilinx/lib
                    /opt/xilinx/lib/x86_64-linux-gnu
                    /usr/lib
                    /usr/lib/x86_64-linux-gnu
                    ${XRT_LIB_SEARCH_PATHS}
            )
        endif()

        if(NOT XRT_INCLUDE_DIRS OR NOT XRT_LIBRARIES)
            message(WARNING "XRT not found. NPU backend will be disabled. Set GGNPU_NPU_BACKEND=OFF.")
            message(WARNING "  XRT headers: ${XRT_INCLUDE_DIRS}")
            message(WARNING "  XRT libs: ${XRT_LIBRARIES}")
            set(GGNPU_NPU_BACKEND OFF)
        else()
            message(STATUS "XRT found: ${XRT_LIBRARIES} (headers: ${XRT_INCLUDE_DIRS})")
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
    target_link_libraries(ggnpu_lib PRIVATE ${XRT_LIBRARIES} dl pthread rt)
    target_compile_definitions(ggnpu_lib PRIVATE GGNPU_HAS_NPU_BACKEND=1)
    if(XRT_PREFIX)
        link_directories(${XRT_PREFIX}/lib ${XRT_PREFIX}/lib/x86_64-linux-gnu)
    endif()
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
