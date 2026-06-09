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

#====//
# NPU Kernel Building (optional, requires mlir-aie + Peano)
#====//
if(GGNPU_BUILD_KERNELS)
    find_package(Peano QUIET)

    # Check for aiecc.py
    set(AIECC_FOUND OFF)
    set(AIE_HOME "" CACHE PATH "mlir-aie build directory")

    if(AIE_HOME)
        if(EXISTS "${AIE_HOME}/bin/aiecc.py")
            set(AIECC_FOUND ON)
            message(STATUS "mlir-aie found: ${AIE_HOME}/bin/aiecc.py")
        else()
            message(WARNING "AIE_HOME set to ${AIE_HOME} but aiecc.py not found")
        endif()
    elseif(NOT Peano_FOUND)
        find_program(AIECC_BIN aiecc.py)
        if(AIECC_BIN)
            set(AIECC_FOUND ON)
            message(STATUS "mlir-aie (aiecc.py) found: ${AIECC_BIN}")
        endif()
    endif()

    if(AIECC_FOUND AND Peano_FOUND)
        message(STATUS "NPU kernel building enabled")

        # Define kernel shapes to compile (typical Llama 3B shapes)
        set(GGNPU_KERNEL_SHAPES
            "256x256x256"
            "512x512x512"
            "1024x1024x1024"
            "2048x2048x2048"
            "3072x3072x3072"
            "4096x4096x4096"
            "8192x8192x8192"
        )

        # Create kernel output directory
        set(KERNEL_OUTPUT_DIR "${CMAKE_BINARY_DIR}/xclbin")
        file(MAKE_DIRECTORY "${KERNEL_OUTPUT_DIR}")

        # Generate aie2p target for each shape
        foreach(SHAPE ${GGNPU_KERNEL_SHAPES})
            string(REPLACE "x" "_" SHAPE_NAME "${SHAPE}")
            set(MLIR_SRC "${CMAKE_SOURCE_DIR}/kernels/amd/matmul_i8/matmul.mlir")
            set(XCLBIN_OUT "${KERNEL_OUTPUT_DIR}/matmul_aie2p_${SHAPE_NAME}.xclbin")

            add_custom_command(
                OUTPUT "${XCLBIN_OUT}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${KERNEL_OUTPUT_DIR}"
                COMMAND ${AIECC_BIN}
                    --target=aie2p
                    --npu-profile=4
                    -I"${Peano_INCLUDE_DIRS}"
                    -L"${Peano_LIBRARIES}"
                    "${MLIR_SRC}"
                    -o "${XCLBIN_OUT}"
                DEPENDS "${MLIR_SRC}"
                COMMENT "Building NPU kernel: matmul ${SHAPE}"
                VERBATIM
            )

            list(APPEND GGNPU_KERNEL_TARGETS "${XCLBIN_OUT}")
        endforeach()

        add_custom_target(build_npu_kernels ALL
            DEPENDS ${GGNPU_KERNEL_TARGETS}
        )

        # Install kernels to prefix
        install(DIRECTORY "${KERNEL_OUTPUT_DIR}/"
            DESTINATION share/ggnpu/xclbin
            FILES_MATCHING PATTERN "*.xclbin"
        )

    elseif(AIECC_FOUND)
        message(WARNING "aiecc.py found but Peano not found. Kernel building may fail.")
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
