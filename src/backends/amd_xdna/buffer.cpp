// Buffer management for AMD XDNA NPU
// This file is compiled only when GGNPU_NPU_BACKEND=ON
// Buffer allocation and DMA are handled within amd_xdna.cpp
// using xrt::bo (XRT buffer objects) for pinned memory transfers.
//
// Buffer types:
//   - Host-backed buffers (xrt::bo::flags::host): pinned memory for DMA
//   - Device buffers: on-NPU memory for kernel execution
//
// The BufferMgr class in amd_xdna.cpp manages:
//   - Input A buffer (M×K floats)
//   - Input B buffer (K×N quantized weights)
//   - Output C buffer (M×N floats)
//
// DMA transfer flow:
//   1. Host → Device: buf_mgr_->copy_to(buf, data, size)
//   2. Kernel execution: run(buf_a, buf_b, buf_c, M, N, K)
//   3. Device → Host: buf_mgr_->copy_from(buf, data, size)
