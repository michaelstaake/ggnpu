# Intel NPU Roadmap

## Future Support

GGNPU targets Intel NPUs behind the same `Backend` interface.

## Target Platform

- **CPU**: Intel Core Ultra (Panther Lake)
- **NPU**: Intel AI Boost (Movidius VPU)
- **OS**: Ubuntu 26.04 LTS, Linux 7.0+

## Implementation Plan

1. Research Intel NPU driver interface (similar to amdxdna)
2. Implement `IntelNpuBackend` class
3. Add Intel NPU kernel compilation pipeline
4. Test on Panther Lake hardware

## Interface

```cpp
// Same Backend interface as AMD
class IntelNpuBackend : public Backend {
    // mul_mat_q, rms_norm, rope, etc.
};
```
