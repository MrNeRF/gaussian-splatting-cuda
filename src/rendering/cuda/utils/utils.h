#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

#define CHECK_CUDA(debug, name)                                                                                                       \
    if (debug) {                                                                                                                      \
        auto ret = cudaDeviceSynchronize();                                                                                           \
        if (ret != cudaSuccess) {                                                                                                     \
            std::cerr << "\n[CUDA ERROR] in " << name << " " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
            throw std::runtime_error(cudaGetErrorString(ret));                                                                        \
        }                                                                                                                             \
    }

template <typename T>
inline __host__ __device__ T div_round_up(T value, T divisor) {
    return (value + divisor - 1) / divisor;
}
