// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once
#include <fstream>
#include <iostream>
#include <torch/torch.h>
#include <vector>

namespace ts {
    inline void save_my_tensor(const torch::Tensor& tensor, std::string filename) {
        std::cout << filename << ": Expected dims: " << tensor.dim() << " expected shape: " << tensor.sizes() << "Expected type: " << tensor.dtype() << std::endl;
        auto cpu_tensor = tensor.to(torch::kCPU); // Move tensor to CPU
        int64_t numel = cpu_tensor.numel();
        std::vector<int64_t> sizes = cpu_tensor.sizes().vec();
        int dims = cpu_tensor.dim();

        std::ofstream outfile(filename, std::ios::binary);

        // Write dimensions
        outfile.write(reinterpret_cast<char*>(&dims), sizeof(int));

        // Write sizes
        outfile.write(reinterpret_cast<char*>(sizes.data()), dims * sizeof(int64_t));

        // Write tensor data
        outfile.write(reinterpret_cast<char*>(cpu_tensor.data_ptr()), numel * sizeof(float));

        outfile.close();
    }
} // namespace ts
