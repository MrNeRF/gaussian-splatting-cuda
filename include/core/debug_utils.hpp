// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <torch/torch.h>
#include <tuple>
#include <vector>

namespace ts {
    // Color codes for terminal output
    namespace color {
        const std::string RESET = "\033[0m";
        const std::string BLACK = "\033[30m";
        const std::string RED = "\033[31m";
        const std::string GREEN = "\033[32m";
        const std::string YELLOW = "\033[33m";
        const std::string BLUE = "\033[34m";
        const std::string MAGENTA = "\033[35m";
        const std::string CYAN = "\033[36m";
        const std::string WHITE = "\033[37m";
        const std::string BOLD = "\033[1m";
    } // namespace color

    // Enhanced tensor saving with metadata
    inline void save_my_tensor(const torch::Tensor& tensor, const std::string& filename, bool save_metadata = true) {
        std::cout << color::CYAN << "Saving tensor to: " << filename << color::RESET << std::endl;
        std::cout << "  Shape: " << tensor.sizes() << ", Type: " << tensor.dtype() << std::endl;

        auto cpu_tensor = tensor.contiguous().to(torch::kCPU); // Ensure contiguous and on CPU
        int64_t numel = cpu_tensor.numel();
        std::vector<int64_t> sizes = cpu_tensor.sizes().vec();
        int dims = cpu_tensor.dim();

        std::ofstream outfile(filename, std::ios::binary);
        if (!outfile.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        // Write magic number for format verification
        uint32_t magic = 0x54534E52; // "TSNR" in hex
        outfile.write(reinterpret_cast<char*>(&magic), sizeof(uint32_t));

        // Write version
        uint32_t version = 2;
        outfile.write(reinterpret_cast<char*>(&version), sizeof(uint32_t));

        // Write data type
        int dtype_code = -1;
        if (cpu_tensor.dtype() == torch::kFloat32)
            dtype_code = 0;
        else if (cpu_tensor.dtype() == torch::kFloat64)
            dtype_code = 1;
        else if (cpu_tensor.dtype() == torch::kInt32)
            dtype_code = 2;
        else if (cpu_tensor.dtype() == torch::kInt64)
            dtype_code = 3;
        else if (cpu_tensor.dtype() == torch::kBool)
            dtype_code = 4;
        else if (cpu_tensor.dtype() == torch::kUInt8)
            dtype_code = 5;
        else if (cpu_tensor.dtype() == torch::kInt16)
            dtype_code = 6;
        else if (cpu_tensor.dtype() == torch::kFloat16)
            dtype_code = 7;
        else {
            throw std::runtime_error("Unsupported tensor type: " + std::string(torch::toString(cpu_tensor.dtype())));
        }
        outfile.write(reinterpret_cast<char*>(&dtype_code), sizeof(int));

        // Write dimensions
        outfile.write(reinterpret_cast<char*>(&dims), sizeof(int));

        // Write sizes
        outfile.write(reinterpret_cast<char*>(sizes.data()), dims * sizeof(int64_t));

        // Write requires_grad flag
        bool requires_grad = tensor.requires_grad();
        outfile.write(reinterpret_cast<char*>(&requires_grad), sizeof(bool));

        // Write tensor data based on its type
        if (cpu_tensor.dtype() == torch::kFloat32) {
            outfile.write(reinterpret_cast<char*>(cpu_tensor.data_ptr<float>()), numel * sizeof(float));
        } else if (cpu_tensor.dtype() == torch::kFloat64) {
            outfile.write(reinterpret_cast<char*>(cpu_tensor.data_ptr<double>()), numel * sizeof(double));
        } else if (cpu_tensor.dtype() == torch::kInt64) {
            outfile.write(reinterpret_cast<char*>(cpu_tensor.data_ptr<int64_t>()), numel * sizeof(int64_t));
        } else if (cpu_tensor.dtype() == torch::kInt32) {
            outfile.write(reinterpret_cast<char*>(cpu_tensor.data_ptr<int32_t>()), numel * sizeof(int32_t));
        } else if (cpu_tensor.dtype() == torch::kBool) {
            // Convert bool to uint8 for storage
            auto uint8_tensor = cpu_tensor.to(torch::kUInt8);
            outfile.write(reinterpret_cast<char*>(uint8_tensor.data_ptr<uint8_t>()), numel * sizeof(uint8_t));
        } else if (cpu_tensor.dtype() == torch::kUInt8) {
            outfile.write(reinterpret_cast<char*>(cpu_tensor.data_ptr<uint8_t>()), numel * sizeof(uint8_t));
        } else if (cpu_tensor.dtype() == torch::kInt16) {
            outfile.write(reinterpret_cast<char*>(cpu_tensor.data_ptr<int16_t>()), numel * sizeof(int16_t));
        }

        outfile.close();

        // Save metadata file
        if (save_metadata) {
            std::string meta_filename = filename + ".meta";
            std::ofstream metafile(meta_filename);
            metafile << "Tensor Metadata\n";
            metafile << "===============\n";
            metafile << "Shape: " << tensor.sizes() << "\n";
            metafile << "Type: " << tensor.dtype() << "\n";
            metafile << "Device: " << tensor.device() << "\n";
            metafile << "Requires Grad: " << (tensor.requires_grad() ? "Yes" : "No") << "\n";
            metafile << "Is Contiguous: " << (tensor.is_contiguous() ? "Yes" : "No") << "\n";
            metafile << "Element Count: " << numel << "\n";
            metafile.close();
        }
    }

    // Enhanced tensor loading with type detection
    inline torch::Tensor load_my_tensor(const std::string& filename, torch::Device device = torch::kCPU) {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile.is_open()) {
            throw std::runtime_error("Failed to open file " + filename);
        }

        // Check magic number
        uint32_t magic;
        infile.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));

        // Handle legacy format (no magic number)
        if (magic != 0x54534E52) {
            infile.seekg(0); // Reset to beginning

            // Legacy loading code
            int dims;
            infile.read(reinterpret_cast<char*>(&dims), sizeof(int));

            std::vector<int64_t> sizes(dims);
            infile.read(reinterpret_cast<char*>(sizes.data()), dims * sizeof(int64_t));

            int64_t numel = 1;
            for (int i = 0; i < dims; ++i) {
                numel *= sizes[i];
            }

            // Legacy format - read data size information
            size_t data_offset = sizeof(int) + dims * sizeof(int64_t);
            infile.seekg(0, std::ios::end);
            size_t file_size = infile.tellg();
            size_t data_size = file_size - data_offset;
            infile.seekg(data_offset);

            // Try to determine data type from file size
            // Assume float32 for legacy format
            if (data_size == numel * sizeof(float)) {
                std::vector<float> data(numel);
                infile.read(reinterpret_cast<char*>(data.data()), numel * sizeof(float));
                infile.close();
                return torch::tensor(data).reshape(sizes).to(device);
            } else if (data_size == numel * sizeof(double)) {
                std::vector<double> data(numel);
                infile.read(reinterpret_cast<char*>(data.data()), numel * sizeof(double));
                infile.close();
                return torch::tensor(data).reshape(sizes).to(device);
            } else {
                // Default to float32
                std::vector<float> data(numel);
                infile.read(reinterpret_cast<char*>(data.data()), numel * sizeof(float));
                infile.close();
                return torch::tensor(data).reshape(sizes).to(device);
            }
        }

        // Read version
        uint32_t version;
        infile.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));

        // Read data type
        int dtype_code;
        infile.read(reinterpret_cast<char*>(&dtype_code), sizeof(int));

        // Read dimensions
        int dims;
        infile.read(reinterpret_cast<char*>(&dims), sizeof(int));

        // Read sizes
        std::vector<int64_t> sizes(dims);
        infile.read(reinterpret_cast<char*>(sizes.data()), dims * sizeof(int64_t));

        // Read requires_grad flag
        bool requires_grad;
        infile.read(reinterpret_cast<char*>(&requires_grad), sizeof(bool));

        // Calculate total elements
        int64_t numel = 1;
        for (int i = 0; i < dims; ++i) {
            numel *= sizes[i];
        }

        torch::Tensor tensor;

        // Load data based on type
        switch (dtype_code) {
        case 0: { // Float32
            std::vector<float> data(numel);
            infile.read(reinterpret_cast<char*>(data.data()), numel * sizeof(float));
            tensor = torch::tensor(data).reshape(sizes);
            break;
        }
        case 1: { // Float64
            std::vector<double> data(numel);
            infile.read(reinterpret_cast<char*>(data.data()), numel * sizeof(double));
            tensor = torch::tensor(data).reshape(sizes);
            break;
        }
        case 2: { // Int32
            std::vector<int32_t> data(numel);
            infile.read(reinterpret_cast<char*>(data.data()), numel * sizeof(int32_t));
            tensor = torch::tensor(data).reshape(sizes);
            break;
        }
        case 3: { // Int64
            std::vector<int64_t> data(numel);
            infile.read(reinterpret_cast<char*>(data.data()), numel * sizeof(int64_t));
            tensor = torch::tensor(data).reshape(sizes);
            break;
        }
        case 4: { // Bool
            // std::vector<bool> is special - need to use uint8_t
            std::vector<uint8_t> data(numel);
            infile.read(reinterpret_cast<char*>(data.data()), numel * sizeof(uint8_t));
            tensor = torch::tensor(data).to(torch::kBool).reshape(sizes);
            break;
        }
        case 5: { // UInt8
            std::vector<uint8_t> data(numel);
            infile.read(reinterpret_cast<char*>(data.data()), numel * sizeof(uint8_t));
            tensor = torch::tensor(data).reshape(sizes);
            break;
        }
        case 6: { // Int16
            std::vector<int16_t> data(numel);
            infile.read(reinterpret_cast<char*>(data.data()), numel * sizeof(int16_t));
            tensor = torch::tensor(data).reshape(sizes);
            break;
        }
        default:
            throw std::runtime_error("Unknown data type code: " + std::to_string(dtype_code));
        }

        infile.close();

        tensor = tensor.to(device);
        tensor.set_requires_grad(requires_grad);

        return tensor;
    }

    // Compute comprehensive tensor statistics
    struct TensorStats {
        double min, max, mean, std, median;
        int64_t nan_count, inf_count, zero_count;
        double sparsity;
        std::vector<double> percentiles;
    };

    inline TensorStats compute_tensor_stats(const torch::Tensor& tensor) {
        TensorStats stats;

        if (tensor.numel() == 0) {
            stats = {0, 0, 0, 0, 0, 0, 0, 0, 1.0, {}};
            return stats;
        }

        auto cpu_tensor = tensor.to(torch::kCPU).to(torch::kFloat32).flatten();

        // Basic statistics
        stats.min = cpu_tensor.min().item<double>();
        stats.max = cpu_tensor.max().item<double>();
        stats.mean = cpu_tensor.mean().item<double>();
        stats.std = cpu_tensor.std().item<double>();

        // Count special values
        stats.nan_count = torch::isnan(cpu_tensor).sum().item<int64_t>();
        stats.inf_count = torch::isinf(cpu_tensor).sum().item<int64_t>();
        stats.zero_count = (cpu_tensor == 0).sum().item<int64_t>();
        stats.sparsity = static_cast<double>(stats.zero_count) / tensor.numel();

        // Compute median and percentiles
        auto sorted_tuple = cpu_tensor.sort();
        auto sorted = std::get<0>(sorted_tuple); // Get the values tensor
        int64_t n = sorted.numel();
        stats.median = sorted[n / 2].item<double>();

        // Compute percentiles (1st, 5th, 25th, 75th, 95th, 99th)
        std::vector<double> percentile_values = {0.01, 0.05, 0.25, 0.75, 0.95, 0.99};
        for (double p : percentile_values) {
            int64_t idx = static_cast<int64_t>(p * (n - 1));
            stats.percentiles.push_back(sorted[idx].item<double>());
        }

        return stats;
    }

    // Pretty print tensor contents with various options
    inline void print_tensor_contents(const torch::Tensor& tensor, int max_elements = 100, int precision = 4) {
        if (tensor.numel() == 0) {
            std::cout << "Empty tensor" << std::endl;
            return;
        }

        auto cpu_tensor = tensor.to(torch::kCPU);
        std::cout << std::fixed << std::setprecision(precision);

        if (tensor.dim() == 0) {
            // Scalar
            std::cout << cpu_tensor.item<double>() << std::endl;
        } else if (tensor.dim() == 1) {
            // Vector
            int64_t size = tensor.size(0);
            std::cout << "[";
            for (int64_t i = 0; i < std::min(size, static_cast<int64_t>(max_elements)); ++i) {
                if (i > 0)
                    std::cout << ", ";
                std::cout << cpu_tensor[i].item<double>();
            }
            if (size > max_elements)
                std::cout << ", ...";
            std::cout << "]" << std::endl;
        } else if (tensor.dim() == 2) {
            // Matrix
            int64_t rows = tensor.size(0);
            int64_t cols = tensor.size(1);
            int max_rows = std::sqrt(max_elements);
            int max_cols = max_elements / max_rows;

            for (int64_t i = 0; i < std::min(rows, static_cast<int64_t>(max_rows)); ++i) {
                std::cout << "[";
                for (int64_t j = 0; j < std::min(cols, static_cast<int64_t>(max_cols)); ++j) {
                    if (j > 0)
                        std::cout << ", ";
                    std::cout << std::setw(8) << cpu_tensor[i][j].item<double>();
                }
                if (cols > max_cols)
                    std::cout << ", ...";
                std::cout << "]" << std::endl;
            }
            if (rows > max_rows)
                std::cout << "..." << std::endl;
        } else {
            // Higher dimensional - show first and last slices
            std::cout << "Tensor shape " << tensor.sizes() << " (showing slice [0, :, ...])" << std::endl;
            auto slice = tensor.select(0, 0);
            print_tensor_contents(slice, max_elements, precision);
        }
    }

    // Enhanced debug info with statistics and memory usage
    inline void print_debug_info(const torch::Tensor& tensor, const std::string& tensor_name,
                                 bool save_tensor = false, bool show_stats = true, bool show_contents = false) {
        std::cout << color::BOLD << color::GREEN << "\n=== Tensor Debug Info: " << tensor_name << " ===" << color::RESET << std::endl;

        // Basic info
        std::cout << color::CYAN << "Basic Information:" << color::RESET << std::endl;
        std::cout << "  Shape: " << tensor.sizes() << std::endl;
        std::cout << "  Data type: " << tensor.dtype() << std::endl;
        std::cout << "  Device: " << tensor.device() << std::endl;
        std::cout << "  Requires grad: " << (tensor.requires_grad() ? "Yes" : "No") << std::endl;
        std::cout << "  Is contiguous: " << (tensor.is_contiguous() ? "Yes" : "No") << std::endl;
        std::cout << "  Element count: " << tensor.numel() << std::endl;

        // Memory info
        size_t bytes = tensor.element_size() * tensor.numel();
        std::cout << "  Memory usage: " << bytes << " bytes ("
                  << std::fixed << std::setprecision(2) << bytes / (1024.0 * 1024.0) << " MB)" << std::endl;

        // Gradient info
        if (tensor.requires_grad() && tensor.grad().defined()) {
            std::cout << color::YELLOW << "Gradient Information:" << color::RESET << std::endl;
            std::cout << "  Grad shape: " << tensor.grad().sizes() << std::endl;
            std::cout << "  Grad norm: " << tensor.grad().norm().item<double>() << std::endl;
        }

        // Statistics
        if (show_stats && tensor.numel() > 0 &&
            (tensor.dtype() == torch::kFloat32 || tensor.dtype() == torch::kFloat64 ||
             tensor.dtype() == torch::kFloat16)) {
            std::cout << color::MAGENTA << "Statistics:" << color::RESET << std::endl;
            auto stats = compute_tensor_stats(tensor);
            std::cout << "  Min: " << stats.min << ", Max: " << stats.max << std::endl;
            std::cout << "  Mean: " << stats.mean << ", Std: " << stats.std << std::endl;
            std::cout << "  Median: " << stats.median << std::endl;
            std::cout << "  Percentiles [1%, 5%, 25%, 75%, 95%, 99%]: [";
            for (size_t i = 0; i < stats.percentiles.size(); ++i) {
                if (i > 0)
                    std::cout << ", ";
                std::cout << std::fixed << std::setprecision(4) << stats.percentiles[i];
            }
            std::cout << "]" << std::endl;
            std::cout << "  NaN count: " << stats.nan_count << std::endl;
            std::cout << "  Inf count: " << stats.inf_count << std::endl;
            std::cout << "  Zero count: " << stats.zero_count
                      << " (sparsity: " << std::fixed << std::setprecision(2)
                      << stats.sparsity * 100 << "%)" << std::endl;

            // Check for anomalies
            if (stats.nan_count > 0) {
                std::cout << color::RED << "  WARNING: Tensor contains NaN values!" << color::RESET << std::endl;
            }
            if (stats.inf_count > 0) {
                std::cout << color::RED << "  WARNING: Tensor contains Inf values!" << color::RESET << std::endl;
            }
            if (stats.std == 0 && tensor.numel() > 1) {
                std::cout << color::YELLOW << "  NOTE: All values are identical (std=0)" << color::RESET << std::endl;
            }
        }

        // Show contents
        if (show_contents) {
            std::cout << color::BLUE << "Contents (first 100 elements):" << color::RESET << std::endl;
            print_tensor_contents(tensor, 100);
        }

        // Save tensor
        if (save_tensor) {
            save_my_tensor(tensor, "debug_" + tensor_name + ".pt");
        }

        std::cout << color::BOLD << color::GREEN << "=== End Debug Info ===" << color::RESET << std::endl;
    }

    // Compare two tensors
    inline void compare_tensors(const torch::Tensor& tensor1, const torch::Tensor& tensor2,
                                const std::string& name1 = "Tensor1", const std::string& name2 = "Tensor2",
                                double rtol = 1e-5, double atol = 1e-8) {
        std::cout << color::BOLD << color::CYAN << "\n=== Tensor Comparison: " << name1 << " vs " << name2 << " ===" << color::RESET << std::endl;

        // Check shapes
        bool same_shape = tensor1.sizes() == tensor2.sizes();
        std::cout << "Shape match: " << (same_shape ? color::GREEN + "Yes" : color::RED + "No") << color::RESET << std::endl;
        if (!same_shape) {
            std::cout << "  " << name1 << ": " << tensor1.sizes() << std::endl;
            std::cout << "  " << name2 << ": " << tensor2.sizes() << std::endl;
        }

        // Check dtypes
        bool same_dtype = tensor1.dtype() == tensor2.dtype();
        std::cout << "Dtype match: " << (same_dtype ? color::GREEN + "Yes" : color::RED + "No") << color::RESET << std::endl;
        if (!same_dtype) {
            std::cout << "  " << name1 << ": " << tensor1.dtype() << std::endl;
            std::cout << "  " << name2 << ": " << tensor2.dtype() << std::endl;
        }

        if (same_shape && same_dtype) {
            // Compute differences
            auto diff = (tensor1 - tensor2).abs();
            auto relative_diff = diff / (tensor2.abs() + 1e-10);

            bool all_close = torch::allclose(tensor1, tensor2, rtol, atol);
            std::cout << "All close (rtol=" << rtol << ", atol=" << atol << "): "
                      << (all_close ? color::GREEN + "Yes" : color::RED + "No") << color::RESET << std::endl;

            if (!all_close) {
                std::cout << "Difference statistics:" << std::endl;
                std::cout << "  Max absolute diff: " << diff.max().item<double>() << std::endl;
                std::cout << "  Mean absolute diff: " << diff.mean().item<double>() << std::endl;
                std::cout << "  Max relative diff: " << relative_diff.max().item<double>() << std::endl;
                std::cout << "  Elements not close: " << (diff > atol).sum().item<int64_t>()
                          << " / " << tensor1.numel() << std::endl;

                // Find location of max diff
                auto max_diff_flat_idx = diff.argmax();
                std::cout << "  Location of max diff (flat index): " << max_diff_flat_idx.item<int64_t>() << std::endl;
            }
        }

        std::cout << color::BOLD << color::CYAN << "=== End Comparison ===" << color::RESET << std::endl;
    }

    // Validate tensor for common issues
    inline bool validate_tensor(const torch::Tensor& tensor, const std::string& tensor_name) {
        bool is_valid = true;
        std::cout << color::BOLD << color::YELLOW << "\n=== Validating: " << tensor_name << " ===" << color::RESET << std::endl;

        // Check if tensor is defined
        if (!tensor.defined()) {
            std::cout << color::RED << "ERROR: Tensor is not defined!" << color::RESET << std::endl;
            return false;
        }

        // Check for NaN/Inf
        if (tensor.dtype() == torch::kFloat32 || tensor.dtype() == torch::kFloat64 || tensor.dtype() == torch::kFloat16) {
            bool has_nan = torch::any(torch::isnan(tensor)).item<bool>();
            bool has_inf = torch::any(torch::isinf(tensor)).item<bool>();

            if (has_nan) {
                std::cout << color::RED << "ERROR: Tensor contains NaN values!" << color::RESET << std::endl;
                is_valid = false;
            }
            if (has_inf) {
                std::cout << color::RED << "ERROR: Tensor contains Inf values!" << color::RESET << std::endl;
                is_valid = false;
            }
        }

        // Check if empty
        if (tensor.numel() == 0) {
            std::cout << color::YELLOW << "WARNING: Tensor is empty (0 elements)" << color::RESET << std::endl;
        }

        // Check memory layout
        if (!tensor.is_contiguous()) {
            std::cout << color::YELLOW << "NOTE: Tensor is not contiguous in memory" << color::RESET << std::endl;
        }

        if (is_valid) {
            std::cout << color::GREEN << "Tensor is valid!" << color::RESET << std::endl;
        }

        return is_valid;
    }

    // Tensor shape analysis
    inline void analyze_tensor_shape(const torch::Tensor& tensor, const std::string& tensor_name) {
        std::cout << color::BOLD << color::BLUE << "\n=== Shape Analysis: " << tensor_name << " ===" << color::RESET << std::endl;

        std::cout << "Dimensions: " << tensor.dim() << std::endl;
        std::cout << "Shape: " << tensor.sizes() << std::endl;
        std::cout << "Strides: " << tensor.strides() << std::endl;

        // Analyze each dimension
        for (int i = 0; i < tensor.dim(); ++i) {
            std::cout << "  Dim " << i << ": size=" << tensor.size(i) << ", stride=" << tensor.stride(i) << std::endl;
        }

        // Memory layout info
        std::cout << "Total elements: " << tensor.numel() << std::endl;
        std::cout << "Contiguous: " << (tensor.is_contiguous() ? "Yes" : "No") << std::endl;
        std::cout << "Element size: " << tensor.element_size() << " bytes" << std::endl;
        std::cout << "Total memory: " << tensor.numel() * tensor.element_size() << " bytes" << std::endl;
    }

    // Memory usage tracker
    class MemoryTracker {
    private:
        std::map<std::string, size_t> tensor_sizes;

    public:
        void track(const std::string& name, const torch::Tensor& tensor) {
            tensor_sizes[name] = tensor.numel() * tensor.element_size();
        }

        void report() const {
            std::cout << color::BOLD << color::MAGENTA << "\n=== Memory Usage Report ===" << color::RESET << std::endl;

            size_t total = 0;
            for (const auto& pair : tensor_sizes) {
                total += pair.second;
                std::cout << std::setw(30) << std::left << pair.first << ": "
                          << std::setw(12) << std::right << pair.second << " bytes ("
                          << std::fixed << std::setprecision(2) << pair.second / (1024.0 * 1024.0) << " MB)" << std::endl;
            }

            std::cout << std::string(50, '-') << std::endl;
            std::cout << std::setw(30) << std::left << "Total"
                      << ": "
                      << std::setw(12) << std::right << total << " bytes ("
                      << std::fixed << std::setprecision(2) << total / (1024.0 * 1024.0) << " MB)" << std::endl;
        }
    };

    // CUDA error checking (unchanged)
#undef DEBUG_ERRORS
    //#define DEBUG_ERRORS

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
    template <typename T>
    inline void check(T err, const char* const func, const char* const file,
                      const int line) {
#ifdef DEBUG_ERRORS
        if (err != cudaSuccess) {
            std::cerr << color::RED << "CUDA Runtime Error at: " << file << ":" << line << color::RESET << std::endl;
            std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
            std::exit(EXIT_FAILURE);
        }
#endif // DEBUG_ERRORS
    }

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
    inline void checkLast(const char* const file, const int line) {
#ifdef DEBUG_ERRORS
        cudaDeviceSynchronize();
        cudaError_t err{cudaGetLastError()};
        if (err != cudaSuccess) {
            std::cerr << color::RED << "CUDA Runtime Error at: " << file << ":" << line << color::RESET << std::endl;
            std::cerr << cudaGetErrorString(err) << std::endl;
            std::exit(EXIT_FAILURE);
        }
#endif // DEBUG_ERRORS
    }

    // Quick tensor inspection macro
#define INSPECT_TENSOR(tensor)      ts::print_debug_info(tensor, #tensor, false, true, false)
#define INSPECT_TENSOR_FULL(tensor) ts::print_debug_info(tensor, #tensor, false, true, true)
#define SAVE_TENSOR(tensor)         ts::save_my_tensor(tensor, #tensor ".pt")
#define VALIDATE_TENSOR(tensor)     ts::validate_tensor(tensor, #tensor)

} // namespace ts