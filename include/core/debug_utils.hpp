#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>
#include <torch/torch.h>

namespace ts {
    namespace color {
        constexpr auto RED = "\033[31m";
        constexpr auto GREEN = "\033[32m";
        constexpr auto YELLOW = "\033[33m";
        constexpr auto BLUE = "\033[34m";
        constexpr auto MAGENTA = "\033[35m";
        constexpr auto CYAN = "\033[36m";
        constexpr auto RESET = "\033[0m";
    } // namespace color
} // namespace ts

// Helper to print tensor with custom precision
inline void print_tensor_with_precision(const torch::Tensor& tensor, int precision = 5) {
    auto old_precision = std::cout.precision();
    std::cout << std::fixed << std::setprecision(precision);
    std::cout << tensor << std::endl;
    std::cout.unsetf(std::ios::fixed);
    std::cout.precision(old_precision);
}

// Macro to inspect tensor properties with 5 decimal precision
#define INSPECT_TENSOR(tensor)                                                                              \
    do {                                                                                                    \
        std::cout << ts::color::CYAN << #tensor << ts::color::RESET << " | "                                \
                  << "shape: " << (tensor).sizes() << " | "                                                 \
                  << "device: " << (tensor).device() << " | "                                               \
                  << "dtype: " << (tensor).dtype() << " | "                                                 \
                  << "requires_grad: " << ((tensor).requires_grad() ? "true" : "false") << " | "            \
                  << "min: " << std::fixed << std::setprecision(5) << (tensor).min().item<float>() << " | " \
                  << "max: " << std::fixed << std::setprecision(5) << (tensor).max().item<float>() << " | " \
                  << "mean: " << std::fixed << std::setprecision(5) << (tensor).mean().item<float>()        \
                  << std::resetiosflags(std::ios::fixed) << std::endl;                                      \
    } while (0)

// Macro to inspect tensor and show first few values with 5 decimal precision
#define INSPECT_TENSOR_VALS(tensor, n)                                 \
    do {                                                               \
        INSPECT_TENSOR(tensor);                                        \
        std::cout << "  First " << (n) << " values: ";                 \
        auto flat = (tensor).flatten();                                \
        auto show_n = std::min((int64_t)(n), flat.numel());            \
        std::cout << std::fixed << std::setprecision(5);               \
        for (int64_t i = 0; i < show_n; ++i) {                         \
            std::cout << flat[i].item<float>() << " ";                 \
        }                                                              \
        std::cout << std::resetiosflags(std::ios::fixed) << std::endl; \
    } while (0)

// Macro to inspect full tensor content with 5 decimal precision
#define INSPECT_TENSOR_FULL(tensor)                                          \
    do {                                                                     \
        std::cout << ts::color::CYAN << #tensor << ts::color::RESET << " | " \
                  << "shape: " << (tensor).sizes() << " | "                  \
                  << "device: " << (tensor).device() << " | "                \
                  << "dtype: " << (tensor).dtype() << std::endl;             \
        print_tensor_with_precision(tensor, 5);                              \
    } while (0)

// Helper function to format float with precision
inline std::string format_float(float value, int precision = 5) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

// Debug print with timestamp
#define DEBUG_PRINT(msg)                                                                                   \
    do {                                                                                                   \
        auto now = std::chrono::system_clock::now();                                                       \
        auto time_t = std::chrono::system_clock::to_time_t(now);                                           \
        std::cout << ts::color::GREEN << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "] " \
                  << ts::color::RESET << msg << std::endl;                                                 \
    } while (0)

// Conditional debug print
#ifdef DEBUG
#define DEBUG_LOG(msg) DEBUG_PRINT(msg)
#else
#define DEBUG_LOG(msg) \
    do {               \
    } while (0)
#endif

// Print gradient info with 5 decimal precision
#define INSPECT_GRADIENT(tensor)                                                                                         \
    do {                                                                                                                 \
        if ((tensor).grad().defined()) {                                                                                 \
            std::cout << ts::color::MAGENTA << #tensor << ".grad" << ts::color::RESET << " | "                           \
                      << "shape: " << (tensor).grad().sizes() << " | "                                                   \
                      << "min: " << std::fixed << std::setprecision(5) << (tensor).grad().min().item<float>() << " | "   \
                      << "max: " << std::fixed << std::setprecision(5) << (tensor).grad().max().item<float>() << " | "   \
                      << "mean: " << std::fixed << std::setprecision(5) << (tensor).grad().mean().item<float>() << " | " \
                      << "norm: " << std::fixed << std::setprecision(5) << (tensor).grad().norm().item<float>()          \
                      << std::resetiosflags(std::ios::fixed) << std::endl;                                               \
        } else {                                                                                                         \
            std::cout << ts::color::MAGENTA << #tensor << ".grad" << ts::color::RESET                                    \
                      << " is not defined!" << std::endl;                                                                \
        }                                                                                                                \
    } while (0)

// Compare two tensors with precision
inline void compare_tensors(const torch::Tensor& a, const torch::Tensor& b,
                            const std::string& name_a = "A",
                            const std::string& name_b = "B",
                            int precision = 5) {
    std::cout << ts::color::YELLOW << "Comparing " << name_a << " and " << name_b << ts::color::RESET << std::endl;

    if (a.sizes() != b.sizes()) {
        std::cout << "  Shape mismatch: " << a.sizes() << " vs " << b.sizes() << std::endl;
        return;
    }

    auto diff = (a - b).abs();
    std::cout << std::fixed << std::setprecision(precision);
    std::cout << "  Max absolute difference: " << diff.max().item<float>() << std::endl;
    std::cout << "  Mean absolute difference: " << diff.mean().item<float>() << std::endl;
    std::cout << "  Relative error: " << (diff / (a.abs() + 1e-8)).mean().item<float>() << std::endl;
    std::cout << std::resetiosflags(std::ios::fixed);
}