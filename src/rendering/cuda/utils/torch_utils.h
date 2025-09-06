/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <functional>
#include <torch/torch.h>

inline std::function<char*(size_t N)> resize_function_wrapper(torch::Tensor& t) {
    auto lambda = [&t](const size_t N) {
        t.resize_({static_cast<long long>(N)});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

#define CHECK_INPUT(debug, tensor, name)                                                                                  \
    if (debug) {                                                                                                          \
        if (!tensor.is_cuda() || !tensor.is_floating_point() || !tensor.is_contiguous()) {                                \
            throw std::runtime_error("Input tensor '" + std::string(name) + "' must be a contiguous CUDA float tensor."); \
        }                                                                                                                 \
    }
