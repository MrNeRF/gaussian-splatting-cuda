#pragma once

#include <torch/torch.h>

namespace gs {
namespace torch_utils {

template <typename T>
inline const T* get_const_data_ptr(const torch::Tensor& tensor, const char* tensor_name = "") {
    TORCH_CHECK(tensor.defined(), "Tensor '", tensor_name, "' is not defined.");
    // TORCH_CHECK(tensor.is_cuda(), "Tensor '", tensor_name, "' must be a CUDA tensor. Device: ", tensor.device()); // Re-enable if strictly CUDA needed
    TORCH_CHECK(tensor.is_contiguous(), "Tensor '", tensor_name, "' must be contiguous.");
    TORCH_CHECK(tensor.numel() == 0 || tensor.data_ptr() != nullptr, "Tensor '", tensor_name, "' has elements but data_ptr is null.");
    return tensor.data_ptr<T>();
}

template <typename T>
inline T* get_data_ptr(torch::Tensor& tensor, const char* tensor_name = "") {
    TORCH_CHECK(tensor.defined(), "Tensor '", tensor_name, "' is not defined.");
    // TORCH_CHECK(tensor.is_cuda(), "Tensor '", tensor_name, "' must be a CUDA tensor. Device: ", tensor.device()); // Re-enable if strictly CUDA needed
    TORCH_CHECK(tensor.is_contiguous(), "Tensor '", tensor_name, "' must be contiguous.");
    TORCH_CHECK(tensor.numel() == 0 || tensor.data_ptr() != nullptr, "Tensor '", tensor_name, "' has elements but data_ptr is null.");
    return tensor.data_ptr<T>();
}

// Helper to get background color from a rendered image tensor (typically B=1, H, W, C)
// Assumes background is at a known location, e.g., top-left pixel if not otherwise specified.
// This is a simplistic helper; real background color might be passed separately.
inline torch::Tensor get_bg_color_from_image(const torch::Tensor& image_tensor) {
    if (!image_tensor.defined() || image_tensor.numel() == 0) {
        return torch::tensor({0.0f, 0.0f, 0.0f}, image_tensor.options()); // Default black
    }
    // Assuming image_tensor is [H, W, C] or [B, H, W, C]
    if (image_tensor.dim() == 4) { // Batched
        return image_tensor.slice(0, 0, 1).slice(1, 0, 1).slice(2, 0, 1).squeeze(); // B, H, W, C -> C
    } else if (image_tensor.dim() == 3) { // HWC
        return image_tensor.slice(0, 0, 1).slice(1, 0, 1).squeeze(); // H, W, C -> C
    }
    return torch::tensor({0.0f, 0.0f, 0.0f}, image_tensor.options()); // Default
}


} // namespace torch_utils
} // namespace gs
