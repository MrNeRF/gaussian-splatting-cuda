#pragma once
#include <torch/torch.h>

namespace gs {
    namespace bilateral_grid {

        // Forward declarations
        void slice_forward_cuda(
            const torch::Tensor& grid, // [12, L, H, W]
            const torch::Tensor& rgb,  // [H, W, 3]
            torch::Tensor& output,     // [H, W, 3]
            bool use_uniform_coords = true);

        std::tuple<torch::Tensor, torch::Tensor> slice_backward_cuda(
            const torch::Tensor& grid,       // [12, L, H, W]
            const torch::Tensor& rgb,        // [H, W, 3]
            const torch::Tensor& grad_output // [H, W, 3]
        );

        torch::Tensor tv_loss_forward_cuda(
            const torch::Tensor& grids // [N, 12, L, H, W]
        );

        torch::Tensor tv_loss_backward_cuda(
            const torch::Tensor& grids,      // [N, 12, L, H, W]
            const torch::Tensor& grad_output // scalar
        );

    } // namespace bilateral_grid
} // namespace gs