#include "core/bilateral_grid.hpp"
#include "kernels/bilateral_grid.cuh"

namespace gs {

    // Autograd function for bilateral grid slicing
    class BilateralGridSliceFunction : public torch::autograd::Function<BilateralGridSliceFunction> {
    public:
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor grid,
            torch::Tensor rgb) {
            // Input validation
            TORCH_CHECK(grid.dim() == 4 && grid.size(0) == 12,
                        "Grid must be [12, L, H, W]");
            TORCH_CHECK(rgb.dim() == 3 && rgb.size(2) == 3,
                        "RGB must be [H, W, 3]");
            TORCH_CHECK(grid.is_cuda() && rgb.is_cuda(),
                        "Tensors must be on CUDA");

            auto output = torch::empty_like(rgb);

            // Call CUDA kernel
            bilateral_grid::slice_forward_cuda(
                grid.contiguous(),
                rgb.contiguous(),
                output,
                true // use uniform coordinates
            );

            ctx->save_for_backward({grid, rgb});
            return {output};
        }

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs) {
            auto saved = ctx->get_saved_variables();
            auto grid = saved[0];
            auto rgb = saved[1];
            auto grad_output = grad_outputs[0];

            auto [grad_grid, grad_rgb] = bilateral_grid::slice_backward_cuda(
                grid, rgb, grad_output.contiguous());

            return {grad_grid, grad_rgb};
        }
    };

    // Autograd function for total variation loss
    class BilateralGridTVLossFunction : public torch::autograd::Function<BilateralGridTVLossFunction> {
    public:
        static torch::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor grids) {
            ctx->save_for_backward({grids});
            return bilateral_grid::tv_loss_forward_cuda(grids.contiguous());
        }

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs) {
            auto grids = ctx->get_saved_variables()[0];
            auto grad_output = grad_outputs[0];

            auto grad_grids = bilateral_grid::tv_loss_backward_cuda(
                grids, grad_output);

            return {grad_grids};
        }
    };

    // BilateralGrid implementation
    BilateralGrid::BilateralGrid(int num_images, int grid_W, int grid_H, int grid_L)
        : num_images_(num_images),
          grid_width_(grid_W),
          grid_height_(grid_H),
          grid_guidance_(grid_L) {

        // Initialize with identity transformation
        auto eye = torch::eye(4, torch::kFloat32).slice(0, 0, 3);
        auto grid = eye.repeat({grid_L * grid_H * grid_W, 1});
        grid = grid.reshape({1, grid_L, grid_H, grid_W, 12});
        grid = grid.permute({0, 4, 1, 2, 3});

        grids_ = grid.repeat({num_images, 1, 1, 1, 1}).to(torch::kCUDA);
        grids_.set_requires_grad(true);
    }

    torch::Tensor BilateralGrid::apply(const torch::Tensor& rgb, int image_idx) {
        TORCH_CHECK(image_idx >= 0 && image_idx < num_images_,
                    "Invalid image index: ", image_idx);

        // Handle different input formats
        torch::Tensor rgb_processed;
        if (rgb.dim() == 4 && rgb.size(0) == 1) {
            // Input is [1, C, H, W] - squeeze batch dimension
            rgb_processed = rgb.squeeze(0); // Now [C, H, W]
        } else if (rgb.dim() == 3) {
            // Input is already [C, H, W]
            rgb_processed = rgb;
        } else {
            TORCH_CHECK(false, "RGB must be [C, H, W] or [1, C, H, W], got ", rgb.sizes());
        }

        // Convert from [C, H, W] to [H, W, C]
        auto rgb_hwc = rgb_processed.permute({1, 2, 0}).contiguous();

        // Apply bilateral grid
        auto grid = grids_[image_idx];
        auto output = BilateralGridSliceFunction::apply(grid, rgb_hwc)[0];

        // Convert back to [C, H, W]
        auto result = output.permute({2, 0, 1}).contiguous();

        // If input had batch dimension, add it back
        if (rgb.dim() == 4) {
            result = result.unsqueeze(0);
        }

        return result;
    }

    torch::Tensor BilateralGrid::tv_loss() const {
        return BilateralGridTVLossFunction::apply(grids_);
    }

} // namespace gs