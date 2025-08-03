#pragma once

#include "Ops.h"
#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include <torch/torch.h>

namespace tgs {

    class RasterizationFunction : public torch::autograd::Function<RasterizationFunction> {
    public:
        static torch::autograd::variable_list forward(
            torch::autograd::AutogradContext* ctx,
            const torch::Tensor& means3D,
            const torch::Tensor& sh0,
            const torch::Tensor& shN,
            const torch::Tensor& colors_precomp,
            const torch::Tensor& opacities,
            const torch::Tensor& scales,
            const torch::Tensor& rotations,
            const torch::Tensor& cov3Ds_precomp,
            const torch::Tensor& viewmat,
            const torch::Tensor& projmat,
            const torch::Tensor& bg_color,
            const torch::Tensor& campos,
            const torch::Tensor& settings
        );

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs
        );
    };

}