#pragma once

#include <torch/torch.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>
#include <variant>

// ---------------------------------------------------------------------------------------------

// Camera-specific types (camera model parameters and returns)

enum class ShutterType {
    ROLLING_TOP_TO_BOTTOM,
    ROLLING_LEFT_TO_RIGHT,
    ROLLING_BOTTOM_TO_TOP,
    ROLLING_RIGHT_TO_LEFT,
    GLOBAL
};

// ---------------------------------------------------------------------------------------------

// Gaussian-specific types
struct UnscentedTransformParameters {
    // See Gustafsson and Hendeby 2012 for sigma point parameterization - this
    // default parameter choice is based on
    //
    // - "The unscented Kalman filter for nonlinear estimation" - Wan and van
    // der Merwe 2000
    float alpha = 0.1;
    float beta = 2.f;
    float kappa = 0.f;

    // Parameters controlling validity of the unscented transform results
    float in_image_margin_factor =
        0.1f; // 10% out of bounds margin is acceptable for "valid" projection
              // state
    bool require_all_sigma_points_valid =
        false; // true: all sigma points must be valid to mark a projection as
               // "valid" false: a single valid sigma point is sufficient to
               // mark a projection as "valid"

    torch::Tensor to_tensor() const {
        return torch::tensor({alpha, beta, kappa, in_image_margin_factor,
                              static_cast<float>(require_all_sigma_points_valid)},
                             torch::TensorOptions().dtype(torch::kFloat32));
    }

    static UnscentedTransformParameters from_tensor(
        const torch::Tensor& tensor) {
        TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == 5,
                    "UnscentedTransformParameters must be a 1D tensor of size 5");
        return UnscentedTransformParameters{
            tensor[0].item<float>(), tensor[1].item<float>(),
            tensor[2].item<float>(), tensor[3].item<float>(),
            tensor[4].item<bool>()};
    }
};
