#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"     // where all the macros are defined
#include "Ops.h"        // a collection of all gsplat operators
#include "Relocation.h" // where the launch function is declared

namespace gsplat {

    std::tuple<at::Tensor, at::Tensor> relocation(
        at::Tensor opacities, // [N]
        at::Tensor scales,    // [N, 3]
        at::Tensor ratios,    // [N]
        at::Tensor binoms,    // [n_max, n_max]
        const int n_max) {
        DEVICE_GUARD(opacities);
        CHECK_INPUT(opacities);
        CHECK_INPUT(scales);
        CHECK_INPUT(ratios);
        CHECK_INPUT(binoms);
        at::Tensor new_opacities = at::empty_like(opacities);
        at::Tensor new_scales = at::empty_like(scales);

        launch_relocation_kernel(
            opacities, scales, ratios, binoms, n_max, new_opacities, new_scales);
        return std::make_tuple(new_opacities, new_scales);
    }

    void add_noise(
        at::Tensor raw_opacities, // [N]
        at::Tensor raw_scales,    // [N, 3]
        at::Tensor raw_quats,     // [N, 4]
        at::Tensor noise,         // [N, 3]
        at::Tensor means,         // [N, 3]
        const float current_lr) {
        DEVICE_GUARD(raw_opacities);
        CHECK_INPUT(raw_opacities);
        CHECK_INPUT(raw_scales);
        CHECK_INPUT(raw_quats);
        CHECK_INPUT(noise);
        CHECK_INPUT(means);

        launch_add_noise_kernel(
            raw_opacities, raw_scales, raw_quats, noise, means, current_lr);
    }

} // namespace gsplat
