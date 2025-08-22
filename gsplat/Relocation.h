#pragma once

#include <cstdint>

namespace at {
    class Tensor;
}

namespace gsplat {

    void launch_relocation_kernel(
        // inputs
        at::Tensor opacities, // [N]
        at::Tensor scales,    // [N, 3]
        at::Tensor ratios,    // [N]
        at::Tensor binoms,    // [n_max, n_max]
        const int n_max,
        // outputs
        at::Tensor new_opacities, // [N]
        at::Tensor new_scales     // [N, 3]
    );

    void launch_add_noise_kernel(
        at::Tensor raw_opacities, // [N]
        at::Tensor raw_scales,    // [N, 3]
        at::Tensor raw_quats,     // [N, 4]
        at::Tensor noise,         // [N, 3]
        at::Tensor means,         // [N, 3]
        const float current_lr);

} // namespace gsplat