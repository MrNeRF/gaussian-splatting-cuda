#pragma once

#include <cstdint>

namespace at {
class Tensor;
}

namespace gsplat {

void launch_spherical_harmonics_fwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor sh0_coeffs,          // [..., 1, 3]
    const at::Tensor shN_coeffs,          // [..., K-1, 3]
    const at::optional<at::Tensor> masks, // [...]
    // outputs
    at::Tensor colors // [..., 2]
);

void launch_spherical_harmonics_bwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor shN_coeffs,          // [..., K-1, 3]
    const at::optional<at::Tensor> masks, // [...]
    const at::Tensor v_colors,            // [..., 3]
    // outputs
    at::Tensor v_coeffs0,           // [..., K, 3]
    at::Tensor v_coeffsN,           // [..., K, 3]
    at::optional<at::Tensor> v_dirs // [..., 3]
);

} // namespace gsplat