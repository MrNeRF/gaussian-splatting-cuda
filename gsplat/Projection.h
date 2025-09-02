#pragma once

#include "Cameras.h"
#include <cstdint>

namespace at {
    class Tensor;
}

namespace gsplat {

    void launch_projection_ut_3dgs_fused_kernel(
        // inputs
        const at::Tensor means,                   // [N, 3]
        const at::Tensor quats,                   // [N, 4]
        const at::Tensor scales,                  // [N, 3]
        const at::optional<at::Tensor> opacities, // [N] optional
        const at::Tensor viewmats0,               // [C, 4, 4]
        const at::optional<at::Tensor> viewmats1, // [C, 4, 4] optional for rolling shutter
        const at::Tensor Ks,                      // [C, 3, 3]
        const uint32_t image_width,
        const uint32_t image_height,
        const float eps2d,
        const float near_plane,
        const float far_plane,
        const float radius_clip,
        const CameraModelType camera_model,
        // uncented transform
        const UnscentedTransformParameters ut_params,
        ShutterType rs_type,
        const at::optional<at::Tensor> radial_coeffs,     // [C, 6] or [C, 4] optional
        const at::optional<at::Tensor> tangential_coeffs, // [C, 2] optional
        const at::optional<at::Tensor> thin_prism_coeffs, // [C, 2] optional
        // outputs
        at::Tensor radii,                      // [C, N, 2]
        at::Tensor means2d,                    // [C, N, 2]
        at::Tensor depths,                     // [C, N]
        at::Tensor conics,                     // [C, N, 3]
        at::optional<at::Tensor> compensations // [C, N] optional
    );

} // namespace gsplat
