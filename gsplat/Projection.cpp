#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Cameras.h"
#include "Common.h"     // where all the macros are defined
#include "Ops.h"        // a collection of all gsplat operators
#include "Projection.h" // where the launch function is declared

namespace gsplat {

    std::tuple<
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor>
    projection_ut_3dgs_fused(
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
        const bool calc_compensations,
        const CameraModelType camera_model,
        // uncented transform
        const UnscentedTransformParameters ut_params,
        ShutterType rs_type,
        const at::optional<at::Tensor> radial_coeffs,     // [C, 6] or [C, 4] optional
        const at::optional<at::Tensor> tangential_coeffs, // [C, 2] optional
        const at::optional<at::Tensor> thin_prism_coeffs  // [C, 2] optional
    ) {
        DEVICE_GUARD(means);
        CHECK_INPUT(means);
        CHECK_INPUT(quats);
        CHECK_INPUT(scales);
        if (opacities.has_value()) {
            CHECK_INPUT(opacities.value());
        }
        CHECK_INPUT(viewmats0);
        if (viewmats1.has_value()) {
            CHECK_INPUT(viewmats1.value());
        }
        CHECK_INPUT(Ks);
        if (radial_coeffs.has_value()) {
            CHECK_INPUT(radial_coeffs.value());
        }
        if (tangential_coeffs.has_value()) {
            CHECK_INPUT(tangential_coeffs.value());
        }
        if (thin_prism_coeffs.has_value()) {
            CHECK_INPUT(thin_prism_coeffs.value());
        }

        uint32_t N = means.size(0); // number of gaussians
        uint32_t C = Ks.size(0);    // number of cameras

        at::Tensor radii = at::empty({C, N, 2}, means.options().dtype(at::kInt));
        at::Tensor means2d = at::empty({C, N, 2}, means.options());
        at::Tensor depths = at::empty({C, N}, means.options());
        at::Tensor conics = at::empty({C, N, 3}, means.options());
        at::Tensor compensations;
        if (calc_compensations) {
            // we dont want NaN to appear in this tensor, so we zero intialize it
            compensations = at::zeros({C, N}, means.options());
        }

        launch_projection_ut_3dgs_fused_kernel(
            // inputs
            means,
            quats,
            scales,
            opacities,
            viewmats0,
            viewmats1,
            Ks,
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            camera_model,
            // uncented transform
            ut_params,
            rs_type,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            // outputs
            radii,
            means2d,
            depths,
            conics,
            calc_compensations ? at::optional<at::Tensor>(compensations)
                               : at::nullopt);
        return std::make_tuple(radii, means2d, depths, conics, compensations);
    }

} // namespace gsplat
