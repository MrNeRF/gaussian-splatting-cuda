#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Cameras.h"
#include "Common.h"
#include "Ops.h"
#include "Rasterization.h"

namespace gsplat {

    ////////////////////////////////////////////////////
    // 3DGS (from world)
    ////////////////////////////////////////////////////

    std::tuple<at::Tensor, at::Tensor, at::Tensor> rasterize_to_pixels_from_world_3dgs_fwd(
        // Gaussian parameters
        const at::Tensor means,                     // [N, 3]
        const at::Tensor quats,                     // [N, 4]
        const at::Tensor scales,                    // [N, 3]
        const at::Tensor colors,                    // [C, N, channels] or [nnz, channels]
        const at::Tensor opacities,                 // [C, N]  or [nnz]
        const at::optional<at::Tensor> backgrounds, // [C, channels]
        const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // camera
        const at::Tensor viewmats0,               // [C, 4, 4]
        const at::optional<at::Tensor> viewmats1, // [C, 4, 4] optional for rolling shutter
        const at::Tensor Ks,                      // [C, 3, 3]
        const CameraModelType camera_model,
        // uncented transform
        const UnscentedTransformParameters ut_params,
        ShutterType rs_type,
        const at::optional<at::Tensor> radial_coeffs,     // [C, 6] or [C, 4] optional
        const at::optional<at::Tensor> tangential_coeffs, // [C, 2] optional
        const at::optional<at::Tensor> thin_prism_coeffs, // [C, 2] optional
        // intersections
        const at::Tensor tile_offsets, // [C, tile_height, tile_width]
        const at::Tensor flatten_ids   // [n_isects]
    ) {
        DEVICE_GUARD(means);
        CHECK_INPUT(means);
        CHECK_INPUT(quats);
        CHECK_INPUT(colors);
        CHECK_INPUT(colors);
        CHECK_INPUT(opacities);
        CHECK_INPUT(tile_offsets);
        CHECK_INPUT(flatten_ids);
        if (backgrounds.has_value()) {
            CHECK_INPUT(backgrounds.value());
        }
        if (masks.has_value()) {
            CHECK_INPUT(masks.value());
        }

        uint32_t C = tile_offsets.size(0); // number of cameras
        uint32_t channels = colors.size(-1);
        assert(channels == 3); // only support RGB for now

        at::Tensor renders =
            at::empty({C, image_height, image_width, channels}, means.options());
        at::Tensor alphas =
            at::empty({C, image_height, image_width, 1}, means.options());
        at::Tensor last_ids = at::empty(
            {C, image_height, image_width}, means.options().dtype(at::kInt));

#define __LAUNCH_KERNEL__(N)                                      \
    case N:                                                       \
        launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel<N>( \
            means,                                                \
            quats,                                                \
            scales,                                               \
            colors,                                               \
            opacities,                                            \
            backgrounds,                                          \
            masks,                                                \
            image_width,                                          \
            image_height,                                         \
            tile_size,                                            \
            viewmats0,                                            \
            viewmats1,                                            \
            Ks,                                                   \
            camera_model,                                         \
            ut_params,                                            \
            rs_type,                                              \
            radial_coeffs,                                        \
            tangential_coeffs,                                    \
            thin_prism_coeffs,                                    \
            tile_offsets,                                         \
            flatten_ids,                                          \
            renders,                                              \
            alphas,                                               \
            last_ids);                                            \
        break;

        // TODO: an optimization can be done by passing the actual number of
        // channels into the kernel functions and avoid necessary global memory
        // writes. This requires moving the channel padding from python to C side.
        switch (channels) {
            __LAUNCH_KERNEL__(1)
            __LAUNCH_KERNEL__(2)
            __LAUNCH_KERNEL__(3)
            __LAUNCH_KERNEL__(4)
            __LAUNCH_KERNEL__(5)
            __LAUNCH_KERNEL__(8)
            __LAUNCH_KERNEL__(9)
            __LAUNCH_KERNEL__(16)
            __LAUNCH_KERNEL__(17)
            __LAUNCH_KERNEL__(32)
            __LAUNCH_KERNEL__(33)
            __LAUNCH_KERNEL__(64)
            __LAUNCH_KERNEL__(65)
            __LAUNCH_KERNEL__(128)
            __LAUNCH_KERNEL__(129)
            __LAUNCH_KERNEL__(256)
            __LAUNCH_KERNEL__(257)
            __LAUNCH_KERNEL__(512)
            __LAUNCH_KERNEL__(513)
        default:
            AT_ERROR("Unsupported number of channels: ", channels);
        }
#undef __LAUNCH_KERNEL__

        return std::make_tuple(renders, alphas, last_ids);
    };

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    rasterize_to_pixels_from_world_3dgs_bwd(
        // Gaussian parameters
        const at::Tensor means,                     // [N, 3]
        const at::Tensor quats,                     // [N, 4]
        const at::Tensor scales,                    // [N, 3]
        const at::Tensor colors,                    // [C, N, 3] or [nnz, 3]
        const at::Tensor opacities,                 // [C, N] or [nnz]
        const at::optional<at::Tensor> backgrounds, // [C, 3]
        const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // camera
        const at::Tensor viewmats0,               // [C, 4, 4]
        const at::optional<at::Tensor> viewmats1, // [C, 4, 4] optional for rolling shutter
        const at::Tensor Ks,                      // [C, 3, 3]
        const CameraModelType camera_model,
        // uncented transform
        const UnscentedTransformParameters ut_params,
        ShutterType rs_type,
        const at::optional<at::Tensor> radial_coeffs,     // [C, 6] or [C, 4] optional
        const at::optional<at::Tensor> tangential_coeffs, // [C, 2] optional
        const at::optional<at::Tensor> thin_prism_coeffs, // [C, 2] optional
        // intersections
        const at::Tensor tile_offsets, // [C, tile_height, tile_width]
        const at::Tensor flatten_ids,  // [n_isects]
        // forward outputs
        const at::Tensor render_alphas, // [C, image_height, image_width, 1]
        const at::Tensor last_ids,      // [C, image_height, image_width]
        // gradients of outputs
        const at::Tensor v_render_colors, // [C, image_height, image_width, 3]
        const at::Tensor v_render_alphas  // [C, image_height, image_width, 1]
    ) {
        DEVICE_GUARD(means);
        CHECK_INPUT(means);
        CHECK_INPUT(quats);
        CHECK_INPUT(scales);
        CHECK_INPUT(colors);
        CHECK_INPUT(opacities);
        CHECK_INPUT(tile_offsets);
        CHECK_INPUT(flatten_ids);
        CHECK_INPUT(render_alphas);
        CHECK_INPUT(last_ids);
        CHECK_INPUT(v_render_colors);
        CHECK_INPUT(v_render_alphas);
        if (backgrounds.has_value()) {
            CHECK_INPUT(backgrounds.value());
        }
        if (masks.has_value()) {
            CHECK_INPUT(masks.value());
        }

        uint32_t channels = colors.size(-1);

        at::Tensor v_means = at::zeros_like(means);
        at::Tensor v_quats = at::zeros_like(quats);
        at::Tensor v_scales = at::zeros_like(scales);
        at::Tensor v_colors = at::zeros_like(colors);
        at::Tensor v_opacities = at::zeros_like(opacities);

#define __LAUNCH_KERNEL__(N)                                      \
    case N:                                                       \
        launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel<N>( \
            means,                                                \
            quats,                                                \
            scales,                                               \
            colors,                                               \
            opacities,                                            \
            backgrounds,                                          \
            masks,                                                \
            image_width,                                          \
            image_height,                                         \
            tile_size,                                            \
            viewmats0,                                            \
            viewmats1,                                            \
            Ks,                                                   \
            camera_model,                                         \
            ut_params,                                            \
            rs_type,                                              \
            radial_coeffs,                                        \
            tangential_coeffs,                                    \
            thin_prism_coeffs,                                    \
            tile_offsets,                                         \
            flatten_ids,                                          \
            render_alphas,                                        \
            last_ids,                                             \
            v_render_colors,                                      \
            v_render_alphas,                                      \
            v_means,                                              \
            v_quats,                                              \
            v_scales,                                             \
            v_colors,                                             \
            v_opacities);                                         \
        break;

        // TODO: an optimization can be done by passing the actual number of
        // channels into the kernel functions and avoid necessary global memory
        // writes. This requires moving the channel padding from python to C side.
        switch (channels) {
            __LAUNCH_KERNEL__(1)
            __LAUNCH_KERNEL__(2)
            __LAUNCH_KERNEL__(3)
            __LAUNCH_KERNEL__(4)
            __LAUNCH_KERNEL__(5)
            __LAUNCH_KERNEL__(8)
            __LAUNCH_KERNEL__(9)
            __LAUNCH_KERNEL__(16)
            __LAUNCH_KERNEL__(17)
            __LAUNCH_KERNEL__(32)
            __LAUNCH_KERNEL__(33)
            __LAUNCH_KERNEL__(64)
            __LAUNCH_KERNEL__(65)
            __LAUNCH_KERNEL__(128)
            __LAUNCH_KERNEL__(129)
            __LAUNCH_KERNEL__(256)
            __LAUNCH_KERNEL__(257)
            __LAUNCH_KERNEL__(512)
            __LAUNCH_KERNEL__(513)
        default:
            AT_ERROR("Unsupported number of channels: ", channels);
        }
#undef __LAUNCH_KERNEL__

        return std::make_tuple(
            v_means, v_quats, v_scales, v_colors, v_opacities);
    }

} // namespace gsplat
