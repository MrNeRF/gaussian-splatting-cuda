#include "gs_rasterizer.hpp"
#include "rasterization_api.h"

namespace gs::rendering {

    struct FastGSSettings {
        torch::Tensor w2c;
        torch::Tensor cam_position;
        int active_sh_bases;
        int width;
        int height;
        float focal_x;
        float focal_y;
        float center_x;
        float center_y;
        float near_plane;
        float far_plane;
    };

    static std::tuple<torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& means,                               // [N, 3]
        const torch::Tensor& scales_raw,                          // [N, 3]
        const torch::Tensor& rotations_raw,                       // [N, 4]
        const torch::Tensor& opacities_raw,                       // [N, 1]
        const torch::Tensor& sh_coefficients_0,                   // [N, 1, 3]
        const torch::Tensor& sh_coefficients_rest,                // [C, B-1, 3]
        torch::Tensor& densification_info,                        // [2, N] or empty tensor
        const FastGSSettings& settings) { // rasterizer settings

        auto outputs = forward_wrapper(
            means,
            scales_raw,
            rotations_raw,
            opacities_raw,
            sh_coefficients_0,
            sh_coefficients_rest,
            settings.w2c,
            settings.cam_position,
            settings.active_sh_bases,
            settings.width,
            settings.height,
            settings.focal_x,
            settings.focal_y,
            settings.center_x,
            settings.center_y,
            settings.near_plane,
            settings.far_plane);

        auto image = std::get<0>(outputs);
        auto alpha = std::get<1>(outputs);
        auto per_primitive_buffers = std::get<2>(outputs);
        auto per_tile_buffers = std::get<3>(outputs);
        auto per_instance_buffers = std::get<4>(outputs);
        auto per_bucket_buffers = std::get<5>(outputs);
        int n_visible_primitives = std::get<6>(outputs);
        int n_instances = std::get<7>(outputs);
        int n_buckets = std::get<8>(outputs);
        int primitive_primitive_indices_selector = std::get<9>(outputs);
        int instance_primitive_indices_selector = std::get<10>(outputs);

        return {image, alpha};
    }

    using torch::indexing::None;
    using torch::indexing::Slice;

    torch::Tensor rasterize(
        Camera& viewpoint_camera,
        SplatData& gaussian_model,
        torch::Tensor& bg_color) {

        // Get camera parameters
        const int width = static_cast<int>(viewpoint_camera.image_width());
        const int height = static_cast<int>(viewpoint_camera.image_height());
        auto [fx, fy, cx, cy] = viewpoint_camera.get_intrinsics();

        // Get Gaussian parameters
        auto means = gaussian_model.means();
        auto raw_opacities = gaussian_model.opacity_raw();
        auto raw_scales = gaussian_model.scaling_raw();
        auto raw_rotations = gaussian_model.rotation_raw();
        auto sh0 = gaussian_model.sh0();
        auto shN = gaussian_model.shN();

        const int sh_degree = gaussian_model.get_active_sh_degree();
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);

        constexpr float near_plane = 0.01f;
        constexpr float far_plane = 1e10f;

        FastGSSettings settings;
        settings.w2c = viewpoint_camera.world_view_transform();
        settings.cam_position = viewpoint_camera.cam_position();
        settings.active_sh_bases = active_sh_bases;
        settings.width = width;
        settings.height = height;
        settings.focal_x = fx;
        settings.focal_y = fy;
        settings.center_x = cx;
        settings.center_y = cy;
        settings.near_plane = near_plane;
        settings.far_plane = far_plane;

        auto [image, alpha] = forward(
            means,
            raw_scales,
            raw_rotations,
            raw_opacities,
            sh0,
            shN,
            gaussian_model._densification_info,
            settings);


        // Manually blend the background since fast_rasterize does not do it
        torch::Tensor bg = bg_color.unsqueeze(1).unsqueeze(2); // [3, 1, 1]
        torch::Tensor blended_image = image + (1.0f - alpha) * bg;

        // Clamp the image to [0, 1] range for consistency with the original rasterize
        blended_image = torch::clamp(blended_image, 0.0f, 1.0f);

        return blended_image;
    }

} // namespace gs