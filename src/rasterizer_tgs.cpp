#include "Ops.h"
#include <torch/torch.h>

#include "core/rasterizer.hpp" // For gs::RenderMode and gs::RenderOutput definitions
#include "core/rasterizer_tgs.hpp"
#include "core/rasterizer_tgs_autograd.hpp"

namespace tgs {

    gs::RenderOutput rasterize(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color,
        float scaling_modifier/*=1*/,
        bool packed/*=false*/,
        bool antialiased/*=false*/,
        gs::RenderMode render_mode) {

        // Ensure we don't use packed mode (not supported in this implementation)
        TORCH_CHECK(!packed, "Packed mode is not supported in this implementation");

        const int image_height = int(viewpoint_camera.image_height());
        const int image_width = int(viewpoint_camera.image_width());

        auto viewmat = viewpoint_camera.world_view_transform().to(torch::kCUDA);
        auto projmat = viewpoint_camera.projmat().to(torch::kCUDA);
        auto campos = viewpoint_camera.campos().to(torch::kCUDA);

        const float tanfovx = viewpoint_camera.tanfovx();
        const float tanfovy = viewpoint_camera.tanfovy();

        auto means3D = gaussian_model.get_means();
        auto opacities = gaussian_model.get_opacity();
        if (opacities.dim() == 2 && opacities.size(1) == 1) {
            opacities = opacities.squeeze(-1);
        }
        const auto scales = gaussian_model.get_scaling();
        const auto rotations = gaussian_model.get_rotation();
        const auto sh0 = gaussian_model.get_sh0();
        const auto shN = gaussian_model.get_shN();
        const int sh_degree = gaussian_model.get_active_sh_degree();

        auto settings = torch::tensor({
            scaling_modifier,
            (float)image_height, (float)image_width,
            (float)sh_degree,
            (float)antialiased,
            tanfovx, tanfovy},
            means3D.options()
        );

        const auto result = tgs::RasterizationFunction::apply(
            means3D,
            sh0, shN,
            /*colors_precomp=*/torch::empty({0}, torch::kFloat32),
            opacities,
            scales,
            rotations,
            /*cov3Ds_precomp=*/torch::empty({0}, torch::kFloat32),
            viewmat,
            projmat,
            bg_color,
            campos,
            settings
        );

        gs::RenderOutput out;
        out.image = result[0];
        out.depths = torch::empty({0}, torch::kFloat32);
        out.radii = result[1];
        out.visibility = (out.radii > 0);
        out.width  = image_width;
        out.height = image_height;
        return out;
        
    }

}