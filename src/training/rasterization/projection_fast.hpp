#pragma once
#include <optional>
#include <torch/torch.h>
#include "rasterization/rasterizer_autograd.hpp"

namespace gs {
    namespace training {

        /// @brief Fast projection shim (GUT backend) with optional distortion.
        /// @return pair {radii [N,2], means2d [N,2]} for single-camera; squeezes [C,N,*] if C==1.
        inline std::pair<torch::Tensor, torch::Tensor> ProjectFast(
            const torch::Tensor& means3D,
            const torch::Tensor& quats,
            const torch::Tensor& scales,
            const torch::Tensor& opacities,
            torch::Tensor viewmat, // [C,4,4] or [4,4]
            torch::Tensor K,       // [C,3,3] or [3,3]
            int image_w, int image_h,
            float eps2d, float near_plane, float far_plane,
            float radius_clip, float scaling_modifier,
            gsplat::CameraModelType camera_model,
            std::optional<torch::Tensor> radial = std::nullopt,     // [...,C,4..6]
            std::optional<torch::Tensor> tangential = std::nullopt, // [...,C,2]
            std::optional<torch::Tensor> thin_prism = std::nullopt  // [...,C,4]
        ) {
            if (viewmat.dim() == 2)
                viewmat = viewmat.unsqueeze(0);
            if (K.dim() == 2)
                K = K.unsqueeze(0);

            auto pad_last = [](torch::Tensor t, int want) {
                if (!t.defined())
                    return t;
                t = t.to(torch::kCUDA).contiguous();
                const int have = static_cast<int>(t.size(-1));
                if (have < want) {
                    t = torch::nn::functional::pad(
                        t,
                        torch::nn::functional::PadFuncOptions({0, want - have})
                            .mode(torch::kConstant)
                            .value(0));
                }
                return t;
            };
            if (radial)
                radial = pad_last(*radial, 4);
            if (tangential)
                tangential = pad_last(*tangential, 2);
            if (thin_prism)
                thin_prism = pad_last(*thin_prism, 4);

            GUTProjectionSettings st{
                image_w, image_h, eps2d,
                near_plane, far_plane,
                radius_clip, scaling_modifier,
                camera_model};

            auto out = fully_fused_projection_with_ut(
                means3D, quats, scales, opacities,
                viewmat, K,
                radial, tangential, thin_prism,
                st,
                UnscentedTransformParameters{} // default UT
            );

            torch::Tensor radii = out[0];
            torch::Tensor means2d = out[1];
            if (radii.dim() == 3 && radii.size(0) == 1)
                radii = radii.squeeze(0);
            if (means2d.dim() == 3 && means2d.size(0) == 1)
                means2d = means2d.squeeze(0);
            return {radii.contiguous(), means2d.contiguous()};
        }

    } // namespace training
} // namespace gs
