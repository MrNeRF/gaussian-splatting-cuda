#include "core/gaussian_init.hpp"
#include "core/debug_utils.hpp"
#include "core/gaussian.hpp"
#include "core/mean_neighbor_dist.hpp"
#include "core/read_utils.hpp"
#include <torch/torch.h>

namespace gauss::init {

    // ────────────────────────────────────────────────────────────────────────────
    // Utility: map RGB in [0,1] to spherical-harmonics DC range
    // ────────────────────────────────────────────────────────────────────────────
    static inline torch::Tensor rgb_to_sh(const torch::Tensor& rgb) {
        constexpr float kInvSH = 0.28209479177387814f; // 1 / √(4π)
        return (rgb - 0.5f) / kInvSH;
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Heavy-weight tensor bootstrapper
    // ────────────────────────────────────────────────────────────────────────────
    InitTensors build_from_point_cloud(PointCloud& pcd, // ❶ non-const
                                       int max_sh_degree,
                                       float /*spatial_lr_scale*/) {
        InitTensors out;

        const auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
        const auto f32_cuda = f32.device(torch::kCUDA);
        const auto u8_cuda = torch::TensorOptions()
                                 .dtype(torch::kUInt8)
                                 .device(torch::kCUDA);

        // 1 ─ xyz ────────────────────────────────────────────────────────────────
        out.xyz = torch::from_blob(pcd._points.data(),
                                   {static_cast<int64_t>(pcd._points.size()), 3},
                                   f32) // still on CPU
                      .to(torch::kCUDA) // copy to GPU
                      .set_requires_grad(true);

        // 2 ─ scaling (log(σ)) ──────────────────────────────────────────────────
        auto nn_dist = torch::clamp_min(compute_mean_neighbor_distances(out.xyz),
                                        1e-7);
        out.scaling = torch::log(torch::sqrt(nn_dist))
                          .unsqueeze(-1)
                          .repeat({1, 3})
                          .to(f32_cuda) // already GPU + f32
                          .set_requires_grad(true);

        // 3 ─ rotation & opacity ────────────────────────────────────────────────
        out.rotation = torch::zeros({out.xyz.size(0), 4}, f32_cuda)
                           .index_put_({torch::indexing::Slice(), 0}, 1)
                           .set_requires_grad(true);

        out.opacity = inverse_sigmoid(0.5f * torch::ones({out.xyz.size(0), 1},
                                                         f32_cuda))
                          .set_requires_grad(true);

        // 4 ─ features (DC + rest) ──────────────────────────────────────────────
        auto rgb = torch::from_blob(pcd._colors.data(),
                                    {static_cast<int64_t>(pcd._colors.size()), 3},
                                    torch::TensorOptions().dtype(torch::kUInt8))
                       .to(f32) / // CPU → f32
                   255.f;

        auto fused_color = rgb_to_sh(rgb).to(torch::kCUDA);

        const int64_t feature_shape =
            static_cast<int64_t>(std::pow(max_sh_degree + 1, 2));

        auto features = torch::zeros({fused_color.size(0), 3, feature_shape},
                                     f32_cuda);

        // DC coefficients
        features.index_put_({torch::indexing::Slice(),
                             torch::indexing::Slice(/*0:3*/),
                             0},
                            fused_color);

        out.features_dc =
            features.index({torch::indexing::Slice(),
                            torch::indexing::Slice(),
                            torch::indexing::Slice(0, 1)})
                .transpose(1, 2)
                .contiguous()
                .set_requires_grad(true);

        out.features_rest =
            features.index({torch::indexing::Slice(),
                            torch::indexing::Slice(),
                            torch::indexing::Slice(1, torch::indexing::None)})
                .transpose(1, 2)
                .contiguous()
                .set_requires_grad(true);

        return out;
    }

} // namespace gauss::init
