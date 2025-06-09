#pragma once

#include <string>
#include <torch/torch.h>
#include <tuple>

namespace reference {

    // Convert quaternion to rotation matrix
    torch::Tensor quat_to_rotmat(const torch::Tensor& quats);

    // Convert quaternion and scale to covariance and precision matrices
    std::tuple<torch::Tensor, torch::Tensor> quat_scale_to_covar_preci(
        const torch::Tensor& quats,
        const torch::Tensor& scales,
        bool compute_covar = true,
        bool compute_preci = true,
        bool triu = false);

    // Perspective projection
    std::tuple<torch::Tensor, torch::Tensor> persp_proj(
        const torch::Tensor& means,
        const torch::Tensor& covars,
        const torch::Tensor& Ks,
        int width,
        int height);

    // World to camera transformation
    std::tuple<torch::Tensor, torch::Tensor> world_to_cam(
        const torch::Tensor& means,
        const torch::Tensor& covars,
        const torch::Tensor& viewmats);

    // Fully fused projection
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    fully_fused_projection(
        const torch::Tensor& means,
        const torch::Tensor& covars,
        const torch::Tensor& viewmats,
        const torch::Tensor& Ks,
        int width,
        int height,
        float eps2d = 0.3f,
        float near_plane = 0.01f,
        float far_plane = 1e10f,
        bool calc_compensations = false,
        const std::string& camera_model = "pinhole");

    // Spherical harmonics evaluation
    torch::Tensor eval_sh_bases_fast(int basis_dim, const torch::Tensor& dirs);

    torch::Tensor spherical_harmonics(
        int degree,
        const torch::Tensor& dirs,
        const torch::Tensor& coeffs);

    // Tile intersection test
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> isect_tiles(
        const torch::Tensor& means2d,
        const torch::Tensor& radii,
        const torch::Tensor& depths,
        int tile_size,
        int tile_width,
        int tile_height,
        bool sort = true);

} // namespace reference