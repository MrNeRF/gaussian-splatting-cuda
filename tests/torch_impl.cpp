// torch_impl.cpp - Reference implementations for testing
#include "torch_impl.hpp"
#include <cstring>

namespace reference {

    // Convert quaternion to rotation matrix
    torch::Tensor quat_to_rotmat(const torch::Tensor& quats) {
        // Normalize quaternions
        auto quats_norm = torch::nn::functional::normalize(
            quats, torch::nn::functional::NormalizeFuncOptions().dim(-1).p(2));

        auto w = quats_norm.select(-1, 0);
        auto x = quats_norm.select(-1, 1);
        auto y = quats_norm.select(-1, 2);
        auto z = quats_norm.select(-1, 3);

        // Build rotation matrix
        std::vector<torch::Tensor> R_components = {
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y)};

        auto R = torch::stack(R_components, -1);
        auto shape = quats.sizes().vec();
        shape.back() = 3;
        shape.push_back(3);
        return R.reshape(shape);
    }

    // Convert quaternion and scale to covariance and precision matrices
    std::tuple<torch::Tensor, torch::Tensor> quat_scale_to_covar_preci(
        const torch::Tensor& quats,  // [N, 4]
        const torch::Tensor& scales, // [N, 3]
        bool compute_covar,
        bool compute_preci,
        bool triu) {

        auto R = quat_to_rotmat(quats); // [N, 3, 3]
        torch::Tensor covars, precis;

        if (compute_covar) {
            // M = R * S (where S is diagonal matrix of scales)
            auto M = R * scales.unsqueeze(-2);           // [N, 3, 3]
            covars = torch::bmm(M, M.transpose(-1, -2)); // [N, 3, 3]

            if (triu) {
                // Convert to upper triangular format
                covars = covars.reshape({covars.size(0), 9});
                // Average symmetric elements
                covars = (covars.index({"...", torch::tensor({0, 1, 2, 4, 5, 8})}) +
                          covars.index({"...", torch::tensor({0, 3, 6, 4, 7, 8})})) /
                         2.0f;
            }
        }

        if (compute_preci) {
            // P = R * (1/S)
            auto P = R * (1.0f / scales).unsqueeze(-2);  // [N, 3, 3]
            precis = torch::bmm(P, P.transpose(-1, -2)); // [N, 3, 3]

            if (triu) {
                precis = precis.reshape({precis.size(0), 9});
                precis = (precis.index({"...", torch::tensor({0, 1, 2, 4, 5, 8})}) +
                          precis.index({"...", torch::tensor({0, 3, 6, 4, 7, 8})})) /
                         2.0f;
            }
        }

        return {covars, precis};
    }

    // Perspective projection
    std::tuple<torch::Tensor, torch::Tensor> persp_proj(
        const torch::Tensor& means,  // [C, N, 3]
        const torch::Tensor& covars, // [C, N, 3, 3]
        const torch::Tensor& Ks,     // [C, 3, 3]
        int width,
        int height) {

        const int C = means.size(0);
        const int N = means.size(1);

        auto tx = means.select(-1, 0); // [C, N]
        auto ty = means.select(-1, 1); // [C, N]
        auto tz = means.select(-1, 2); // [C, N]
        auto tz2 = tz * tz;

        auto fx = Ks.select(-2, 0).select(-1, 0).unsqueeze(-1); // [C, 1]
        auto fy = Ks.select(-2, 1).select(-1, 1).unsqueeze(-1); // [C, 1]
        auto cx = Ks.select(-2, 0).select(-1, 2).unsqueeze(-1); // [C, 1]
        auto cy = Ks.select(-2, 1).select(-1, 2).unsqueeze(-1); // [C, 1]

        auto tan_fovx = 0.5f * width / fx;
        auto tan_fovy = 0.5f * height / fy;

        auto lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
        auto lim_x_neg = cx / fx + 0.3f * tan_fovx;
        auto lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
        auto lim_y_neg = cy / fy + 0.3f * tan_fovy;

        tx = tz * torch::clamp(tx / tz, -lim_x_neg, lim_x_pos);
        ty = tz * torch::clamp(ty / tz, -lim_y_neg, lim_y_pos);

        auto O = torch::zeros({C, N}, means.options());
        auto J = torch::stack({fx / tz, O, -fx * tx / tz2,
                               O, fy / tz, -fy * ty / tz2},
                              -1)
                     .reshape({C, N, 2, 3});

        // Compute 2D covariance: J @ covars @ J^T
        auto cov2d = torch::einsum("...ij,...jk,...kl->...il", {J, covars, J.transpose(-1, -2)});

        // Compute 2D means: K[:2, :3] @ means
        auto means2d = torch::einsum("cij,cnj->cni", {Ks.slice(-2, 0, 2), means});
        means2d = means2d / tz.unsqueeze(-1);

        return {means2d, cov2d};
    }

    // World to camera transformation
    std::tuple<torch::Tensor, torch::Tensor> world_to_cam(
        const torch::Tensor& means,      // [N, 3]
        const torch::Tensor& covars,     // [N, 3, 3]
        const torch::Tensor& viewmats) { // [C, 4, 4]

        auto R = viewmats.slice(-2, 0, 3).slice(-1, 0, 3); // [C, 3, 3]
        auto t = viewmats.slice(-2, 0, 3).select(-1, 3);   // [C, 3]

        // Transform means: R @ means^T + t
        auto means_c = torch::einsum("cij,nj->cni", {R, means}) + t.unsqueeze(1);

        // Transform covariance: R @ covars @ R^T
        auto covars_c = torch::einsum("cij,njk,clk->cnil", {R, covars, R});

        return {means_c, covars_c};
    }

    // Fully fused projection
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    fully_fused_projection(
        const torch::Tensor& means,    // [N, 3]
        const torch::Tensor& covars,   // [N, 3, 3]
        const torch::Tensor& viewmats, // [C, 4, 4]
        const torch::Tensor& Ks,       // [C, 3, 3]
        int width,
        int height,
        float eps2d,
        float near_plane,
        float far_plane,
        bool calc_compensations,
        const std::string& camera_model) {

        // Transform to camera space
        auto [means_c, covars_c] = world_to_cam(means, covars, viewmats);

        // Project to 2D
        torch::Tensor means2d, covars2d;
        if (camera_model == "pinhole") {
            std::tie(means2d, covars2d) = persp_proj(means_c, covars_c, Ks, width, height);
        } else {
            throw std::runtime_error("Only pinhole camera model implemented in reference");
        }

        // Compute determinant before adding epsilon
        auto det_orig = covars2d.select(-2, 0).select(-1, 0) * covars2d.select(-2, 1).select(-1, 1) -
                        covars2d.select(-2, 0).select(-1, 1) * covars2d.select(-2, 1).select(-1, 0);

        // Add epsilon to diagonal
        auto eye = torch::eye(2, means.options()) * eps2d;
        covars2d = covars2d + eye;

        // Compute determinant after epsilon
        auto det = covars2d.select(-2, 0).select(-1, 0) * covars2d.select(-2, 1).select(-1, 1) -
                   covars2d.select(-2, 0).select(-1, 1) * covars2d.select(-2, 1).select(-1, 0);
        det = torch::clamp_min(det, 1e-10);

        // Compute compensations if requested
        torch::Tensor compensations;
        if (calc_compensations) {
            compensations = torch::sqrt(torch::clamp_min(det_orig / det, 0.0f));
        }

        // Compute conics (inverse of 2D covariance)
        auto conics = torch::stack({covars2d.select(-2, 1).select(-1, 1) / det,
                                    -(covars2d.select(-2, 0).select(-1, 1) + covars2d.select(-2, 1).select(-1, 0)) / 2.0f / det,
                                    covars2d.select(-2, 0).select(-1, 0) / det},
                                   -1);

        // Depths
        auto depths = means_c.select(-1, 2);

        // Compute radii
        auto radius_x = torch::ceil(3.33f * torch::sqrt(covars2d.select(-2, 0).select(-1, 0)));
        auto radius_y = torch::ceil(3.33f * torch::sqrt(covars2d.select(-2, 1).select(-1, 1)));
        auto radius = torch::stack({radius_x, radius_y}, -1);

        // Validity checks
        auto valid = (det > 0) & (depths > near_plane) & (depths < far_plane);
        radius = torch::where(valid.unsqueeze(-1), radius, torch::zeros_like(radius));

        // Inside checks
        auto inside = (means2d.select(-1, 0) + radius.select(-1, 0) > 0) &
                      (means2d.select(-1, 0) - radius.select(-1, 0) < width) &
                      (means2d.select(-1, 1) + radius.select(-1, 1) > 0) &
                      (means2d.select(-1, 1) - radius.select(-1, 1) < height);
        radius = torch::where(inside.unsqueeze(-1), radius, torch::zeros_like(radius));

        auto radii = radius.to(torch::kInt32);

        return {radii, means2d, depths, conics, compensations};
    }

    // Spherical harmonics evaluation
    torch::Tensor eval_sh_bases_fast(int basis_dim, const torch::Tensor& dirs) {
        // Support arbitrary batch dimensions like Python version
        auto result_shape = dirs.sizes().vec();
        result_shape.back() = basis_dim;
        auto result = torch::empty(result_shape, dirs.options());

        // Fill first coefficient
        result.index({"...", 0}).fill_(0.2820947917738781f);

        if (basis_dim <= 1)
            return result;

        // Extract x, y, z components
        auto x = dirs.index({"...", 0});
        auto y = dirs.index({"...", 1});
        auto z = dirs.index({"...", 2});

        float fTmpA = -0.48860251190292f;
        result.index({"...", 2}) = -fTmpA * z;
        result.index({"...", 3}) = fTmpA * x;
        result.index({"...", 1}) = fTmpA * y;

        if (basis_dim <= 4)
            return result;

        auto z2 = z * z;
        auto fTmpB = -1.092548430592079f * z;
        fTmpA = 0.5462742152960395f;
        auto fC1 = x * x - y * y;
        auto fS1 = 2 * x * y;
        result.index({"...", 6}) = 0.9461746957575601f * z2 - 0.3153915652525201f;
        result.index({"...", 7}) = fTmpB * x;
        result.index({"...", 5}) = fTmpB * y;
        result.index({"...", 8}) = fTmpA * fC1;
        result.index({"...", 4}) = fTmpA * fS1;

        if (basis_dim <= 9)
            return result;

        auto fTmpC = -2.285228997322329f * z2 + 0.4570457994644658f;
        fTmpB = 1.445305721320277f * z;
        fTmpA = -0.5900435899266435f;
        auto fC2 = x * fC1 - y * fS1;
        auto fS2 = x * fS1 + y * fC1;
        result.index({"...", 12}) = z * (1.865881662950577f * z2 - 1.119528997770346f);
        result.index({"...", 13}) = fTmpC * x;
        result.index({"...", 11}) = fTmpC * y;
        result.index({"...", 14}) = fTmpB * fC1;
        result.index({"...", 10}) = fTmpB * fS1;
        result.index({"...", 15}) = fTmpA * fC2;
        result.index({"...", 9}) = fTmpA * fS2;

        if (basis_dim <= 16)
            return result;

        auto fTmpD = z * (-4.683325804901025f * z2 + 2.007139630671868f);
        fTmpC = 3.31161143515146f * z2 - 0.47308734787878f;
        fTmpB = -1.770130769779931f * z;
        fTmpA = 0.6258357354491763f;
        auto fC3 = x * fC2 - y * fS2;
        auto fS3 = x * fS2 + y * fC2;
        result.index({"...", 20}) = 1.984313483298443f * z2 * (1.865881662950577f * z2 - 1.119528997770346f) +
                                    -1.006230589874905f * (0.9461746957575601f * z2 - 0.3153915652525201f);
        result.index({"...", 21}) = fTmpD * x;
        result.index({"...", 19}) = fTmpD * y;
        result.index({"...", 22}) = fTmpC * fC1;
        result.index({"...", 18}) = fTmpC * fS1;
        result.index({"...", 23}) = fTmpB * fC2;
        result.index({"...", 17}) = fTmpB * fS2;
        result.index({"...", 24}) = fTmpA * fC3;
        result.index({"...", 16}) = fTmpA * fS3;

        return result;
    }

    torch::Tensor spherical_harmonics(
        int degree,
        const torch::Tensor& dirs,     // [..., 3]
        const torch::Tensor& coeffs) { // [..., K, 3]

        // Normalize directions
        auto dirs_norm = torch::nn::functional::normalize(
            dirs, torch::nn::functional::NormalizeFuncOptions().dim(-1).p(2));

        int num_bases = (degree + 1) * (degree + 1);

        // Create bases tensor with same shape as coeffs[..., 0]
        auto bases_shape = coeffs.sizes().vec();
        bases_shape.pop_back(); // Remove last dimension (3)
        auto bases = torch::zeros(bases_shape, coeffs.options());

        // Only fill up to num_bases if K >= num_bases
        if (num_bases > 0 && bases.size(-1) >= num_bases) {
            auto sh_bases = eval_sh_bases_fast(num_bases, dirs_norm);
            // Use index instead of slice for better compatibility
            bases.index({"...", torch::indexing::Slice(0, num_bases)}) = sh_bases;
        }

        // Compute colors: sum(bases * coeffs, dim=-2)
        return (bases.unsqueeze(-1) * coeffs).sum(-2);
    }

    // Tile intersection test
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> isect_tiles(
        const torch::Tensor& means2d, // [C, N, 2]
        const torch::Tensor& radii,   // [C, N, 2]
        const torch::Tensor& depths,  // [C, N]
        int tile_size,
        int tile_width,
        int tile_height,
        bool sort) {

        int C = means2d.size(0);
        int N = means2d.size(1);
        auto device = means2d.device();

        // Compute tiles per gaussian
        auto tile_means2d = means2d / tile_size;
        auto tile_radii = radii.to(torch::kFloat32) / tile_size;
        auto tile_mins = torch::floor(tile_means2d - tile_radii).to(torch::kInt32);
        auto tile_maxs = torch::ceil(tile_means2d + tile_radii).to(torch::kInt32);

        // Clamp to tile bounds
        tile_mins.select(-1, 0) = torch::clamp(tile_mins.select(-1, 0), 0, tile_width);
        tile_mins.select(-1, 1) = torch::clamp(tile_mins.select(-1, 1), 0, tile_height);
        tile_maxs.select(-1, 0) = torch::clamp(tile_maxs.select(-1, 0), 0, tile_width);
        tile_maxs.select(-1, 1) = torch::clamp(tile_maxs.select(-1, 1), 0, tile_height);

        auto tiles_per_gauss = (tile_maxs - tile_mins).prod(-1);
        tiles_per_gauss = tiles_per_gauss * (radii > 0).all(-1).to(torch::kInt32);

        // Count total intersections
        int64_t n_isects = tiles_per_gauss.sum().item<int64_t>();

        auto isect_ids = torch::empty({n_isects}, torch::TensorOptions().dtype(torch::kInt64).device(device));
        auto flatten_ids = torch::empty({n_isects}, torch::TensorOptions().dtype(torch::kInt32).device(device));

        // CPU implementation for simplicity (can be optimized)
        auto tiles_per_gauss_cpu = tiles_per_gauss.to(torch::kCPU);
        auto tile_mins_cpu = tile_mins.to(torch::kCPU);
        auto tile_maxs_cpu = tile_maxs.to(torch::kCPU);
        auto depths_cpu = depths.to(torch::kCPU);
        auto radii_cpu = radii.to(torch::kCPU);

        std::vector<int64_t> isect_ids_vec;
        std::vector<int32_t> flatten_ids_vec;

        int tile_n_bits = std::ceil(std::log2(tile_width * tile_height));

        for (int cam_id = 0; cam_id < C; ++cam_id) {
            for (int gauss_id = 0; gauss_id < N; ++gauss_id) {
                if (radii_cpu[cam_id][gauss_id][0].item<int32_t>() <= 0 ||
                    radii_cpu[cam_id][gauss_id][1].item<int32_t>() <= 0) {
                    continue;
                }

                int index = cam_id * N + gauss_id;
                float depth_f32 = depths_cpu[cam_id][gauss_id].item<float>();

                // Reinterpret float as int32
                int32_t depth_id;
                std::memcpy(&depth_id, &depth_f32, sizeof(float));
                int64_t depth_id_64 = static_cast<int64_t>(depth_id) & 0xFFFFFFFF;

                auto tile_min = tile_mins_cpu[cam_id][gauss_id];
                auto tile_max = tile_maxs_cpu[cam_id][gauss_id];

                for (int y = tile_min[1].item<int>(); y < tile_max[1].item<int>(); ++y) {
                    for (int x = tile_min[0].item<int>(); x < tile_max[0].item<int>(); ++x) {
                        int64_t tile_id = y * tile_width + x;
                        int64_t isect_id = (static_cast<int64_t>(cam_id) << (32 + tile_n_bits)) |
                                           (tile_id << 32) | depth_id_64;
                        isect_ids_vec.push_back(isect_id);
                        flatten_ids_vec.push_back(index);
                    }
                }
            }
        }

        // Copy back to GPU
        isect_ids = torch::from_blob(isect_ids_vec.data(),
                                     {static_cast<int64_t>(isect_ids_vec.size())},
                                     torch::kInt64)
                        .clone()
                        .to(device);
        flatten_ids = torch::from_blob(flatten_ids_vec.data(),
                                       {static_cast<int64_t>(flatten_ids_vec.size())},
                                       torch::kInt32)
                          .clone()
                          .to(device);

        if (sort) {
            auto sorted_indices = torch::argsort(isect_ids);
            isect_ids = isect_ids.index_select(0, sorted_indices);
            flatten_ids = flatten_ids.index_select(0, sorted_indices);
        }

        return {tiles_per_gauss, isect_ids, flatten_ids};
    }

} // namespace reference