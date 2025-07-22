#include "core/ply_loader.hpp"
#include "external/tinyply.hpp"
#include <format>
#include <fstream>
#include <print>

namespace gs {

    std::expected<SplatData, std::string> load_ply(const std::filesystem::path& filepath) {
        try {
            if (!std::filesystem::exists(filepath)) {
                return std::unexpected(std::format("PLY file does not exist: {}", filepath.string()));
            }

            std::ifstream file_stream(filepath, std::ios::binary);
            if (!file_stream) {
                return std::unexpected(std::format("Failed to open PLY file: {}", filepath.string()));
            }

            tinyply::PlyFile ply_file;
            ply_file.parse_header(file_stream);

            // Print what we found
            auto elements = ply_file.get_elements();
            size_t vertex_count = 0;
            for (const auto& e : elements) {
                if (e.name == "vertex") {
                    vertex_count = e.size;
                    break;
                }
            }
            std::println("PLY contains {} vertices", vertex_count);

            // Request vertex properties
            std::shared_ptr<tinyply::PlyData> positions, normals;
            std::vector<std::shared_ptr<tinyply::PlyData>> f_dc_components;
            std::vector<std::shared_ptr<tinyply::PlyData>> f_rest_components;
            std::shared_ptr<tinyply::PlyData> opacity;
            std::shared_ptr<tinyply::PlyData> scale_0, scale_1, scale_2;
            std::shared_ptr<tinyply::PlyData> rot_0, rot_1, rot_2, rot_3;

            // Try to request all properties we know about
            try {
                positions = ply_file.request_properties_from_element("vertex", {"x", "y", "z"});
                normals = ply_file.request_properties_from_element("vertex", {"nx", "ny", "nz"});

                // SH coefficients - DC terms (f_dc_0, f_dc_1, f_dc_2)
                for (int i = 0; i < 48; ++i) {
                    try {
                        auto component = ply_file.request_properties_from_element(
                            "vertex", {std::format("f_dc_{}", i)});
                        if (component) {
                            f_dc_components.push_back(component);
                        }
                    } catch (...) {
                        break;
                    }
                }

                // SH coefficients - rest terms (f_rest_0, f_rest_1, f_rest_2, ...)
                for (int i = 0; i < 135; ++i) {
                    try {
                        auto component = ply_file.request_properties_from_element(
                            "vertex", {std::format("f_rest_{}", i)});
                        if (component) {
                            f_rest_components.push_back(component);
                        }
                    } catch (...) {
                        break;
                    }
                }

                opacity = ply_file.request_properties_from_element("vertex", {"opacity"});
                scale_0 = ply_file.request_properties_from_element("vertex", {"scale_0"});
                scale_1 = ply_file.request_properties_from_element("vertex", {"scale_1"});
                scale_2 = ply_file.request_properties_from_element("vertex", {"scale_2"});
                rot_0 = ply_file.request_properties_from_element("vertex", {"rot_0"});
                rot_1 = ply_file.request_properties_from_element("vertex", {"rot_1"});
                rot_2 = ply_file.request_properties_from_element("vertex", {"rot_2"});
                rot_3 = ply_file.request_properties_from_element("vertex", {"rot_3"});
            } catch (const std::exception& e) {
                std::println("Note: Some properties not found ({}), continuing...", e.what());
            }

            // Read the data
            ply_file.read(file_stream);

            if (!positions || positions->count == 0) {
                return std::unexpected("No position data found in PLY file");
            }

            const size_t num_points = positions->count;
            std::println("Loading {} Gaussian splats", num_points);

            // Create tensors for each property
            auto options = torch::TensorOptions().dtype(torch::kFloat32);

            // Positions
            auto means = torch::from_blob(
                             positions->buffer.get(),
                             {static_cast<int64_t>(num_points), 3},
                             options)
                             .clone();

            // SH coefficients - following the Python code logic
            torch::Tensor sh0, shN;

            // Process DC components
            if (!f_dc_components.empty()) {
                // Load all f_dc components
                std::vector<torch::Tensor> dc_values;
                for (const auto& comp : f_dc_components) {
                    auto val = torch::from_blob(comp->buffer.get(),
                                                {static_cast<int64_t>(num_points)}, options)
                                   .clone();
                    dc_values.push_back(val);
                }

                // Stack to create [N, dc_dim]
                auto f_dc = torch::stack(dc_values, 1);
                int dc_dim = f_dc.size(1);

                if (dc_dim % 3 != 0) {
                    return std::unexpected(std::format("f_dc dimension {} is not a multiple of 3", dc_dim));
                }

                // Reshape from [N, dc_dim] to [N, 3, B0] then transpose to [N, B0, 3]
                int B0 = dc_dim / 3;
                sh0 = f_dc.reshape({static_cast<int64_t>(num_points), 3, B0}).transpose(1, 2);
            } else {
                sh0 = torch::zeros({static_cast<int64_t>(num_points), 1, 3}, options);
            }

            // Process rest components
            if (!f_rest_components.empty()) {
                // Load all f_rest components
                std::vector<torch::Tensor> rest_values;
                for (const auto& comp : f_rest_components) {
                    auto val = torch::from_blob(comp->buffer.get(),
                                                {static_cast<int64_t>(num_points)}, options)
                                   .clone();
                    rest_values.push_back(val);
                }

                // Stack to create [N, rest_dim]
                auto f_rest = torch::stack(rest_values, 1);
                int rest_dim = f_rest.size(1);

                if (rest_dim % 3 != 0) {
                    return std::unexpected(std::format("f_rest dimension {} is not a multiple of 3", rest_dim));
                }

                // Reshape from [N, rest_dim] to [N, 3, Bn] then transpose to [N, Bn, 3]
                int Bn = rest_dim / 3;
                shN = f_rest.reshape({static_cast<int64_t>(num_points), 3, Bn}).transpose(1, 2);
            } else {
                // Default: assume SH degree 3 -> 15 rest coefficients
                shN = torch::zeros({static_cast<int64_t>(num_points), 15, 3}, options);
            }

            // Opacity - raw values are stored (already inverse sigmoid)
            torch::Tensor opacity_tensor;
            if (opacity) {
                opacity_tensor = torch::from_blob(
                                     opacity->buffer.get(),
                                     {static_cast<int64_t>(num_points), 1},
                                     options)
                                     .clone();
            } else {
                opacity_tensor = torch::zeros({static_cast<int64_t>(num_points), 1}, options);
            }

            // Scaling - raw values are stored (already log scale)
            torch::Tensor scaling;
            if (scale_0 && scale_1 && scale_2) {
                auto s0 = torch::from_blob(scale_0->buffer.get(), {static_cast<int64_t>(num_points)}, options).clone();
                auto s1 = torch::from_blob(scale_1->buffer.get(), {static_cast<int64_t>(num_points)}, options).clone();
                auto s2 = torch::from_blob(scale_2->buffer.get(), {static_cast<int64_t>(num_points)}, options).clone();
                scaling = torch::stack({s0, s1, s2}, 1);
            } else {
                scaling = torch::full({static_cast<int64_t>(num_points), 3}, -5.0, options);
            }

            // Rotation quaternion - raw values are stored
            torch::Tensor rotation;
            if (rot_0 && rot_1 && rot_2 && rot_3) {
                auto r0 = torch::from_blob(rot_0->buffer.get(), {static_cast<int64_t>(num_points)}, options).clone();
                auto r1 = torch::from_blob(rot_1->buffer.get(), {static_cast<int64_t>(num_points)}, options).clone();
                auto r2 = torch::from_blob(rot_2->buffer.get(), {static_cast<int64_t>(num_points)}, options).clone();
                auto r3 = torch::from_blob(rot_3->buffer.get(), {static_cast<int64_t>(num_points)}, options).clone();
                rotation = torch::stack({r0, r1, r2, r3}, 1);
            } else {
                rotation = torch::zeros({static_cast<int64_t>(num_points), 4}, options);
                rotation.index_put_({torch::indexing::Slice(), 0}, 1.0); // Identity quaternion
            }

            // Move everything to CUDA
            means = means.to(torch::kCUDA).set_requires_grad(true);
            sh0 = sh0.to(torch::kCUDA).set_requires_grad(true);
            shN = shN.to(torch::kCUDA).set_requires_grad(true);
            scaling = scaling.to(torch::kCUDA).set_requires_grad(true);
            rotation = rotation.to(torch::kCUDA).set_requires_grad(true);
            opacity_tensor = opacity_tensor.to(torch::kCUDA).set_requires_grad(true);

            // Calculate SH degree from shN dimensions
            int sh_degree = static_cast<int>(std::sqrt(shN.size(1) + 1)) - 1;

            // Estimate scene scale from point cloud bounds
            auto min_vals = std::get<0>(means.min(0));
            auto max_vals = std::get<0>(means.max(0));
            float scene_scale = (max_vals - min_vals).max().item<float>() * 0.5f;

            std::println("Successfully loaded {} Gaussians with SH degree {}", num_points, sh_degree);
            std::println("Scene scale: {}", scene_scale);

            return SplatData(
                sh_degree,
                means,
                sh0,
                shN,
                scaling,
                rotation,
                opacity_tensor,
                scene_scale);

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to load PLY file: {}", e.what()));
        }
    }

} // namespace gs