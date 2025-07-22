#include "core/ply_loader.hpp"
#include "external/tinyply.hpp"
#include <format>
#include <fstream>
#include <print>

namespace gs {

    // Constants for PLY format
    namespace ply_constants {
        // Maximum number of SH coefficient components
        constexpr int MAX_DC_COMPONENTS = 48;
        constexpr int MAX_REST_COMPONENTS = 135;

        // Dimension requirements
        constexpr int COLOR_CHANNELS = 3;
        constexpr int POSITION_DIMS = 3;
        constexpr int SCALE_DIMS = 3;
        constexpr int QUATERNION_DIMS = 4;

        // Default values
        constexpr float DEFAULT_LOG_SCALE = -5.0f;
        constexpr float IDENTITY_QUATERNION_W = 1.0f;
        constexpr float SCENE_SCALE_FACTOR = 0.5f;

        // SH degree calculation
        constexpr int SH_DEGREE_3_REST_COEFFS = 15; // (4^2 - 1) = 15 for degree 3
        constexpr int SH_DEGREE_OFFSET = 1;

        // Property names
        constexpr const char* VERTEX_ELEMENT = "vertex";
        constexpr const char* POS_X = "x";
        constexpr const char* POS_Y = "y";
        constexpr const char* POS_Z = "z";
        constexpr const char* NORMAL_X = "nx";
        constexpr const char* NORMAL_Y = "ny";
        constexpr const char* NORMAL_Z = "nz";
        constexpr const char* OPACITY = "opacity";
        constexpr const char* DC_PREFIX = "f_dc_";
        constexpr const char* REST_PREFIX = "f_rest_";
        constexpr const char* SCALE_PREFIX = "scale_";
        constexpr const char* ROT_PREFIX = "rot_";
    } // namespace ply_constants

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
                if (e.name == ply_constants::VERTEX_ELEMENT) {
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
                positions = ply_file.request_properties_from_element(
                    ply_constants::VERTEX_ELEMENT,
                    {ply_constants::POS_X, ply_constants::POS_Y, ply_constants::POS_Z});

                normals = ply_file.request_properties_from_element(
                    ply_constants::VERTEX_ELEMENT,
                    {ply_constants::NORMAL_X, ply_constants::NORMAL_Y, ply_constants::NORMAL_Z});

                // SH coefficients - DC terms
                for (int i = 0; i < ply_constants::MAX_DC_COMPONENTS; ++i) {
                    try {
                        auto component = ply_file.request_properties_from_element(
                            ply_constants::VERTEX_ELEMENT,
                            {std::format("{}{}", ply_constants::DC_PREFIX, i)});
                        if (component) {
                            f_dc_components.push_back(component);
                        }
                    } catch (...) {
                        break;
                    }
                }

                // SH coefficients - rest terms
                for (int i = 0; i < ply_constants::MAX_REST_COMPONENTS; ++i) {
                    try {
                        auto component = ply_file.request_properties_from_element(
                            ply_constants::VERTEX_ELEMENT,
                            {std::format("{}{}", ply_constants::REST_PREFIX, i)});
                        if (component) {
                            f_rest_components.push_back(component);
                        }
                    } catch (...) {
                        break;
                    }
                }

                opacity = ply_file.request_properties_from_element(
                    ply_constants::VERTEX_ELEMENT, {ply_constants::OPACITY});

                // Scale components
                scale_0 = ply_file.request_properties_from_element(
                    ply_constants::VERTEX_ELEMENT, {std::format("{}0", ply_constants::SCALE_PREFIX)});
                scale_1 = ply_file.request_properties_from_element(
                    ply_constants::VERTEX_ELEMENT, {std::format("{}1", ply_constants::SCALE_PREFIX)});
                scale_2 = ply_file.request_properties_from_element(
                    ply_constants::VERTEX_ELEMENT, {std::format("{}2", ply_constants::SCALE_PREFIX)});

                // Rotation components
                rot_0 = ply_file.request_properties_from_element(
                    ply_constants::VERTEX_ELEMENT, {std::format("{}0", ply_constants::ROT_PREFIX)});
                rot_1 = ply_file.request_properties_from_element(
                    ply_constants::VERTEX_ELEMENT, {std::format("{}1", ply_constants::ROT_PREFIX)});
                rot_2 = ply_file.request_properties_from_element(
                    ply_constants::VERTEX_ELEMENT, {std::format("{}2", ply_constants::ROT_PREFIX)});
                rot_3 = ply_file.request_properties_from_element(
                    ply_constants::VERTEX_ELEMENT, {std::format("{}3", ply_constants::ROT_PREFIX)});
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
                             {static_cast<int64_t>(num_points), ply_constants::POSITION_DIMS},
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

                if (dc_dim % ply_constants::COLOR_CHANNELS != 0) {
                    return std::unexpected(std::format("f_dc dimension {} is not a multiple of {}",
                                                       dc_dim, ply_constants::COLOR_CHANNELS));
                }

                // Reshape from [N, dc_dim] to [N, 3, B0] then transpose to [N, B0, 3]
                int B0 = dc_dim / ply_constants::COLOR_CHANNELS;
                sh0 = f_dc.reshape({static_cast<int64_t>(num_points), ply_constants::COLOR_CHANNELS, B0})
                          .transpose(1, 2);
            } else {
                sh0 = torch::zeros({static_cast<int64_t>(num_points), 1, ply_constants::COLOR_CHANNELS}, options);
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

                if (rest_dim % ply_constants::COLOR_CHANNELS != 0) {
                    return std::unexpected(std::format("f_rest dimension {} is not a multiple of {}",
                                                       rest_dim, ply_constants::COLOR_CHANNELS));
                }

                // Reshape from [N, rest_dim] to [N, 3, Bn] then transpose to [N, Bn, 3]
                int Bn = rest_dim / ply_constants::COLOR_CHANNELS;
                shN = f_rest.reshape({static_cast<int64_t>(num_points), ply_constants::COLOR_CHANNELS, Bn})
                          .transpose(1, 2);
            } else {
                // Default: assume SH degree 3 -> 15 rest coefficients
                shN = torch::zeros({static_cast<int64_t>(num_points),
                                    ply_constants::SH_DEGREE_3_REST_COEFFS,
                                    ply_constants::COLOR_CHANNELS},
                                   options);
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
                scaling = torch::full({static_cast<int64_t>(num_points), ply_constants::SCALE_DIMS},
                                      ply_constants::DEFAULT_LOG_SCALE, options);
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
                rotation = torch::zeros({static_cast<int64_t>(num_points), ply_constants::QUATERNION_DIMS}, options);
                rotation.index_put_({torch::indexing::Slice(), 0}, ply_constants::IDENTITY_QUATERNION_W);
            }

            // Move everything to CUDA
            means = means.to(torch::kCUDA);
            sh0 = sh0.to(torch::kCUDA);
            shN = shN.to(torch::kCUDA);
            scaling = scaling.to(torch::kCUDA);
            rotation = rotation.to(torch::kCUDA);
            opacity_tensor = opacity_tensor.to(torch::kCUDA);

            // Calculate SH degree from shN dimensions
            int sh_degree = static_cast<int>(std::sqrt(shN.size(1) + ply_constants::SH_DEGREE_OFFSET)) -
                            ply_constants::SH_DEGREE_OFFSET;

            // Estimate scene scale from point cloud bounds
            auto min_vals = std::get<0>(means.min(0));
            auto max_vals = std::get<0>(means.max(0));

            std::println("Successfully loaded {} Gaussians with SH degree {}", num_points, sh_degree);

            return SplatData(
                sh_degree,
                means,
                sh0,
                shN,
                scaling,
                rotation,
                opacity_tensor,
                1.f);

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to load PLY file: {}", e.what()));
        }
    }

} // namespace gs