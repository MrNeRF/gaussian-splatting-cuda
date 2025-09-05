/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef _WIN32
#define NOMINMAX
#endif

#include "sogs.hpp"
#include "core/logger.hpp"
#include <archive.h>
#include <archive_entry.h>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <torch/torch.h>
#include <unordered_map>
#include <webp/decode.h>

namespace gs::loader {

    namespace {

#ifdef _WIN32
        using ssize_t = std::ptrdiff_t;
#endif

        // Identity layout matching the exporter
        int identity_layout(int index, [[maybe_unused]] int width) {
            return index;
        }

        // SH coefficient counts per degree
        constexpr int SH_COEFFS[] = {0, 3, 8, 15};

        float inverse_log_transform(float value) {
            float sign = value >= 0 ? 1.0f : -1.0f;
            return sign * (std::exp(std::abs(value)) - 1.0f);
        }

        std::array<float, 4> unpack_quaternion(
            uint8_t a, uint8_t b, uint8_t c, uint8_t type) {

            // Determine which component was largest during packing
            int largest = type - 252; // 0=w, 1=x, 2=y, 3=z
            if (largest < 0 || largest > 3) {
                LOG_WARN("Invalid quaternion type: {}, defaulting to w", type);
                largest = 0;
            }

            // Unpack the three stored components with sqrt(2) scaling
            constexpr float sqrt2 = 1.41421356237f;
            float v0 = (a / 255.0f - 0.5f) * sqrt2;
            float v1 = (b / 255.0f - 0.5f) * sqrt2;
            float v2 = (c / 255.0f - 0.5f) * sqrt2;

            // Reconstruct the largest component
            float largest_val = std::sqrt(std::clamp(1.0f - (v0 * v0 + v1 * v1 + v2 * v2), 0.0f, 1.0f));

            // Build the quaternion [x, y, z, w] based on what was packed
            std::array<float, 4> quat; // [x, y, z, w]

            if (largest == 0) {
                // w was largest, stored [x, y, z]
                quat[0] = v0;          // x
                quat[1] = v1;          // y
                quat[2] = v2;          // z
                quat[3] = largest_val; // w
            } else if (largest == 1) {
                // x was largest, stored [w, y, z]
                quat[0] = largest_val; // x
                quat[1] = v1;          // y
                quat[2] = v2;          // z
                quat[3] = v0;          // w
            } else if (largest == 2) {
                // y was largest, stored [w, x, z]
                quat[0] = v1;          // x
                quat[1] = largest_val; // y
                quat[2] = v2;          // z
                quat[3] = v0;          // w
            } else {                   // largest == 3
                // z was largest, stored [w, x, y]
                quat[0] = v1;          // x
                quat[1] = v2;          // y
                quat[2] = largest_val; // z
                quat[3] = v0;          // w
            }

            // Normalize quaternion
            float len = std::sqrt(quat[0] * quat[0] + quat[1] * quat[1] +
                                  quat[2] * quat[2] + quat[3] * quat[3]);
            if (len > 0) {
                for (auto& v : quat)
                    v /= len;
            }

            return quat; // Returns [x, y, z, w]
        }

        std::expected<std::vector<uint8_t>, std::string> decode_webp(
            const uint8_t* data, size_t size, int& width, int& height) {

            if (!data || size == 0) {
                return std::unexpected("Invalid WebP data");
            }

            // Get image info
            if (!WebPGetInfo(data, size, &width, &height)) {
                return std::unexpected("Failed to get WebP info");
            }

            // Decode RGBA
            std::vector<uint8_t> rgba(width * height * 4);
            uint8_t* result = WebPDecodeRGBA(data, size, &width, &height);

            if (!result) {
                return std::unexpected("Failed to decode WebP image");
            }

            // Copy to vector
            std::memcpy(rgba.data(), result, width * height * 4);
            WebPFree(result);

            return rgba;
        }

        struct SogMetadata {
            int version = 0;
            int count = 0;
            int width = 0;
            int height = 0;

            // Position bounds
            std::vector<float> means_mins;
            std::vector<float> means_maxs;
            std::vector<std::string> means_files;

            // Scale codebook
            std::vector<float> scales_codebook;
            std::vector<std::string> scales_files;

            // Quaternion files
            std::vector<std::string> quats_files;

            // Color codebook
            std::vector<float> sh0_codebook;
            std::vector<std::string> sh0_files;

            // Optional spherical harmonics
            struct SHData {
                std::vector<float> codebook;
                int palette_size = 0;
                int bands = 0;
                int coeffs = 0;
                std::vector<std::string> files;
            };
            std::optional<SHData> shN;
        };

        std::expected<SogMetadata, std::string> parse_metadata(
            const std::string& json_str) {

            try {
                auto json = nlohmann::json::parse(json_str);
                SogMetadata meta;

                // Basic fields
                meta.version = json.at("version").get<int>();
                meta.count = json.at("count").get<int>();

                // Width and height might not be in meta.json (calculated from texture)
                if (json.contains("width")) {
                    meta.width = json["width"].get<int>();
                }
                if (json.contains("height")) {
                    meta.height = json["height"].get<int>();
                }

                // Position bounds
                auto means = json.at("means");
                meta.means_mins = means.at("mins").get<std::vector<float>>();
                meta.means_maxs = means.at("maxs").get<std::vector<float>>();
                meta.means_files = means.at("files").get<std::vector<std::string>>();

                // Scales
                auto scales = json.at("scales");
                meta.scales_codebook = scales.at("codebook").get<std::vector<float>>();
                meta.scales_files = scales.at("files").get<std::vector<std::string>>();

                // Quaternions
                auto quats = json.at("quats");
                meta.quats_files = quats.at("files").get<std::vector<std::string>>();

                // Colors
                auto sh0 = json.at("sh0");
                meta.sh0_codebook = sh0.at("codebook").get<std::vector<float>>();
                meta.sh0_files = sh0.at("files").get<std::vector<std::string>>();

                // Optional spherical harmonics
                if (json.contains("shN")) {
                    auto shN = json.at("shN");
                    SogMetadata::SHData sh_data;
                    sh_data.codebook = shN.at("codebook").get<std::vector<float>>();
                    sh_data.files = shN.at("files").get<std::vector<std::string>>();

                    // Optional fields
                    if (shN.contains("palette_size")) {
                        sh_data.palette_size = shN["palette_size"].get<int>();
                    }
                    if (shN.contains("bands")) {
                        sh_data.bands = shN["bands"].get<int>();
                    }
                    if (shN.contains("coeffs")) {
                        sh_data.coeffs = shN["coeffs"].get<int>();
                    }

                    meta.shN = sh_data;
                }

                return meta;

            } catch (const std::exception& e) {
                return std::unexpected(std::format("Failed to parse metadata: {}", e.what()));
            }
        }

        std::expected<SplatData, std::string> reconstruct_splat_data(
            const SogMetadata& meta,
            const std::unordered_map<std::string, std::vector<uint8_t>>& images) {

            const int num_splats = meta.count;

            // If width/height not in metadata, calculate from count
            int width = meta.width;
            int height = meta.height;
            if (width == 0 || height == 0) {
                width = ((int)std::ceil(std::sqrt(num_splats) / 4.0)) * 4;
                height = ((int)std::ceil(num_splats / (float)width / 4.0)) * 4;
            }

            LOG_DEBUG("Reconstructing {} splats from {}x{} textures", num_splats, width, height);

            // Create output tensors
            torch::Tensor means = torch::zeros({num_splats, 3}, torch::kFloat32);
            torch::Tensor scales = torch::zeros({num_splats, 3}, torch::kFloat32);
            torch::Tensor rotations = torch::zeros({num_splats, 4}, torch::kFloat32);
            torch::Tensor sh0 = torch::zeros({num_splats, 1, 3}, torch::kFloat32);
            torch::Tensor opacity = torch::zeros({num_splats, 1}, torch::kFloat32);

            // Accessors for fast CPU access
            auto means_acc = means.accessor<float, 2>();
            auto scales_acc = scales.accessor<float, 2>();
            auto rot_acc = rotations.accessor<float, 2>();
            auto sh0_acc = sh0.accessor<float, 3>();
            auto opacity_acc = opacity.accessor<float, 2>();

            // 1. Decode positions from means_l and means_u
            {
                auto it_l = images.find("means_l.webp");
                auto it_u = images.find("means_u.webp");

                if (it_l == images.end() || it_u == images.end()) {
                    return std::unexpected("Missing position textures");
                }

                const auto& means_l = it_l->second;
                const auto& means_u = it_u->second;

                for (int i = 0; i < num_splats; ++i) {
                    int ti = identity_layout(i, width) * 4;

                    // Reconstruct 16-bit values
                    uint16_t x16 = means_l[ti + 0] | (means_u[ti + 0] << 8);
                    uint16_t y16 = means_l[ti + 1] | (means_u[ti + 1] << 8);
                    uint16_t z16 = means_l[ti + 2] | (means_u[ti + 2] << 8);

                    // Normalize and inverse transform
                    float x_norm = x16 / 65535.0f;
                    float y_norm = y16 / 65535.0f;
                    float z_norm = z16 / 65535.0f;

                    float x_log = x_norm * (meta.means_maxs[0] - meta.means_mins[0]) + meta.means_mins[0];
                    float y_log = y_norm * (meta.means_maxs[1] - meta.means_mins[1]) + meta.means_mins[1];
                    float z_log = z_norm * (meta.means_maxs[2] - meta.means_mins[2]) + meta.means_mins[2];

                    means_acc[i][0] = inverse_log_transform(x_log);
                    means_acc[i][1] = inverse_log_transform(y_log);
                    means_acc[i][2] = inverse_log_transform(z_log);
                }
            }

            // 2. Decode quaternions
            {
                auto it = images.find("quats.webp");
                if (it == images.end()) {
                    return std::unexpected("Missing quaternion texture");
                }

                const auto& quats = it->second;

                for (int i = 0; i < num_splats; ++i) {
                    int ti = identity_layout(i, width) * 4;

                    auto quat = unpack_quaternion(
                        quats[ti + 0],
                        quats[ti + 1],
                        quats[ti + 2],
                        quats[ti + 3]);

                    // unpack_quaternion returns [x, y, z, w]
                    // Store as [w, x, y, z] for SplatData format
                    rot_acc[i][0] = quat[3]; // w
                    rot_acc[i][1] = quat[0]; // x
                    rot_acc[i][2] = quat[1]; // y
                    rot_acc[i][3] = quat[2]; // z
                }
            }

            // 3. Decode scales
            {
                auto it = images.find("scales.webp");
                if (it == images.end()) {
                    return std::unexpected("Missing scales texture");
                }

                const auto& scales_img = it->second;

                for (int i = 0; i < num_splats; ++i) {
                    int ti = identity_layout(i, width) * 4;

                    // Get indices and validate
                    uint8_t idx0 = scales_img[ti + 0];
                    uint8_t idx1 = scales_img[ti + 1];
                    uint8_t idx2 = scales_img[ti + 2];

                    // Ensure indices are within codebook bounds
                    if (idx0 >= meta.scales_codebook.size() ||
                        idx1 >= meta.scales_codebook.size() ||
                        idx2 >= meta.scales_codebook.size()) {
                        LOG_ERROR("Scale codebook index out of bounds: {}, {}, {} (codebook size: {})",
                                  idx0, idx1, idx2, meta.scales_codebook.size());
                        return std::unexpected("Invalid scale codebook index");
                    }

                    // Look up from codebook (already in log space)
                    scales_acc[i][0] = meta.scales_codebook[idx0];
                    scales_acc[i][1] = meta.scales_codebook[idx1];
                    scales_acc[i][2] = meta.scales_codebook[idx2];
                }
            }

            // 4. Decode colors and opacity
            {
                auto it = images.find("sh0.webp");
                if (it == images.end()) {
                    return std::unexpected("Missing color texture");
                }

                const auto& sh0_img = it->second;

                for (int i = 0; i < num_splats; ++i) {
                    int ti = identity_layout(i, width) * 4;

                    // Get indices and validate
                    uint8_t idx0 = sh0_img[ti + 0];
                    uint8_t idx1 = sh0_img[ti + 1];
                    uint8_t idx2 = sh0_img[ti + 2];

                    // Ensure indices are within codebook bounds
                    if (idx0 >= meta.sh0_codebook.size() ||
                        idx1 >= meta.sh0_codebook.size() ||
                        idx2 >= meta.sh0_codebook.size()) {
                        LOG_ERROR("Color codebook index out of bounds: {}, {}, {} (codebook size: {})",
                                  idx0, idx1, idx2, meta.sh0_codebook.size());
                        return std::unexpected("Invalid color codebook index");
                    }

                    // Look up colors from codebook
                    sh0_acc[i][0][0] = meta.sh0_codebook[idx0];
                    sh0_acc[i][0][1] = meta.sh0_codebook[idx1];
                    sh0_acc[i][0][2] = meta.sh0_codebook[idx2];

                    // Decode opacity (inverse sigmoid)
                    float opacity_norm = sh0_img[ti + 3] / 255.0f;
                    // Clamp with a safer epsilon to prevent infinity
                    opacity_norm = std::clamp(opacity_norm, 1e-5f, 1.0f - 1e-5f);
                    opacity_acc[i][0] = std::log(opacity_norm / (1.0f - opacity_norm));
                }
            }

            // 5. Decode spherical harmonics if present
            torch::Tensor shN;
            if (meta.shN.has_value()) {
                const auto& sh_meta = meta.shN.value();

                auto it_centroids = images.find("shN_centroids.webp");
                auto it_labels = images.find("shN_labels.webp");

                if (it_centroids != images.end() && it_labels != images.end()) {
                    const auto& centroids_img = it_centroids->second;
                    const auto& labels_img = it_labels->second;

                    // Determine SH configuration
                    int sh_degree = sh_meta.bands > 0 ? sh_meta.bands : (sh_meta.coeffs == 3 ? 1 : sh_meta.coeffs == 8 ? 2
                                                                                               : sh_meta.coeffs == 15  ? 3
                                                                                                                       : 0);

                    int num_coeffs = SH_COEFFS[sh_degree];
                    int palette_size = sh_meta.palette_size > 0 ? sh_meta.palette_size : centroids_img.size() / (64 * num_coeffs * 4);

                    LOG_DEBUG("Decoding SH: degree={}, coeffs={}, palette_size={}",
                              sh_degree, num_coeffs, palette_size);

                    // Create SH tensor
                    shN = torch::zeros({num_splats, num_coeffs, 3}, torch::kFloat32);
                    auto shN_acc = shN.accessor<float, 3>();

                    // Decode centroids from texture
                    std::vector<std::vector<float>> centroids(palette_size);
                    for (int i = 0; i < palette_size; ++i) {
                        centroids[i].resize(num_coeffs * 3);

                        for (int j = 0; j < num_coeffs; ++j) {
                            int pixel_idx = i * num_coeffs + j;

                            // Decode from codebook
                            for (int c = 0; c < 3; ++c) {
                                uint8_t idx = centroids_img[pixel_idx * 4 + c];

                                // Validate index
                                if (idx >= sh_meta.codebook.size()) {
                                    LOG_ERROR("SH codebook index out of bounds: {} (codebook size: {})",
                                              idx, sh_meta.codebook.size());
                                    return std::unexpected("Invalid SH codebook index");
                                }

                                // Band-major ordering
                                int coeff_idx = j + c * num_coeffs;
                                centroids[i][coeff_idx] = sh_meta.codebook[idx];
                            }
                        }
                    }

                    // Apply labels
                    for (int i = 0; i < num_splats; ++i) {
                        int ti = identity_layout(i, width) * 4;

                        // Reconstruct label from 16-bit value
                        int label = labels_img[ti + 0] | (labels_img[ti + 1] << 8);

                        if (label < palette_size) {
                            const auto& centroid = centroids[label];

                            // Unpack in band-major order
                            for (int c = 0; c < 3; ++c) {
                                for (int j = 0; j < num_coeffs; ++j) {
                                    shN_acc[i][j][c] = centroid[j + c * num_coeffs];
                                }
                            }
                        }
                    }
                }
            }

            // Move tensors to CUDA
            means = means.to(torch::kCUDA);
            scales = scales.to(torch::kCUDA);
            rotations = rotations.to(torch::kCUDA);
            sh0 = sh0.to(torch::kCUDA);
            opacity = opacity.to(torch::kCUDA);
            if (shN.defined()) {
                shN = shN.to(torch::kCUDA);
            } else {
                shN = torch::zeros({num_splats, 0, 3}, torch::kFloat32).to(torch::kCUDA);
            }

            // Create SplatData
            LOG_INFO("Successfully reconstructed {} splats", num_splats);

            return SplatData(
                meta.shN.has_value() ? meta.shN->bands : 0,
                means,
                sh0,
                shN,
                scales,
                rotations,
                opacity,
                1.0f // scene_scale
            );
        }

        std::expected<SplatData, std::string> read_sog_bundle(
            const std::filesystem::path& path) {

            LOG_INFO("Reading SOG bundle: {}", path.string());

            struct archive* a = archive_read_new();
            archive_read_support_format_zip(a);
            archive_read_support_filter_all(a);

            if (archive_read_open_filename(a, path.string().c_str(), 10240) != ARCHIVE_OK) {
                archive_read_free(a);
                return std::unexpected(std::format("Failed to open archive: {}",
                                                   archive_error_string(a)));
            }

            struct archive_entry* entry;
            std::string metadata_json;
            std::unordered_map<std::string, std::vector<uint8_t>> images;

            // Read all files from archive
            while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
                std::string filename = archive_entry_pathname(entry);
                size_t size = archive_entry_size(entry);

                LOG_DEBUG("Reading {} ({} bytes)", filename, size);

                std::vector<uint8_t> data(size);
                ssize_t r = archive_read_data(a, data.data(), size);

                if (r != static_cast<ssize_t>(size)) {
                    archive_read_free(a);
                    return std::unexpected(std::format("Failed to read {} from archive", filename));
                }

                if (filename == "meta.json") {
                    metadata_json = std::string(data.begin(), data.end());
                } else if (filename.ends_with(".webp")) {
                    // Decode WebP
                    int width, height;
                    auto decoded = decode_webp(data.data(), data.size(), width, height);
                    if (!decoded) {
                        archive_read_free(a);
                        return std::unexpected(decoded.error());
                    }
                    images[filename] = std::move(decoded.value());
                }
            }

            archive_read_free(a);

            if (metadata_json.empty()) {
                return std::unexpected("Missing meta.json in archive");
            }

            // Parse metadata
            auto meta_result = parse_metadata(metadata_json);
            if (!meta_result) {
                return std::unexpected(meta_result.error());
            }

            // Reconstruct SplatData
            return reconstruct_splat_data(meta_result.value(), images);
        }

        std::expected<SplatData, std::string> read_sog_directory(
            const std::filesystem::path& path) {

            LOG_INFO("Reading SOG from directory: {}", path.string());

            // Read meta.json
            auto meta_path = path / "meta.json";
            if (!std::filesystem::exists(meta_path)) {
                return std::unexpected("Missing meta.json");
            }

            std::ifstream meta_file(meta_path);
            if (!meta_file) {
                return std::unexpected("Failed to open meta.json");
            }

            std::stringstream buffer;
            buffer << meta_file.rdbuf();

            auto meta_result = parse_metadata(buffer.str());
            if (!meta_result) {
                return std::unexpected(meta_result.error());
            }

            auto& meta = meta_result.value();
            std::unordered_map<std::string, std::vector<uint8_t>> images;

            // Helper to read and decode WebP files
            auto read_webp = [&](const std::string& filename) -> bool {
                auto file_path = path / filename;

                // Also check with .webp extension if not present
                if (!std::filesystem::exists(file_path) && !filename.ends_with(".webp")) {
                    file_path = path / (filename + ".webp");
                }

                if (!std::filesystem::exists(file_path)) {
                    LOG_ERROR("Missing file: {}", file_path.string());
                    return false;
                }

                // Read file
                std::ifstream file(file_path, std::ios::binary);
                if (!file) {
                    LOG_ERROR("Failed to open: {}", file_path.string());
                    return false;
                }

                file.seekg(0, std::ios::end);
                size_t size = file.tellg();
                file.seekg(0, std::ios::beg);

                std::vector<uint8_t> data(size);
                file.read(reinterpret_cast<char*>(data.data()), size);

                // Decode WebP
                int width, height;
                auto decoded = decode_webp(data.data(), data.size(), width, height);
                if (!decoded) {
                    LOG_ERROR("Failed to decode {}: {}", filename, decoded.error());
                    return false;
                }

                images[filename] = std::move(decoded.value());
                return true;
            };

            // Read all required files
            for (const auto& file : meta.means_files) {
                if (!read_webp(file))
                    return std::unexpected("Failed to read " + file);
            }
            for (const auto& file : meta.scales_files) {
                if (!read_webp(file))
                    return std::unexpected("Failed to read " + file);
            }
            for (const auto& file : meta.quats_files) {
                if (!read_webp(file))
                    return std::unexpected("Failed to read " + file);
            }
            for (const auto& file : meta.sh0_files) {
                if (!read_webp(file))
                    return std::unexpected("Failed to read " + file);
            }

            // Read optional SH files
            if (meta.shN.has_value()) {
                for (const auto& file : meta.shN->files) {
                    if (!read_webp(file)) {
                        LOG_WARN("Failed to read SH file {}, continuing without SH", file);
                        meta.shN.reset();
                        break;
                    }
                }
            }

            // Reconstruct SplatData
            return reconstruct_splat_data(meta, images);
        }

    } // anonymous namespace

    std::expected<SplatData, std::string> load_sog(const std::filesystem::path& path) {
        LOG_TIMER("SOG File Loading");

        if (!std::filesystem::exists(path)) {
            std::string error_msg = std::format("SOG file/directory does not exist: {}", path.string());
            LOG_ERROR("{}", error_msg);
            throw std::runtime_error(error_msg);
        }

        // Check if it's a .sog bundle
        if (path.extension() == ".sog") {
            return read_sog_bundle(path);
        }

        // Check if it's a meta.json file
        if (path.filename() == "meta.json") {
            return read_sog_directory(path.parent_path());
        }

        // Check if it's a directory
        if (std::filesystem::is_directory(path)) {
            return read_sog_directory(path);
        }

        return std::unexpected(std::format("Unknown SOG format: {}", path.string()));
    }

} // namespace gs::loader