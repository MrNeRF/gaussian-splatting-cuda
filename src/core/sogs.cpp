/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/sogs.hpp"
#include "core/logger.hpp"
#include "kernels/morton_encoding.cuh"
#include "kernels/kmeans.cuh"
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <webp/encode.h>
#include <archive.h>
#include <archive_entry.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <print>
#include <array>
#include <chrono>
#include <iomanip>
#include <cstring>

namespace gs::core {

namespace {

struct KMeansResult {
    torch::Tensor centroids;
    torch::Tensor labels;
};

// Wrapper for GPU/CPU k-means selection
KMeansResult cluster_1d(const torch::Tensor& data, int k, int iterations) {
    auto data_gpu = data.to(torch::kCUDA);
    auto [centroids, labels] = gs::cuda::kmeans_1d(data_gpu, k, iterations);
    return {centroids.cpu(), labels.cpu()};
}

KMeansResult cluster_nd(const torch::Tensor& data, int k, int iterations) {
    auto data_gpu = data.to(torch::kCUDA);
    auto [centroids, labels] = gs::cuda::kmeans(data_gpu, k, iterations);
    return {centroids.cpu(), labels.cpu()};
}

// Apply log transform for better quantization
float log_transform(float value) {
    return std::copysign(std::log(std::abs(value) + 1.0f), value);
}

// Pack quaternion into 8-bit values
std::array<uint8_t, 4> pack_quaternion(float w, float x, float y, float z) {
    // Normalize
    float len = std::sqrt(w*w + x*x + y*y + z*z);
    if (len > 0) {
        w /= len; x /= len; y /= len; z /= len;
    }

    // Find largest component (in absolute value)
    float max_val = std::abs(w);
    int max_idx = 0; // 0 = w, 1 = x, 2 = y, 3 = z

    if (std::abs(x) > max_val) {
        max_val = std::abs(x);
        max_idx = 1;
    }
    if (std::abs(y) > max_val) {
        max_val = std::abs(y);
        max_idx = 2;
    }
    if (std::abs(z) > max_val) {
        max_val = std::abs(z);
        max_idx = 3;
    }

    // Ensure largest component is positive
    if ((max_idx == 0 && w < 0) ||
        (max_idx == 1 && x < 0) ||
        (max_idx == 2 && y < 0) ||
        (max_idx == 3 && z < 0)) {
        w = -w; x = -x; y = -y; z = -z;
    }

    // Scale the quaternion components by sqrt(2)
    constexpr float sqrt2 = 1.41421356237f;
    w *= sqrt2;
    x *= sqrt2;
    y *= sqrt2;
    z *= sqrt2;

    // Pack the other 3 components based on which is largest
    std::array<uint8_t, 4> result;

    if (max_idx == 0) {
        // w is largest, store x, y, z
        result[0] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, (x * 0.5f + 0.5f) * 255.0f)));
        result[1] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, (y * 0.5f + 0.5f) * 255.0f)));
        result[2] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, (z * 0.5f + 0.5f) * 255.0f)));
    } else if (max_idx == 1) {
        // x is largest, store w, y, z
        result[0] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, (w * 0.5f + 0.5f) * 255.0f)));
        result[1] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, (y * 0.5f + 0.5f) * 255.0f)));
        result[2] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, (z * 0.5f + 0.5f) * 255.0f)));
    } else if (max_idx == 2) {
        // y is largest, store w, x, z
        result[0] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, (w * 0.5f + 0.5f) * 255.0f)));
        result[1] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, (x * 0.5f + 0.5f) * 255.0f)));
        result[2] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, (z * 0.5f + 0.5f) * 255.0f)));
    } else {
        // z is largest, store w, x, y
        result[0] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, (w * 0.5f + 0.5f) * 255.0f)));
        result[1] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, (x * 0.5f + 0.5f) * 255.0f)));
        result[2] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, (y * 0.5f + 0.5f) * 255.0f)));
    }

    // Store which component was largest
    result[3] = 252 + max_idx;

    return result;
}

// Write WebP image
bool write_webp_image(const std::filesystem::path& path,
                     const uint8_t* data,
                     int width,
                     int height,
                     int channels = 4) {

    if (!data || width <= 0 || height <= 0) {
        LOG_ERROR("Invalid input to write_webp_image: data={}, width={}, height={}",
                  (void*)data, width, height);
        return false;
    }

    uint8_t* output = nullptr;
    size_t output_size = 0;

    std::vector<uint8_t> rgba_buffer;

    if (channels == 4) {
        rgba_buffer.resize(width * height * 4);
        std::memcpy(rgba_buffer.data(), data, width * height * 4);

        output_size = WebPEncodeLosslessRGBA(
            rgba_buffer.data(),
            width,
            height,
            width * 4,
            &output
        );
    } else if (channels == 3) {
        rgba_buffer.resize(width * height * 4);
        for (int i = 0; i < width * height; ++i) {
            rgba_buffer[i * 4 + 0] = data[i * 3 + 0];
            rgba_buffer[i * 4 + 1] = data[i * 3 + 1];
            rgba_buffer[i * 4 + 2] = data[i * 3 + 2];
            rgba_buffer[i * 4 + 3] = 255;
        }

        output_size = WebPEncodeLosslessRGBA(
            rgba_buffer.data(),
            width,
            height,
            width * 4,
            &output
        );
    } else {
        LOG_ERROR("Unsupported number of channels: {}", channels);
        return false;
    }

    if (output_size == 0 || output == nullptr) {
        LOG_ERROR("WebP encoding failed for {} (size={})", path.string(), output_size);
        if (output) WebPFree(output);
        return false;
    }

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        WebPFree(output);
        LOG_ERROR("Failed to open file: {}", path.string());
        return false;
    }

    file.write(reinterpret_cast<const char*>(output), output_size);
    WebPFree(output);

    if (!file.good()) {
        LOG_ERROR("Failed to write file: {}", path.string());
        return false;
    }

    LOG_DEBUG("Successfully wrote WebP: {} ({}x{}, {} bytes)",
             path.string(), width, height, output_size);
    return true;
}

// Create a ZIP archive for .sog bundle
class SogArchive {
    struct archive* a;
    std::filesystem::path path;

public:
    SogArchive(const std::filesystem::path& output_path) : path(output_path) {
        a = archive_write_new();
        archive_write_set_format_zip(a);
        archive_write_open_filename(a, path.string().c_str());
    }

    ~SogArchive() {
        if (a) {
            archive_write_close(a);
            archive_write_free(a);
        }
    }

    bool add_file(const std::string& filename, const void* data, size_t size) {
        struct archive_entry* entry = archive_entry_new();

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        archive_entry_set_pathname(entry, filename.c_str());
        archive_entry_set_size(entry, size);
        archive_entry_set_filetype(entry, AE_IFREG);
        archive_entry_set_perm(entry, 0644);
        archive_entry_set_mtime(entry, time_t, 0);

        if (archive_write_header(a, entry) != ARCHIVE_OK) {
            archive_entry_free(entry);
            LOG_ERROR("Failed to write archive header: {}", archive_error_string(a));
            return false;
        }

        if (archive_write_data(a, data, size) != static_cast<ssize_t>(size)) {
            archive_entry_free(entry);
            LOG_ERROR("Failed to write archive data: {}", archive_error_string(a));
            return false;
        }

        archive_entry_free(entry);
        return true;
    }

    bool add_webp(const std::string& filename, const uint8_t* data,
                  int width, int height, int channels = 4) {

        if (!data || width <= 0 || height <= 0) {
            LOG_ERROR("Invalid input to add_webp: data={}, width={}, height={}",
                      (void*)data, width, height);
            return false;
        }

        uint8_t* output = nullptr;
        size_t output_size = 0;

        std::vector<uint8_t> rgba_buffer;

        if (channels == 4) {
            rgba_buffer.resize(width * height * 4);
            std::memcpy(rgba_buffer.data(), data, width * height * 4);

            output_size = WebPEncodeLosslessRGBA(
                rgba_buffer.data(),
                width,
                height,
                width * 4,
                &output
            );
        } else if (channels == 3) {
            rgba_buffer.resize(width * height * 4);
            for (int i = 0; i < width * height; ++i) {
                rgba_buffer[i * 4 + 0] = data[i * 3 + 0];
                rgba_buffer[i * 4 + 1] = data[i * 3 + 1];
                rgba_buffer[i * 4 + 2] = data[i * 3 + 2];
                rgba_buffer[i * 4 + 3] = 255;
            }

            output_size = WebPEncodeLosslessRGBA(
                rgba_buffer.data(),
                width,
                height,
                width * 4,
                &output
            );
        } else {
            LOG_ERROR("Unsupported number of channels: {}", channels);
            return false;
        }

        if (output_size == 0 || output == nullptr) {
            LOG_ERROR("WebP encoding failed for {} in archive", filename);
            if (output) WebPFree(output);
            return false;
        }

        bool result = add_file(filename, output, output_size);
        WebPFree(output);

        if (result) {
            LOG_DEBUG("Added {} to archive ({}x{}, {} bytes)",
                     filename, width, height, output_size);
        }

        return result;
    }
};

// Identity layout function - matches TypeScript
int identity_layout(int index, int width) {
    return index;
}

} // anonymous namespace

std::expected<void, std::string> write_sog(
    const SplatData& splat_data,
    const SogWriteOptions& options) {

    try {
        LOG_INFO("Writing SOG format to: {}", options.output_path.string());

        const int64_t num_splats = splat_data.size();
        if (num_splats == 0) {
            return std::unexpected("No splats to write");
        }

        // Calculate texture dimensions (multiple of 4) - matches TypeScript
        const int width = ((int)std::ceil(std::sqrt(num_splats) / 4.0)) * 4;
        const int height = ((int)std::ceil(num_splats / (float)width / 4.0)) * 4;
        const int channels = 4;  // Always use RGBA

        LOG_DEBUG("SOG texture dimensions: {}x{} for {} splats", width, height, num_splats);

        // Get data tensors on CPU
        auto means = splat_data.get_means().cpu().contiguous();
        auto scales = splat_data.scaling_raw().cpu().contiguous();
        auto rotations = splat_data.get_rotation().cpu().contiguous();
        auto opacities = splat_data.get_opacity().cpu().contiguous();
        auto sh0 = splat_data.sh0().cpu().contiguous();
        auto shN = splat_data.shN().cpu().contiguous();

        // Determine SH degree from shN shape
        int sh_degree = 0;
        if (shN.defined() && shN.numel() > 0) {
            int num_coeffs = shN.size(1);  // shN shape is [N, num_coeffs, 3]
            if (num_coeffs == 3) sh_degree = 1;
            else if (num_coeffs == 8) sh_degree = 2;
            else if (num_coeffs == 15) sh_degree = 3;
            else LOG_WARN("Unexpected number of SH coefficients: {}, defaulting to degree 0", num_coeffs);
        }
        LOG_DEBUG("Detected SH degree: {}", sh_degree);

        auto means_gpu = means.to(torch::kCUDA);
        auto morton_codes = morton_encode(means_gpu);
        auto indices = morton_sort_indices(morton_codes).cpu();

        // Check if output is .sog bundle or individual files
        bool is_bundle = options.output_path.extension() == ".sog";
        std::unique_ptr<SogArchive> archive;
        std::filesystem::path base_path;

        if (is_bundle) {
            archive = std::make_unique<SogArchive>(options.output_path);
            base_path = options.output_path.parent_path();
        } else {
            base_path = options.output_path.parent_path();
            std::filesystem::create_directories(base_path);
        }

        // Helper lambda to write images
        auto write_image = [&](const std::string& filename,
                               const uint8_t* data,
                               int w = -1, int h = -1) -> bool {
            if (w == -1) w = width;
            if (h == -1) h = height;

            if (!data) {
                LOG_ERROR("Null data pointer for {}", filename);
                return false;
            }

            if (archive) {
                LOG_DEBUG("Adding {} to archive ({}x{})", filename, w, h);
                return archive->add_webp(filename, data, w, h, channels);
            } else {
                auto file_path = base_path / filename;
                auto webp_path = file_path;
                if (webp_path.extension() != ".webp") {
                    webp_path.replace_extension(".webp");
                }
                LOG_DEBUG("Writing {} ({}x{})", webp_path.string(), w, h);
                return write_webp_image(webp_path, data, w, h, channels);
            }
        };

        LOG_DEBUG("Processing positions with log transform");

        // 1. Process positions with log transform
        std::vector<uint8_t> means_l(width * height * channels, 255);  // Initialize alpha to 255
        std::vector<uint8_t> means_u(width * height * channels, 255);

        // Apply log transform and find min/max
        torch::Tensor means_log = torch::zeros_like(means);
        auto means_data = means.accessor<float, 2>();
        auto means_log_data = means_log.accessor<float, 2>();

        for (int64_t i = 0; i < num_splats; ++i) {
            for (int j = 0; j < 3; ++j) {
                means_log_data[i][j] = log_transform(means_data[i][j]);
            }
        }

        auto [means_min, _] = means_log.min(0);
        auto [means_max, __] = means_log.max(0);

        auto means_min_acc = means_min.accessor<float, 1>();
        auto means_max_acc = means_max.accessor<float, 1>();
        auto indices_acc = indices.accessor<int64_t, 1>();

        for (int64_t i = 0; i < num_splats; ++i) {
            int64_t idx = indices_acc[i];
            int ti = identity_layout(i, width);

            float x = (means_log_data[idx][0] - means_min_acc[0]) /
                     (means_max_acc[0] - means_min_acc[0] + 1e-10f);
            float y = (means_log_data[idx][1] - means_min_acc[1]) /
                     (means_max_acc[1] - means_min_acc[1] + 1e-10f);
            float z = (means_log_data[idx][2] - means_min_acc[2]) /
                     (means_max_acc[2] - means_min_acc[2] + 1e-10f);

            uint16_t x16 = static_cast<uint16_t>(65535 * std::max(0.0f, std::min(1.0f, x)));
            uint16_t y16 = static_cast<uint16_t>(65535 * std::max(0.0f, std::min(1.0f, y)));
            uint16_t z16 = static_cast<uint16_t>(65535 * std::max(0.0f, std::min(1.0f, z)));

            means_l[ti * 4 + 0] = x16 & 0xff;
            means_l[ti * 4 + 1] = y16 & 0xff;
            means_l[ti * 4 + 2] = z16 & 0xff;
            // Alpha already set to 255

            means_u[ti * 4 + 0] = (x16 >> 8) & 0xff;
            means_u[ti * 4 + 1] = (y16 >> 8) & 0xff;
            means_u[ti * 4 + 2] = (z16 >> 8) & 0xff;
            // Alpha already set to 255
        }

        if (!write_image("means_l.webp", means_l.data())) {
            return std::unexpected("Failed to write means_l.webp");
        }
        if (!write_image("means_u.webp", means_u.data())) {
            return std::unexpected("Failed to write means_u.webp");
        }

        LOG_DEBUG("Processing quaternions");

        // 2. Process quaternions
        std::vector<uint8_t> quats(width * height * channels, 255);
        auto rot_acc = rotations.accessor<float, 2>();

        for (int64_t i = 0; i < num_splats; ++i) {
            int64_t idx = indices_acc[i];
            int ti = identity_layout(i, width);

            // Rotations are stored as [w, x, y, z] in SplatData
            auto quat = pack_quaternion(
                rot_acc[idx][0],  // w
                rot_acc[idx][1],  // x
                rot_acc[idx][2],  // y
                rot_acc[idx][3]   // z
            );

            quats[ti * 4 + 0] = quat[0];
            quats[ti * 4 + 1] = quat[1];
            quats[ti * 4 + 2] = quat[2];
            quats[ti * 4 + 3] = quat[3];
        }

        if (!write_image("quats.webp", quats.data())) {
            return std::unexpected("Failed to write quats.webp");
        }

        // 3. Cluster scales using k-means
        LOG_DEBUG("Clustering scales with k=256, iterations={}", options.iterations);

        // Flatten scales in column-major order to match TypeScript
        auto scales_flat = torch::zeros({num_splats * 3}, torch::kFloat32);
        auto scales_acc = scales.accessor<float, 2>();
        auto scales_flat_ptr = scales_flat.data_ptr<float>();

        for (int64_t i = 0; i < num_splats; ++i) {
            scales_flat_ptr[i] = scales_acc[i][0];
            scales_flat_ptr[num_splats + i] = scales_acc[i][1];
            scales_flat_ptr[2 * num_splats + i] = scales_acc[i][2];
        }

        auto scales_result = cluster_1d(scales_flat, 256, options.iterations);

        std::vector<uint8_t> scales_data(width * height * channels, 255);
        auto scales_labels_acc = scales_result.labels.accessor<int32_t, 1>();

        for (int64_t i = 0; i < num_splats; ++i) {
            int64_t idx = indices_acc[i];
            int ti = identity_layout(i, width);

            scales_data[ti * 4 + 0] = static_cast<uint8_t>(scales_labels_acc[idx]);
            scales_data[ti * 4 + 1] = static_cast<uint8_t>(scales_labels_acc[num_splats + idx]);
            scales_data[ti * 4 + 2] = static_cast<uint8_t>(scales_labels_acc[2 * num_splats + idx]);
            // Alpha already set to 255
        }

        if (!write_image("scales.webp", scales_data.data())) {
            return std::unexpected("Failed to write scales.webp");
        }

        // 4. Cluster colors using k-means
        LOG_DEBUG("Clustering colors with k=256, iterations={}", options.iterations);

        auto sh0_reshaped = sh0.reshape({num_splats, 3});

        // Create concatenated 1D tensor in column-major order to match TypeScript
        torch::Tensor colors_1d = torch::zeros({num_splats * 3}, torch::kFloat32);
        auto sh0_acc = sh0_reshaped.accessor<float, 2>();
        auto colors_1d_ptr = colors_1d.data_ptr<float>();

        for (int64_t i = 0; i < num_splats; ++i) {
            colors_1d_ptr[i] = sh0_acc[i][0];                    // R values
            colors_1d_ptr[num_splats + i] = sh0_acc[i][1];       // G values
            colors_1d_ptr[2 * num_splats + i] = sh0_acc[i][2];   // B values
        }

        auto colors_result = cluster_1d(colors_1d, 256, options.iterations);

        std::vector<uint8_t> sh0_data(width * height * channels, 0);
        auto colors_labels_acc = colors_result.labels.accessor<int32_t, 1>();
        auto opacity_acc = opacities.accessor<float, 1>();

        for (int64_t i = 0; i < num_splats; ++i) {
            int64_t idx = indices_acc[i];
            int ti = identity_layout(i, width);

            sh0_data[ti * 4 + 0] = static_cast<uint8_t>(colors_labels_acc[idx]);
            sh0_data[ti * 4 + 1] = static_cast<uint8_t>(colors_labels_acc[num_splats + idx]);
            sh0_data[ti * 4 + 2] = static_cast<uint8_t>(colors_labels_acc[2 * num_splats + idx]);

            // Add opacity (already sigmoid applied in get_opacity())
            float opacity = opacity_acc[idx];
            sh0_data[ti * 4 + 3] = static_cast<uint8_t>(255 * std::max(0.0f, std::min(1.0f, opacity)));
        }

        if (!write_image("sh0.webp", sh0_data.data())) {
            return std::unexpected("Failed to write sh0.webp");
        }

        // Create meta.json
        nlohmann::json meta;
        meta["version"] = 2;
        meta["count"] = num_splats;
        meta["width"] = width;
        meta["height"] = height;

        // Store means min/max
        meta["means"]["mins"] = {
            means_min_acc[0],
            means_min_acc[1],
            means_min_acc[2]
        };
        meta["means"]["maxs"] = {
            means_max_acc[0],
            means_max_acc[1],
            means_max_acc[2]
        };
        meta["means"]["files"] = {"means_l.webp", "means_u.webp"};

        // Convert scale centroids to vector
        std::vector<float> scale_codebook;
        auto scale_centroids_acc = scales_result.centroids.accessor<float, 2>();
        for (int i = 0; i < scales_result.centroids.size(0); ++i) {
            scale_codebook.push_back(scale_centroids_acc[i][0]);
        }
        meta["scales"]["codebook"] = scale_codebook;
        meta["scales"]["files"] = {"scales.webp"};

        meta["quats"]["files"] = {"quats.webp"};

        // Convert color centroids to vector
        std::vector<float> color_codebook;
        auto color_centroids_acc = colors_result.centroids.accessor<float, 2>();
        for (int i = 0; i < colors_result.centroids.size(0); ++i) {
            color_codebook.push_back(color_centroids_acc[i][0]);
        }
        meta["sh0"]["codebook"] = color_codebook;
        meta["sh0"]["files"] = {"sh0.webp"};

        // Handle higher-order spherical harmonics if present
        if (sh_degree > 0 && shN.defined() && shN.numel() > 0) {
            LOG_DEBUG("Processing spherical harmonics bands (degree {})", sh_degree);

            const int sh_coeffs = shN.size(1);  // Number of coefficients per color channel

            // Flatten SH coefficients for clustering
            auto shN_reshaped = shN.reshape({num_splats, sh_coeffs * 3});

            // Calculate palette size - matches TypeScript logic
            int palette_size = std::min(64,
                std::max(1, static_cast<int>(std::pow(2, std::floor(std::log2(num_splats / 1024.0)))) * 1024));
            palette_size = std::min(palette_size, static_cast<int>(num_splats));

            LOG_DEBUG("Clustering SH with palette_size={}, sh_coeffs={}", palette_size, sh_coeffs);

            // Cluster SH coefficients
            auto sh_result = cluster_nd(shN_reshaped, palette_size, options.iterations);

            if (sh_result.centroids.size(0) == 0) {
                LOG_WARN("SH clustering returned empty centroids, skipping SH compression");
            } else {
                int actual_palette_size = sh_result.centroids.size(0);
                LOG_DEBUG("SH clustering complete, actual_palette_size={}", actual_palette_size);

                // Further cluster the centroids to create codebook
                auto codebook_result = cluster_1d(sh_result.centroids.flatten(), 256, options.iterations);

                // Calculate dimensions for centroids texture
                const int centroids_width = 64 * sh_coeffs;
                const int centroids_height = (actual_palette_size + 63) / 64;

                LOG_DEBUG("Writing SH centroids with dimensions {}x{}",
                         centroids_width, centroids_height);

                // Write centroids with proper band-major ordering
                std::vector<uint8_t> centroids_buf(centroids_width * centroids_height * channels, 255);
                auto codebook_labels_acc = codebook_result.labels.accessor<int32_t, 1>();

                for (int i = 0; i < actual_palette_size; ++i) {
                    for (int j = 0; j < sh_coeffs; ++j) {
                        int pixel_idx = i * sh_coeffs + j;

                        if (pixel_idx < centroids_width * centroids_height) {
                            // Band-major ordering: iterate through bands, then coefficients
                            for (int c = 0; c < 3; ++c) {
                                int coeff_idx = j + c * sh_coeffs;
                                int centroid_idx = i * (sh_coeffs * 3) + coeff_idx;

                                if (centroid_idx < codebook_result.labels.size(0)) {
                                    centroids_buf[pixel_idx * 4 + c] =
                                        static_cast<uint8_t>(codebook_labels_acc[centroid_idx]);
                                }
                            }
                            // Alpha already set to 255
                        }
                    }
                }

                if (!write_image("shN_centroids.webp", centroids_buf.data(), centroids_width, centroids_height)) {
                    return std::unexpected("Failed to write shN_centroids.webp");
                }

                LOG_DEBUG("Writing SH labels");

                // Write labels
                std::vector<uint8_t> labels_buf(width * height * channels, 255);
                auto sh_labels_acc = sh_result.labels.accessor<int32_t, 1>();

                for (int64_t i = 0; i < num_splats; ++i) {
                    int64_t idx = indices_acc[i];
                    int32_t label = sh_labels_acc[idx];
                    int ti = identity_layout(i, width);

                    labels_buf[ti * 4 + 0] = label & 0xff;
                    labels_buf[ti * 4 + 1] = (label >> 8) & 0xff;
                    labels_buf[ti * 4 + 2] = 0;
                    // Alpha already set to 255
                }

                if (!write_image("shN_labels.webp", labels_buf.data())) {
                    return std::unexpected("Failed to write shN_labels.webp");
                }

                // Add to meta.json with all required fields
                std::vector<float> sh_codebook;
                auto sh_codebook_acc = codebook_result.centroids.accessor<float, 2>();
                int codebook_size = std::min(256, static_cast<int>(codebook_result.centroids.size(0)));
                for (int i = 0; i < codebook_size; ++i) {
                    sh_codebook.push_back(sh_codebook_acc[i][0]);
                }

                meta["shN"]["codebook"] = sh_codebook;
                meta["shN"]["palette_size"] = actual_palette_size;
                meta["shN"]["bands"] = sh_degree;
                meta["shN"]["coeffs"] = sh_coeffs;
                meta["shN"]["files"] = {"shN_centroids.webp", "shN_labels.webp"};

                LOG_DEBUG("SH processing complete - codebook size: {}, palette: {}, bands: {}, coeffs: {}",
                         sh_codebook.size(), actual_palette_size, sh_degree, sh_coeffs);
            }
        }

        // Write meta.json
        std::string meta_json = meta.dump(2);

        if (archive) {
            LOG_INFO("Writing meta.json to archive");
            if (!archive->add_file("meta.json", meta_json.c_str(), meta_json.size())) {
                return std::unexpected("Failed to write meta.json to archive");
            }
            LOG_INFO("Successfully wrote SOG bundle: {}", options.output_path.string());
        } else {
            auto meta_path = options.output_path;
            if (meta_path.extension() != ".json") {
                meta_path = base_path / "meta.json";
            }

            LOG_INFO("Writing meta.json to: {}", meta_path.string());
            std::ofstream meta_file(meta_path);
            if (!meta_file) {
                LOG_ERROR("Failed to open meta.json for writing at: {}", meta_path.string());
                return std::unexpected("Failed to open meta.json for writing");
            }
            meta_file << meta_json;
            meta_file.close();

            if (!meta_file) {
                LOG_ERROR("Failed to write meta.json");
                return std::unexpected("Failed to write meta.json");
            }

            LOG_INFO("Successfully wrote SOG format as individual files to: {}", base_path.string());
        }

        LOG_INFO("Successfully completed SOG write with {} splats", num_splats);

        return {};

    } catch (const std::exception& e) {
        LOG_ERROR("Exception in write_sog: {}", e.what());
        return std::unexpected(std::format("Failed to write SOG: {}", e.what()));
    }
}
} // namespace gs::core