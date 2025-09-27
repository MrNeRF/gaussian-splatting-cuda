/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "loader/loaders/colmap_loader.hpp"
#include "core/camera.hpp"
#include "core/logger.hpp"
#include "core/point_cloud.hpp"
#include "formats/colmap.hpp"
#include "loader/filesystem_utils.hpp"
#include "training/dataset.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <format>

namespace gs::loader {

    std::expected<LoadResult, std::string> ColmapLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        LOG_TIMER("COLMAP Loading");
        auto start_time = std::chrono::high_resolution_clock::now();

        // Validate directory exists
        if (!std::filesystem::exists(path)) {
            std::string error_msg = std::format("Path does not exist: {}", path.string());
            LOG_ERROR("{}", error_msg);
            throw std::runtime_error(error_msg);
        }

        if (!std::filesystem::is_directory(path)) {
            LOG_ERROR("COLMAP dataset must be a directory: {}", path.string());
            throw std::runtime_error("COLMAP dataset must be a directory");
        }

        // Report initial progress
        if (options.progress) {
            options.progress(0.0f, "Loading COLMAP dataset...");
        }

        // Get search paths for COLMAP files
        auto search_paths = get_colmap_search_paths(path);

        // Check for required COLMAP files in any of the search paths
        auto cameras_bin = find_file_in_paths(search_paths, "cameras.bin");
        auto images_bin = find_file_in_paths(search_paths, "images.bin");
        auto points_bin = find_file_in_paths(search_paths, "points3D.bin");

        auto cameras_txt = find_file_in_paths(search_paths, "cameras.txt");
        auto images_txt = find_file_in_paths(search_paths, "images.txt");
        auto points_txt = find_file_in_paths(search_paths, "points3D.txt");

        bool has_cameras = !cameras_bin.empty();
        bool has_images = !images_bin.empty();
        bool has_points = !points_bin.empty();

        bool has_cameras_text = !cameras_txt.empty();
        bool has_images_text = !images_txt.empty();
        bool has_points_text = !points_txt.empty();

        if ((has_cameras || has_images || has_points) &&
            (has_cameras_text || has_images_text || has_points_text)) {
            LOG_WARN("Found both binary and text COLMAP files. Prioritizing binary files.");
        }

        bool trying_text = !(has_cameras && has_images) && (has_cameras_text && has_images_text);
        LOG_INFO("Loading COLMAP in {} format", trying_text ? "text" : "binary");

        // If you don't have binary cameras or images AND you are not trying text: error
        if ((!has_cameras || !has_images) && !trying_text) {
            std::string error_msg = std::format(
                "Missing required COLMAP files. cameras.bin: {}, images.bin: {}",
                has_cameras ? "found" : "missing",
                has_images ? "found" : "missing");
            LOG_ERROR("{}", error_msg);
            throw std::runtime_error(error_msg);
        }

        // If you don't have text cameras or images AND you trying text: error
        if ((!has_cameras_text || !has_images_text) && trying_text) {
            std::string error_msg = std::format(
                "Missing required COLMAP text files. cameras.txt: {}, images.txt: {}",
                has_cameras_text ? "found" : "missing",
                has_images_text ? "found" : "missing");
            LOG_ERROR("{}", error_msg);
            throw std::runtime_error(error_msg);
        }

        // First check if the requested images folder exists
        std::string actual_images_folder = options.images_folder;
        std::filesystem::path image_dir = path / actual_images_folder;

        // If the specified images folder doesn't exist, check if we're in a flat structure
        if (!std::filesystem::exists(image_dir)) {
            // Check if COLMAP files are in the root (flat structure)
            bool is_flat_structure = (!cameras_txt.empty() && cameras_txt.parent_path() == path) ||
                                     (!cameras_bin.empty() && cameras_bin.parent_path() == path);

            if (is_flat_structure) {
                // In flat structure, images are typically in the root directory
                // Check if there are image files in the root
                bool has_images_in_root = false;
                for (const auto& entry : std::filesystem::directory_iterator(path)) {
                    if (entry.is_regular_file() && is_image_file(entry.path())) {
                        has_images_in_root = true;
                        break;
                    }
                }

                if (has_images_in_root) {
                    // Use root directory as images folder for flat structure
                    actual_images_folder = ".";
                    image_dir = path;
                    LOG_INFO("Detected flat structure - using root directory for images");
                } else {
                    std::string error_msg = std::format(
                        "Images directory '{}' not found and no images in root", options.images_folder);
                    LOG_ERROR("{}", error_msg);
                    throw std::runtime_error(error_msg);
                }
            } else {
                std::string error_msg = std::format(
                    "Images directory '{}' not found", options.images_folder);
                LOG_ERROR("{}", error_msg);
                throw std::runtime_error(error_msg);
            }
        }

        // Validation only mode
        if (options.validate_only) {
            if (options.progress) {
                options.progress(100.0f, "COLMAP validation complete");
            }

            LOG_DEBUG("COLMAP validation successful");

            auto end_time = std::chrono::high_resolution_clock::now();
            return LoadResult{
                .data = LoadedScene{
                    .cameras = nullptr,
                    .point_cloud = nullptr},
                .scene_center = torch::zeros({3}),
                .loader_used = name(),
                .load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                .warnings = (has_points || has_points_text) ? std::vector<std::string>{} : std::vector<std::string>{"No sparse point cloud found (points3D.bin|txt) - will use random initialization"}};
        }

        // Load cameras and images
        if (options.progress) {
            options.progress(20.0f, "Reading camera parameters...");
        }

        try {
            std::vector<CameraData> camera_infos;
            torch::Tensor scene_center;
            if (has_cameras && has_images) {
                LOG_DEBUG("Reading binary COLMAP data");
                // Read binary COLMAP data with actual images folder
                std::tie(camera_infos, scene_center) = read_colmap_cameras_and_images(path, actual_images_folder);
            } else if (has_cameras_text && has_images_text) {
                LOG_DEBUG("Reading text COLMAP data");
                // Read text-based COLMAP data with actual images folder
                std::tie(camera_infos, scene_center) = read_colmap_cameras_and_images_text(path, actual_images_folder);
            } else {
                LOG_ERROR("No valid COLMAP camera and image data found");
                throw std::runtime_error("No valid COLMAP camera and image data found");
            }

            if (options.progress) {
                options.progress(40.0f, std::format("Creating {} cameras...", camera_infos.size()));
            }

            LOG_DEBUG("Creating {} camera objects", camera_infos.size());

            // Create Camera objects
            std::vector<std::shared_ptr<Camera>> cameras;
            cameras.reserve(camera_infos.size());

            for (size_t i = 0; i < camera_infos.size(); ++i) {
                const auto& info = camera_infos[i];

                auto cam = std::make_shared<Camera>(
                    info._R,
                    info._T,
                    info._focal_x,
                    info._focal_y,
                    info._center_x,
                    info._center_y,
                    info._radial_distortion,
                    info._tangential_distortion,
                    info._camera_model_type,
                    info._image_name,
                    info._image_path,
                    info._mask_path,
                    info._width,
                    info._height,
                    static_cast<int>(i));

                cameras.push_back(std::move(cam));
            }

            // Create dataset configuration with actual images folder
            gs::param::DatasetConfig dataset_config;
            dataset_config.data_path = path;
            dataset_config.images = actual_images_folder;
            dataset_config.resize_factor = options.resize_factor;
            dataset_config.max_width = options.max_width;

            // Create dataset with ALL images - use correct namespace
            auto dataset = std::make_shared<gs::training::CameraDataset>(
                std::move(cameras), dataset_config, gs::training::CameraDataset::Split::ALL);

            if (options.progress) {
                options.progress(60.0f, "Loading point cloud...");
            }

            // Load point cloud if it exists
            std::shared_ptr<PointCloud> point_cloud;
            if (has_points) {
                LOG_DEBUG("Loading binary point cloud");
                auto loaded_pc = read_colmap_point_cloud(path);
                point_cloud = std::make_shared<PointCloud>(std::move(loaded_pc));
                LOG_INFO("Loaded {} points from COLMAP", point_cloud->size());
            } else if (has_points_text) {
                LOG_DEBUG("Loading text point cloud");
                auto loaded_pc = read_colmap_point_cloud_text(path);
                point_cloud = std::make_shared<PointCloud>(std::move(loaded_pc));
                LOG_INFO("Loaded {} points from COLMAP text file", point_cloud->size());
            } else {
                LOG_WARN("No point cloud found - will use random initialization");
                point_cloud = std::make_shared<PointCloud>();
            }

            if (options.progress) {
                options.progress(100.0f, "COLMAP loading complete");
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);

            // Create result with shared_ptr
            LoadResult result{
                .data = LoadedScene{
                    .cameras = std::move(dataset),
                    .point_cloud = std::move(point_cloud)},
                .scene_center = scene_center,
                .loader_used = name(),
                .load_time = load_time,
                .warnings = (has_points || has_points_text) ? std::vector<std::string>{} : std::vector<std::string>{"No sparse point cloud found - using random initialization"}};

            LOG_INFO("COLMAP dataset loaded successfully in {}ms", load_time.count());
            LOG_INFO("  - {} cameras", camera_infos.size());
            LOG_DEBUG("  - Scene center: [{:.3f}, {:.3f}, {:.3f}]",
                      scene_center[0].item<float>(),
                      scene_center[1].item<float>(),
                      scene_center[2].item<float>());

            return result;

        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to load COLMAP dataset: {}", e.what());
            LOG_ERROR("{}", error_msg);
            throw std::runtime_error(error_msg);
        }
    }

    bool ColmapLoader::canLoad(const std::filesystem::path& path) const {
        if (!safe_exists(path) || !safe_is_directory(path)) {
            return false;
        }

        auto search_paths = get_colmap_search_paths(path);

        // Check for COLMAP files in any location
        const std::vector<std::string> colmap_files = {
            "cameras.bin", "cameras.txt",
            "images.bin", "images.txt"};

        for (const auto& filename : colmap_files) {
            if (!find_file_in_paths(search_paths, filename).empty()) {
                return true;
            }
        }

        return false;
    }

    std::string ColmapLoader::name() const {
        return "COLMAP";
    }

    std::vector<std::string> ColmapLoader::supportedExtensions() const {
        return {}; // Directory-based, no file extensions
    }

    int ColmapLoader::priority() const {
        return 5; // Medium priority
    }
} // namespace gs::loader