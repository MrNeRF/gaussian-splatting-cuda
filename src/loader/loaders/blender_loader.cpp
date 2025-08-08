#include "loader/loaders/blender_loader.hpp"
#include "core/camera.hpp"
#include "core/dataset.hpp"
#include "core/point_cloud.hpp"
#include "formats/transforms.hpp"
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <print>

namespace gs::loader {

    std::expected<LoadResult, std::string> BlenderLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        auto start_time = std::chrono::high_resolution_clock::now();

        // Validate path exists
        if (!std::filesystem::exists(path)) {
            return std::unexpected(std::format("Path does not exist: {}", path.string()));
        }

        // Report initial progress
        if (options.progress) {
            options.progress(0.0f, "Loading Blender/NeRF dataset...");
        }

        // Determine transforms file path
        std::filesystem::path transforms_file;

        if (std::filesystem::is_directory(path)) {
            // Look for transforms files in directory
            if (std::filesystem::exists(path / "transforms_train.json")) {
                transforms_file = path / "transforms_train.json";
            } else if (std::filesystem::exists(path / "transforms.json")) {
                transforms_file = path / "transforms.json";
            } else {
                return std::unexpected(
                    "No transforms file found (expected 'transforms.json' or 'transforms_train.json')");
            }
        } else if (path.extension() == ".json") {
            // Direct path to transforms file
            transforms_file = path;
        } else {
            return std::unexpected("Path must be a directory or a JSON file");
        }

        // Validation only mode
        if (options.validate_only) {
            // Check if the transforms file is valid JSON
            std::ifstream file(transforms_file);
            if (!file) {
                return std::unexpected("Cannot open transforms file");
            }

            // Try to parse as JSON (basic validation)
            try {
                nlohmann::json j;
                file >> j;

                if (!j.contains("frames") || !j["frames"].is_array()) {
                    return std::unexpected("Invalid transforms file: missing 'frames' array");
                }
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Invalid JSON: {}", e.what()));
            }

            if (options.progress) {
                options.progress(100.0f, "Blender/NeRF validation complete");
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            return LoadResult{
                .data = LoadedScene{
                    .cameras = nullptr,
                    .point_cloud = nullptr},
                .scene_center = torch::zeros({3}),
                .loader_used = name(),
                .load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                .warnings = {"Blender/NeRF datasets use random point cloud initialization"}};
        }

        // Load the dataset
        if (options.progress) {
            options.progress(20.0f, "Reading transforms file...");
        }

        try {
            // Read transforms and create cameras
            auto [camera_infos, scene_center] = read_transforms_cameras_and_images(transforms_file);

            if (options.progress) {
                options.progress(40.0f, std::format("Creating {} cameras...", camera_infos.size()));
            }

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
                    info._width,
                    info._height,
                    static_cast<int>(i));

                cameras.push_back(std::move(cam));
            }

            // Create dataset configuration
            gs::param::DatasetConfig dataset_config;
            dataset_config.data_path = path;
            dataset_config.images = options.images_folder;
            dataset_config.resolution = options.resolution;

            // Create dataset with ALL images
            auto dataset = std::make_shared<gs::CameraDataset>(
                std::move(cameras), dataset_config, gs::CameraDataset::Split::ALL);

            if (options.progress) {
                options.progress(60.0f, "Generating random point cloud...");
            }

            // Generate random point cloud for Blender datasets
            auto random_pc = generate_random_point_cloud();
            auto point_cloud = std::make_shared<PointCloud>(std::move(random_pc));
            std::println("Generated random point cloud with {} points", point_cloud->size());

            if (options.progress) {
                options.progress(100.0f, "Blender/NeRF loading complete");
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
                .warnings = {"Using random point cloud initialization"}};

            std::println("Blender/NeRF dataset loaded successfully in {}ms", load_time.count());
            std::println("  - {} cameras", camera_infos.size());
            std::println("  - Scene center: [{:.3f}, {:.3f}, {:.3f}]",
                         scene_center[0].item<float>(),
                         scene_center[1].item<float>(),
                         scene_center[2].item<float>());

            return result;

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to load Blender/NeRF dataset: {}", e.what()));
        }
    }

    bool BlenderLoader::canLoad(const std::filesystem::path& path) const {
        if (!std::filesystem::exists(path)) {
            return false;
        }

        if (std::filesystem::is_directory(path)) {
            // Check for transforms files in directory
            return std::filesystem::exists(path / "transforms.json") ||
                   std::filesystem::exists(path / "transforms_train.json");
        } else {
            // Check if it's a JSON file
            return path.extension() == ".json";
        }
    }

    std::string BlenderLoader::name() const {
        return "Blender/NeRF";
    }

    std::vector<std::string> BlenderLoader::supportedExtensions() const {
        return {".json"}; // Can load JSON files directly
    }

    int BlenderLoader::priority() const {
        return 5; // Medium priority
    }

    std::vector<CameraData> BlenderLoader::getImagesCams(const std::filesystem::path& path) const {
        if (!canLoad(path)) {
            return {};
        }

        try {
            auto [camera_infos, scene_center] = read_transforms_cameras_and_images(path);

            return camera_infos;
        } catch (std::runtime_error& e) {
            // something unxpected happen thow
            std::println("getImagesCams unexpected error: {}", e.what());
        }
        return {};
    }

} // namespace gs::loader