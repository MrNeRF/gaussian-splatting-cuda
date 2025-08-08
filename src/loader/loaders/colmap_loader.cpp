#include "loader/loaders/colmap_loader.hpp"
#include "core/camera.hpp"
#include "core/dataset.hpp"
#include "core/point_cloud.hpp"
#include "formats/colmap.hpp"
#include <chrono>
#include <filesystem>
#include <format>
#include <print>

namespace gs::loader {

    std::expected<LoadResult, std::string> ColmapLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        auto start_time = std::chrono::high_resolution_clock::now();

        // Validate directory exists
        if (!std::filesystem::exists(path)) {
            return std::unexpected(std::format("Path does not exist: {}", path.string()));
        }

        if (!std::filesystem::is_directory(path)) {
            return std::unexpected("COLMAP dataset must be a directory");
        }

        // Report initial progress
        if (options.progress) {
            options.progress(0.0f, "Loading COLMAP dataset...");
        }

        // Check for sparse reconstruction
        std::filesystem::path sparse_path;
        if (std::filesystem::exists(path / "sparse" / "0")) {
            sparse_path = path / "sparse" / "0";
        } else if (std::filesystem::exists(path / "sparse")) {
            sparse_path = path / "sparse";
        } else {
            return std::unexpected("No sparse reconstruction found (expected 'sparse' or 'sparse/0' directory)");
        }

        // Check for required COLMAP files
        bool has_cameras = std::filesystem::exists(sparse_path / "cameras.bin");
        bool has_images = std::filesystem::exists(sparse_path / "images.bin");
        bool has_points = std::filesystem::exists(sparse_path / "points3D.bin");

        if (!has_cameras || !has_images) {
            return std::unexpected(std::format(
                "Missing required COLMAP files. cameras.bin: {}, images.bin: {}",
                has_cameras ? "found" : "missing",
                has_images ? "found" : "missing"));
        }

        // Check for image directory
        std::filesystem::path image_dir = path / options.images_folder;
        if (!std::filesystem::exists(image_dir)) {
            return std::unexpected(std::format(
                "Images directory '{}' not found", options.images_folder));
        }

        // Validation only mode
        if (options.validate_only) {
            if (options.progress) {
                options.progress(100.0f, "COLMAP validation complete");
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            return LoadResult{
                .data = LoadedScene{
                    .cameras = nullptr,
                    .point_cloud = nullptr},
                .scene_center = torch::zeros({3}),
                .loader_used = name(),
                .load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                .warnings = has_points ? std::vector<std::string>{} : std::vector<std::string>{"No sparse point cloud found (points3D.bin) - will use random initialization"}};
        }

        // Load cameras and images
        if (options.progress) {
            options.progress(20.0f, "Reading camera parameters...");
        }

        try {
            // Read COLMAP data
            auto [camera_infos, scene_center] = read_colmap_cameras_and_images(
                path, options.images_folder);

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
                options.progress(60.0f, "Loading point cloud...");
            }

            // Load point cloud if it exists
            std::shared_ptr<PointCloud> point_cloud;
            if (has_points) {
                auto loaded_pc = read_colmap_point_cloud(path);
                point_cloud = std::make_shared<PointCloud>(std::move(loaded_pc));
                std::println("Loaded {} points from COLMAP", point_cloud->size());
            } else {
                point_cloud = std::make_shared<PointCloud>();
                std::println("No point cloud found - will use random initialization");
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
                .warnings = has_points ? std::vector<std::string>{} : std::vector<std::string>{"No sparse point cloud found - using random initialization"}};

            std::println("COLMAP dataset loaded successfully in {}ms", load_time.count());
            std::println("  - {} cameras", camera_infos.size());
            std::println("  - Scene center: [{:.3f}, {:.3f}, {:.3f}]",
                         scene_center[0].item<float>(),
                         scene_center[1].item<float>(),
                         scene_center[2].item<float>());

            return result;

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to load COLMAP dataset: {}", e.what()));
        }
    }

    bool ColmapLoader::canLoad(const std::filesystem::path& path) const {
        if (!std::filesystem::exists(path) || !std::filesystem::is_directory(path)) {
            return false;
        }

        // Check for COLMAP structure
        return std::filesystem::exists(path / "sparse" / "0" / "cameras.bin") ||
               std::filesystem::exists(path / "sparse" / "cameras.bin");
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

    std::vector<CameraData> ColmapLoader::getImagesCams(const std::filesystem::path& path) const {
        if (!canLoad(path)) {
            return {};
        }
        // Check for image directory
        std::filesystem::path image_dir = path / "images";
        if (!std::filesystem::exists(image_dir)) {
            std::println("images directory not found");
            return {};
        }

        try {
            // Read COLMAP data
            auto [camera_infos, scene_center] = read_colmap_cameras_and_images(
                path, "images");
            return camera_infos;
        } catch (std::runtime_error& e) {

            // something unxpected happen thow
            std::println("getImagesCams unexpected error: {}", e.what());
        }
        return {};
    }
} // namespace gs::loader