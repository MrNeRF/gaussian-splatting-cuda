#pragma once

#include <chrono>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <variant>
#include <vector>

// Forward declarations only - hide implementation details
namespace gs {
    class SplatData;
    class CameraDataset;
    struct PointCloud;
} // namespace gs

namespace gs::loader {

    // Progress callback type
    using ProgressCallback = std::function<void(float percentage, const std::string& message)>;

    // Public types that clients need
    struct LoadOptions {
        int resolution = -1;
        std::string images_folder = "images";
        bool validate_only = false;
        ProgressCallback progress = nullptr;
    };

    struct LoadedScene {
        std::shared_ptr<CameraDataset> cameras;
        std::shared_ptr<PointCloud> point_cloud;
    };

    struct LoadResult {
        std::variant<std::shared_ptr<SplatData>, LoadedScene> data;
        torch::Tensor scene_center;
        std::string loader_used;
        std::chrono::milliseconds load_time{0};
        std::vector<std::string> warnings;
    };

    /**
     * @brief Main loader interface - the ONLY public API for the loader module
     *
     * This class provides a clean facade over all loading functionality.
     * All implementation details are hidden behind this interface.
     */
    class Loader {
    public:
        /**
         * @brief Create a loader instance
         */
        static std::unique_ptr<Loader> create();

        /**
         * @brief Load data from any supported format
         * @param path File or directory to load
         * @param options Loading options
         * @return LoadResult on success, error string on failure
         */
        virtual std::expected<LoadResult, std::string> load(
            const std::filesystem::path& path,
            const LoadOptions& options = {}) = 0;

        /**
         * @brief Check if a path can be loaded
         * @param path File or directory to check
         * @return true if the path can be loaded
         */
        virtual bool canLoad(const std::filesystem::path& path) const = 0;

        /**
         * @brief Get list of supported formats
         * @return Human-readable list of supported formats
         */
        virtual std::vector<std::string> getSupportedFormats() const = 0;

        /**
         * @brief Get list of supported file extensions
         * @return List of extensions (e.g., ".ply", ".json")
         */
        virtual std::vector<std::string> getSupportedExtensions() const = 0;

        virtual ~Loader() = default;
    };

} // namespace gs::loader
