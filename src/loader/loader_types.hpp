#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <variant>
#include <vector>

// Forward declarations - we can only use these as pointers/references
namespace gs {
    class SplatData;
    struct PointCloud;
    class CameraDataset;
} // namespace gs

namespace gs::loader {

    // Progress callback type
    using ProgressCallback = std::function<void(float percentage, const std::string& message)>;

    // ============================================================================
    // Loading Options
    // ============================================================================

    struct LoadOptions {
        // Dataset configuration
        int resolution = -1;                  // Image resolution override
        std::string images_folder = "images"; // For COLMAP datasets

        // Loading behavior
        bool validate_only = false; // Only validate, don't load

        // Progress reporting
        ProgressCallback progress = nullptr;
    };

    // ============================================================================
    // Scene Data (for COLMAP/Blender datasets)
    // ============================================================================

    struct SceneData {
        std::shared_ptr<gs::CameraDataset> cameras;
        std::shared_ptr<gs::PointCloud> point_cloud; // Use shared_ptr for forward declared type
    };

    // ============================================================================
    // Load Result
    // ============================================================================

    struct LoadResult {
        // What we loaded - using alias to avoid comma confusion
        using SplatDataPtr = std::shared_ptr<gs::SplatData>;
        std::variant<SplatDataPtr, SceneData> data;

        // Common metadata
        torch::Tensor scene_center = torch::zeros({3});
        std::string loader_used;
        std::chrono::milliseconds load_time{0};

        // Validation warnings (non-fatal issues)
        std::vector<std::string> warnings;
    };

} // namespace gs::loader
