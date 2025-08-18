#pragma once

#include <array>
#include <glm/glm.hpp>
#include <memory>
#include <optional>

// Forward declarations
namespace torch {
    class Tensor;
}

namespace gs {
    class SplatData;
}

namespace gs::rendering {

    // Public types
    struct ViewportData {
        glm::mat3 rotation;
        glm::vec3 translation;
        glm::ivec2 size;
        float fov = 60.0f;
    };

    struct BoundingBox {
        glm::vec3 min;
        glm::vec3 max;
        glm::mat4 transform{1.0f};
    };

    struct RenderRequest {
        ViewportData viewport;
        float scaling_modifier = 1.0f;
        bool antialiasing = false;
        glm::vec3 background_color{0.0f, 0.0f, 0.0f};
        std::optional<BoundingBox> crop_box;
        bool point_cloud_mode = false;
        float voxel_size = 0.01f;
    };

    struct RenderResult {
        std::shared_ptr<torch::Tensor> image;
        std::shared_ptr<torch::Tensor> depth;
        bool valid = false;
    };

    enum class GridPlane {
        YZ = 0, // X plane
        XZ = 1, // Y plane
        XY = 2  // Z plane
    };

    // Main rendering engine
    class RenderingEngine {
    public:
        static std::unique_ptr<RenderingEngine> create();

        virtual ~RenderingEngine() = default;

        // Lifecycle
        virtual void initialize() = 0;
        virtual void shutdown() = 0;
        virtual bool isInitialized() const = 0;

        // Core rendering
        virtual RenderResult renderGaussians(
            const SplatData& splat_data,
            const RenderRequest& request) = 0;

        // Present to screen
        virtual void presentToScreen(
            const RenderResult& result,
            const glm::ivec2& viewport_pos,
            const glm::ivec2& viewport_size) = 0;

        // Overlay rendering
        virtual void renderGrid(
            const ViewportData& viewport,
            GridPlane plane = GridPlane::XZ,
            float opacity = 0.5f) = 0;

        virtual void renderBoundingBox(
            const BoundingBox& box,
            const ViewportData& viewport,
            const glm::vec3& color = glm::vec3(1.0f, 1.0f, 0.0f),
            float line_width = 2.0f) = 0;

        virtual void renderCoordinateAxes(
            const ViewportData& viewport,
            float size = 2.0f,
            const std::array<bool, 3>& visible = {true, true, true}) = 0;
    };

} // namespace gs::rendering