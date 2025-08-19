#pragma once

#include "axes_renderer.hpp"
#include "bbox_renderer.hpp"
#include "grid_renderer.hpp"
#include "point_cloud_renderer.hpp"
#include "rendering/rendering.hpp"
#include "rendering_pipeline.hpp"
#include "screen_renderer.hpp"
#include "shader_manager.hpp"
#include "viewport_gizmo.hpp"

namespace gs::rendering {

    class RenderingEngineImpl : public RenderingEngine {
    public:
        RenderingEngineImpl();
        ~RenderingEngineImpl() override;

        Result<void> initialize() override;
        void shutdown() override;
        bool isInitialized() const override;

        Result<RenderResult> renderGaussians(
            const SplatData& splat_data,
            const RenderRequest& request) override;

        Result<void> presentToScreen(
            const RenderResult& result,
            const glm::ivec2& viewport_pos,
            const glm::ivec2& viewport_size) override;

        Result<void> renderGrid(
            const ViewportData& viewport,
            GridPlane plane,
            float opacity) override;

        Result<void> renderBoundingBox(
            const BoundingBox& box,
            const ViewportData& viewport,
            const glm::vec3& color,
            float line_width) override;

        Result<void> renderCoordinateAxes(
            const ViewportData& viewport,
            float size,
            const std::array<bool, 3>& visible) override;

        Result<void> renderViewportGizmo(
            const glm::mat3& camera_rotation,
            const glm::vec2& viewport_pos,
            const glm::vec2& viewport_size) override;

        // Pipeline compatibility
        RenderingPipelineResult renderWithPipeline(
            const SplatData& model,
            const RenderingPipelineRequest& request) override;

        // Factory methods
        Result<std::shared_ptr<IBoundingBox>> createBoundingBox() override;
        Result<std::shared_ptr<ICoordinateAxes>> createCoordinateAxes() override;

    private:
        Result<void> initializeShaders();
        glm::mat4 createProjectionMatrix(const ViewportData& viewport) const;
        glm::mat4 createViewMatrix(const ViewportData& viewport) const;

        // Core components
        RenderingPipeline pipeline_;
        PointCloudRenderer point_cloud_renderer_;
        std::shared_ptr<ScreenQuadRenderer> screen_renderer_;

        // Overlay renderers
        RenderInfiniteGrid grid_renderer_;
        RenderBoundingBox bbox_renderer_;
        RenderCoordinateAxes axes_renderer_;
        ViewportGizmo viewport_gizmo_;

        // Shaders
        ManagedShader quad_shader_;
    };

} // namespace gs::rendering