#pragma once

#include "axes_renderer.hpp"
#include "bbox_renderer.hpp"
#include "grid_renderer.hpp"
#include "point_cloud_renderer.hpp"
#include "rendering/rendering.hpp"
#include "rendering_pipeline.hpp"
#include "screen_renderer.hpp"
#include "viewport_gizmo.hpp"

namespace gs::rendering {

    class RenderingEngineImpl : public RenderingEngine {
    public:
        RenderingEngineImpl();
        ~RenderingEngineImpl() override;

        void initialize() override;
        void shutdown() override;
        bool isInitialized() const override { return initialized_; }

        RenderResult renderGaussians(
            const SplatData& splat_data,
            const RenderRequest& request) override;

        void presentToScreen(
            const RenderResult& result,
            const glm::ivec2& viewport_pos,
            const glm::ivec2& viewport_size) override;

        void renderGrid(
            const ViewportData& viewport,
            GridPlane plane,
            float opacity) override;

        void renderBoundingBox(
            const BoundingBox& box,
            const ViewportData& viewport,
            const glm::vec3& color,
            float line_width) override;

        void renderCoordinateAxes(
            const ViewportData& viewport,
            float size,
            const std::array<bool, 3>& visible) override;

        void renderViewportGizmo(
            const glm::mat3& camera_rotation,
            const glm::vec2& viewport_pos,
            const glm::vec2& viewport_size) override;

        // Pipeline compatibility
        RenderingPipelineResult renderWithPipeline(
            const SplatData& model,
            const RenderingPipelineRequest& request) override;

        // Factory methods
        std::shared_ptr<IBoundingBox> createBoundingBox() override;
        std::shared_ptr<ICoordinateAxes> createCoordinateAxes() override;

    private:
        void initializeShaders();
        glm::mat4 createProjectionMatrix(const ViewportData& viewport) const;
        glm::mat4 createViewMatrix(const ViewportData& viewport) const;

        // Core components
        std::unique_ptr<RenderingPipeline> pipeline_;
        std::unique_ptr<PointCloudRenderer> point_cloud_renderer_;
        std::shared_ptr<ScreenQuadRenderer> screen_renderer_;

        // Overlay renderers
        std::unique_ptr<RenderInfiniteGrid> grid_renderer_;
        std::unique_ptr<RenderBoundingBox> bbox_renderer_;
        std::unique_ptr<RenderCoordinateAxes> axes_renderer_;
        std::unique_ptr<ViewportGizmo> viewport_gizmo_;

        // Shaders
        std::shared_ptr<Shader> quad_shader_;

        bool initialized_ = false;
    };

} // namespace gs::rendering