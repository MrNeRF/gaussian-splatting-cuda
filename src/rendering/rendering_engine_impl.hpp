/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "axes_renderer.hpp"
#include "bbox_renderer.hpp"
#include "grid_renderer.hpp"
#include "rendering/rendering.hpp"
#include "rendering_pipeline.hpp"
#include "screen_renderer.hpp"
#include "shader_manager.hpp"
#include "split_view_renderer.hpp"
#include "translation_gizmo.hpp"
#include "viewport_gizmo.hpp"

namespace gs::rendering {

    // Adapter to bridge public interface with internal implementation
    class GizmoInteractionAdapter : public GizmoInteraction {
        TranslationGizmo* gizmo_;

    public:
        explicit GizmoInteractionAdapter(TranslationGizmo* gizmo) : gizmo_(gizmo) {}

        GizmoElement pick(const glm::vec2& mouse_pos, const glm::mat4& view,
                          const glm::mat4& projection, const glm::vec3& position) override {
            auto elem = gizmo_->pick(mouse_pos, view, projection, position);
            return static_cast<GizmoElement>(elem);
        }

        glm::vec3 startDrag(GizmoElement element, const glm::vec2& mouse_pos,
                            const glm::mat4& view, const glm::mat4& projection,
                            const glm::vec3& position) override {
            return gizmo_->startDrag(static_cast<TranslationGizmo::Element>(element),
                                     mouse_pos, view, projection, position);
        }

        glm::vec3 updateDrag(const glm::vec2& mouse_pos, const glm::mat4& view,
                             const glm::mat4& projection) override {
            return gizmo_->updateDrag(mouse_pos, view, projection);
        }

        void endDrag() override { gizmo_->endDrag(); }
        bool isDragging() const override { return gizmo_->isDragging(); }

        void setHovered(GizmoElement element) override {
            gizmo_->setHoveredElement(static_cast<TranslationGizmo::Element>(element));
        }

        GizmoElement getHovered() const override {
            return static_cast<GizmoElement>(gizmo_->getHoveredElement());
        }
    };

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

        Result<RenderResult> renderSplitView(
            const SplitViewRequest& request) override;

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

        Result<void> renderTranslationGizmo(
            const glm::vec3& position,
            const ViewportData& viewport,
            float scale) override;

        std::shared_ptr<GizmoInteraction> getGizmoInteraction() override;

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
        std::shared_ptr<ScreenQuadRenderer> screen_renderer_;

        // Split view renderer
        std::unique_ptr<SplitViewRenderer> split_view_renderer_;

        // Overlay renderers
        RenderInfiniteGrid grid_renderer_;
        RenderBoundingBox bbox_renderer_;
        RenderCoordinateAxes axes_renderer_;
        ViewportGizmo viewport_gizmo_;
        TranslationGizmo translation_gizmo_;

        // Gizmo interaction adapter
        std::shared_ptr<GizmoInteractionAdapter> gizmo_interaction_;

        // Shaders
        ManagedShader quad_shader_;
    };

} // namespace gs::rendering