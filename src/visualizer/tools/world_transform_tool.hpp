#pragma once

#include "geometry/euclidean_transform.hpp"
#include "rendering/rendering.hpp"
#include "tools/tool_base.hpp"
#include <memory>

namespace gs::visualizer {

    class WorldTransformTool : public ToolBase {
    public:
        WorldTransformTool();
        ~WorldTransformTool() override;

        // Tool interface
        std::string_view getName() const override { return "World Transform"; }
        std::string_view getDescription() const override {
            return "Transform the world coordinate system";
        }

        // ToolBase overrides
        bool initialize(const ToolContext& ctx) override;
        void shutdown() override;
        void update(const ToolContext& ctx) override;
        void render(const ToolContext& ctx) override;
        void renderUI(const gui::UIContext& ctx, bool* open) override;

        // World transform specific methods
        void setTransform(const geometry::EuclideanTransform& transform);
        std::shared_ptr<const geometry::EuclideanTransform> GetTransform() const { return transform_; }

        bool IsTrivialTrans() const;
        bool ShouldShowAxes() const { return show_axes_; }

        std::shared_ptr<const gs::rendering::ICoordinateAxes> getAxes() const { return coordinate_axes_; }

    private:
        void drawControls(const gui::UIContext& ctx);
        void drawTransformInfo();
        void resetTransform();
        void updateAxes();

        // Transform state
        std::shared_ptr<geometry::EuclideanTransform> transform_;

        // UI state
        bool show_axes_ = true;
        float axes_size_ = 2.0f;
        float line_width_ = 3.0f;

        // Axes rendering
        std::shared_ptr<gs::rendering::ICoordinateAxes> coordinate_axes_;

        // Transform parameters for UI
        float rotation_[3] = {0.0f, 0.0f, 0.0f}; // Euler angles in degrees
        float translation_[3] = {0.0f, 0.0f, 0.0f};
    };

} // namespace gs::visualizer