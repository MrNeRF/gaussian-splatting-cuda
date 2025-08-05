#pragma once

#include "geometry//euclidean_transform.hpp"
#include "rendering/render_coordinate_axes.hpp"
#include "tools/tool_base.hpp"
#include <glm/glm.hpp>
#include <memory>

namespace gs::visualizer {

    class WorldTransformTool : public ToolBase {
    public:
        WorldTransformTool();
        ~WorldTransformTool() override;

        std::string_view getName() const override { return "World Transform"; }
        std::string_view getDescription() const override {
            return "Transform World coordinates";
        }

        bool initialize(const ToolContext& ctx) override;
        void shutdown() override;
        void update(const ToolContext& ctx) override;
        void render(const ToolContext& ctx) override;
        void renderUI(const gs::gui::UIContext& ui_ctx, bool* p_open) override;

        // Input handling
        void registerInputHandlers(InputHandler& handler) override;

        // World box specific methods
        std::shared_ptr<gs::RenderCoordinateAxes> getAxes() { return coordinate_axes_; }

    protected:
        void onEnabledChanged(bool enabled) override;

    private:
        glm::vec3 translation_;
        glm::vec3 angles_rad_;

        void setupEventHandlers();
        void drawControls(const gs::gui::UIContext& ui_ctx);

        std::shared_ptr<gs::RenderCoordinateAxes> coordinate_axes_;

        // UI state
        bool show_axes_ = true;
        float line_width_ = 2.0f;
        float axes_size_ = 2.0f;

        // For rotation controls
        float rotate_timer_x_ = 0.0f;
        float rotate_timer_y_ = 0.0f;
        float rotate_timer_z_ = 0.0f;
    };

} // namespace gs::visualizer
