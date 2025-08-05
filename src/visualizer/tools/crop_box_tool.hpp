#pragma once

#include "rendering/render_bounding_box.hpp"
#include "tools/tool_base.hpp"
#include <glm/glm.hpp>
#include <memory>

namespace gs::visualizer {

    class CropBoxTool : public ToolBase {
    public:
        CropBoxTool();
        ~CropBoxTool() override;

        std::string_view getName() const override { return "Crop Box"; }
        std::string_view getDescription() const override {
            return "3D bounding box for cropping the scene";
        }

        bool initialize(const ToolContext& ctx) override;
        void shutdown() override;
        void update(const ToolContext& ctx) override;
        void render(const ToolContext& ctx) override;
        void renderUI(const gs::gui::UIContext& ui_ctx, bool* p_open) override;

        // Crop box specific methods
        std::shared_ptr<gs::RenderBoundingBox> getBoundingBox() { return bounding_box_; }
        bool shouldShowBox() const { return show_crop_box_; }
        bool shouldUseBox() const { return use_crop_box_; }

    protected:
        void onEnabledChanged(bool enabled) override;

    private:
        void setupEventHandlers();
        void drawControls(const gs::gui::UIContext& ui_ctx);
        void updateRotationMatrix(float delta_x, float delta_y, float delta_z);

        // Input handling helpers
        bool isMouseOverHandle(const glm::dvec2& mouse_pos) const;
        void startDragging(const glm::dvec2& mouse_pos);
        void updateDragging(const glm::dvec2& mouse_pos);
        void stopDragging();

        std::shared_ptr<gs::RenderBoundingBox> bounding_box_;

        // UI state
        bool show_crop_box_ = false;
        bool use_crop_box_ = false;
        float bbox_color_[3] = {1.0f, 1.0f, 0.0f};
        float line_width_ = 2.0f;

        // For rotation controls
        float rotate_timer_x_ = 0.0f;
        float rotate_timer_y_ = 0.0f;
        float rotate_timer_z_ = 0.0f;

        // Interaction state
        bool is_dragging_ = false;
        glm::dvec2 drag_start_pos_;
        glm::vec3 drag_start_box_min_;
        glm::vec3 drag_start_box_max_;
        enum class DragHandle {
            None,
            MinX,
            MinY,
            MinZ,
            MaxX,
            MaxY,
            MaxZ,
            Center
        } current_handle_ = DragHandle::None;
    };

} // namespace gs::visualizer