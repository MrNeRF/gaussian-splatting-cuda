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
        glm::mat4 orthonormalizeRotation(const glm::mat4& matrix);
        static float wrapAngle(float angle);

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
    };

} // namespace gs::visualizer