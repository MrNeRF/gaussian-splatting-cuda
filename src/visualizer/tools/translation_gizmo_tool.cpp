/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/translation_gizmo_tool.hpp"
#include "core/events.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <print>

namespace gs::visualizer::tools {

    TranslationGizmoTool::TranslationGizmoTool() {
        // Initialize with identity transform
        current_transform_ = geometry::EuclideanTransform();
    }

    bool TranslationGizmoTool::initialize(const ToolContext& ctx) {
        // Store context for later use
        tool_context_ = &ctx;

        auto* render_manager = ctx.getRenderingManager();
        if (!render_manager) {
            return false;
        }

        auto* engine = render_manager->getRenderingEngine();
        if (!engine) {
            return false;
        }

        // Get the gizmo interaction interface
        gizmo_interaction_ = engine->getGizmoInteraction();
        if (!gizmo_interaction_) {
            std::println("Failed to get gizmo interaction interface");
            return false;
        }

        // Initialize transform from current world transform
        auto settings = render_manager->getSettings();
        current_transform_ = settings.world_transform;

        std::println("Translation Gizmo Tool initialized");
        return true;
    }

    void TranslationGizmoTool::shutdown() {
        gizmo_interaction_.reset();
        is_dragging_ = false;
        selected_element_ = gs::rendering::GizmoElement::None;
        hovered_element_ = gs::rendering::GizmoElement::None;
        tool_context_ = nullptr;
    }

    void TranslationGizmoTool::onEnabledChanged(bool enabled) {
        if (!enabled && is_dragging_) {
            // Cancel any ongoing drag
            is_dragging_ = false;
            selected_element_ = gs::rendering::GizmoElement::None;
            if (gizmo_interaction_) {
                gizmo_interaction_->endDrag();
            }
        }

        // Don't emit events here - that would create circular dependency
    }

    void TranslationGizmoTool::update([[maybe_unused]] const ToolContext& ctx) {
        if (!isEnabled() || !gizmo_interaction_) {
            return;
        }

        // Update hover state if not dragging
        if (!is_dragging_) {
            // This would be done by input controller passing mouse position
            // For now, we'll handle it in handleMouseMove
        }
    }

    bool TranslationGizmoTool::handleMouseButton(int button, int action, double x, double y,
                                                 const ToolContext& ctx) {
        if (!isEnabled() || !gizmo_interaction_) {
            return false;
        }

        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_PRESS) {
                // Get matrices
                glm::mat4 view = getViewMatrix(ctx);
                glm::mat4 projection = getProjectionMatrix(ctx);
                glm::vec3 position = current_transform_.getTranslation();

                // Check for gizmo hit
                auto hit_element = gizmo_interaction_->pick(
                    glm::vec2(x, y), view, projection, position);

                if (hit_element != gs::rendering::GizmoElement::None) {
                    // Start dragging
                    selected_element_ = hit_element;
                    is_dragging_ = true;
                    drag_start_transform_ = current_transform_;
                    drag_start_gizmo_pos_ = position;

                    // Start the drag operation
                    drag_start_position_ = gizmo_interaction_->startDrag(
                        hit_element, glm::vec2(x, y), view, projection, position);

                    std::println("Started dragging gizmo element: {}", static_cast<int>(hit_element));
                    return true; // Consume the event
                }
            } else if (action == GLFW_RELEASE && is_dragging_) {
                // End dragging
                is_dragging_ = false;
                selected_element_ = gs::rendering::GizmoElement::None;
                gizmo_interaction_->endDrag();

                // Apply the final transform
                updateWorldTransform(ctx);

                std::println("Ended gizmo drag. Final position: ({:.2f}, {:.2f}, {:.2f})",
                             current_transform_.getTranslation().x,
                             current_transform_.getTranslation().y,
                             current_transform_.getTranslation().z);
                return true;
            }
        }

        return false;
    }

    bool TranslationGizmoTool::handleMouseMove(double x, double y, const ToolContext& ctx) {
        if (!isEnabled() || !gizmo_interaction_) {
            return false;
        }

        if (is_dragging_) {
            // Update drag
            glm::mat4 view = getViewMatrix(ctx);
            glm::mat4 projection = getProjectionMatrix(ctx);

            glm::vec3 new_position = gizmo_interaction_->updateDrag(
                glm::vec2(x, y), view, projection);

            // Update transform with new position
            current_transform_ = geometry::EuclideanTransform(
                current_transform_.getRotationMat(),
                new_position);

            // Update world transform in real-time
            updateWorldTransform(ctx);

            return true; // Consume the event
        } else {
            // Update hover state
            glm::mat4 view = getViewMatrix(ctx);
            glm::mat4 projection = getProjectionMatrix(ctx);
            glm::vec3 position = current_transform_.getTranslation();

            auto new_hover = gizmo_interaction_->pick(
                glm::vec2(x, y), view, projection, position);

            if (new_hover != hovered_element_) {
                hovered_element_ = new_hover;
                gizmo_interaction_->setHovered(new_hover);
            }
        }

        return false;
    }

    void TranslationGizmoTool::renderUI([[maybe_unused]] const gs::gui::UIContext& ui_ctx, bool* p_open) {
        if (ImGui::Begin("Translation Gizmo", p_open)) {
            ImGui::Text("Gizmo Settings");
            ImGui::Separator();

            // Enable/disable
            bool enabled = isEnabled();
            if (ImGui::Checkbox("Enable Gizmo", &enabled)) {
                setEnabled(enabled);
            }

            if (enabled) {
                ImGui::Checkbox("Show in Viewport", &show_in_viewport_);
                ImGui::SliderFloat("Gizmo Scale", &gizmo_scale_, 0.5f, 3.0f);

                ImGui::Separator();
                ImGui::Text("Transform");

                // Display current position
                glm::vec3 position = current_transform_.getTranslation();
                if (ImGui::DragFloat3("Position", &position.x, 0.01f)) {
                    current_transform_ = geometry::EuclideanTransform(
                        current_transform_.getRotationMat(),
                        position);
                    if (tool_context_) {
                        updateWorldTransform(*tool_context_);
                    }
                }

                // Reset button
                if (ImGui::Button("Reset Transform")) {
                    current_transform_ = geometry::EuclideanTransform();
                    if (tool_context_) {
                        updateWorldTransform(*tool_context_);
                    }
                }

                // Status
                ImGui::Separator();
                if (is_dragging_) {
                    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Dragging: %s",
                                       selected_element_ == gs::rendering::GizmoElement::XAxis ? "X Axis" : selected_element_ == gs::rendering::GizmoElement::YAxis ? "Y Axis"
                                                                                                        : selected_element_ == gs::rendering::GizmoElement::ZAxis   ? "Z Axis"
                                                                                                        : selected_element_ == gs::rendering::GizmoElement::XYPlane ? "XY Plane"
                                                                                                        : selected_element_ == gs::rendering::GizmoElement::XZPlane ? "XZ Plane"
                                                                                                        : selected_element_ == gs::rendering::GizmoElement::YZPlane ? "YZ Plane"
                                                                                                                                                                    : "Unknown");
                } else if (hovered_element_ != gs::rendering::GizmoElement::None) {
                    ImGui::Text("Hovering: %s",
                                hovered_element_ == gs::rendering::GizmoElement::XAxis ? "X Axis" : hovered_element_ == gs::rendering::GizmoElement::YAxis ? "Y Axis"
                                                                                                : hovered_element_ == gs::rendering::GizmoElement::ZAxis   ? "Z Axis"
                                                                                                : hovered_element_ == gs::rendering::GizmoElement::XYPlane ? "XY Plane"
                                                                                                : hovered_element_ == gs::rendering::GizmoElement::XZPlane ? "XZ Plane"
                                                                                                : hovered_element_ == gs::rendering::GizmoElement::YZPlane ? "YZ Plane"
                                                                                                                                                           : "Unknown");
                } else {
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1), "Ready");
                }
            }
        }
        ImGui::End();
    }

    glm::mat4 TranslationGizmoTool::getViewMatrix(const ToolContext& ctx) const {
        const auto& viewport = ctx.getViewport();
        return viewport.getViewMatrix();
    }

    glm::mat4 TranslationGizmoTool::getProjectionMatrix(const ToolContext& ctx) const {
        const auto& viewport = ctx.getViewport();
        auto* render_manager = ctx.getRenderingManager();
        float fov = render_manager ? render_manager->getSettings().fov : 60.0f;
        return viewport.getProjectionMatrix(fov);
    }

    void TranslationGizmoTool::updateWorldTransform(const ToolContext& ctx) {
        auto* render_manager = const_cast<RenderingManager*>(ctx.getRenderingManager());
        if (!render_manager) {
            return;
        }

        // Update the world transform in rendering settings
        auto settings = render_manager->getSettings();
        settings.world_transform = current_transform_;
        render_manager->updateSettings(settings);

        // Also emit event to update the world transform panel if it exists
        events::ui::RenderSettingsChanged{}.emit();
    }

} // namespace gs::visualizer::tools