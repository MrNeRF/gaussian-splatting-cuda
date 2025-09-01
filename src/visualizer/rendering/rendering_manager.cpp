/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager.hpp"
#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "geometry/euclidean_transform.hpp"
#include "rendering/rendering.hpp"
#include "scene/scene_manager.hpp"
#include "training/training_manager.hpp"
#include <glad/glad.h>
#include <stdexcept>

namespace gs::visualizer {

    RenderingManager::RenderingManager() {
        setupEventHandlers();
    }

    RenderingManager::~RenderingManager() = default;

    void RenderingManager::initialize() {
        if (initialized_)
            return;

        LOG_TIMER("RenderingEngine initialization");

        engine_ = gs::rendering::RenderingEngine::create();
        auto init_result = engine_->initialize();
        if (!init_result) {
            LOG_ERROR("Failed to initialize rendering engine: {}", init_result.error());
            throw std::runtime_error("Failed to initialize rendering engine: " + init_result.error());
        }

        initialized_ = true;
        LOG_INFO("Rendering engine initialized successfully");
    }

    void RenderingManager::setupEventHandlers() {
        // Listen for split view toggle
        events::cmd::ToggleSplitView::when([this](const auto&) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.split_view_enabled = !settings_.split_view_enabled;
            settings_.split_view_offset = 0; // Reset when toggling
            LOG_INFO("Split view: {}", settings_.split_view_enabled ? "enabled" : "disabled");

            // Clear split info when disabled
            if (!settings_.split_view_enabled) {
                std::lock_guard<std::mutex> info_lock(split_info_mutex_);
                current_split_info_ = SplitViewInfo{};
            }

            markDirty();
        });

        // Listen for split position changes
        events::ui::SplitPositionChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.split_position = event.position;
            LOG_TRACE("Split position changed to: {}", event.position);
            markDirty();
        });

        // Listen for settings changes
        events::ui::RenderSettingsChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            if (event.fov) {
                settings_.fov = *event.fov;
                LOG_TRACE("FOV changed to: {}", settings_.fov);
            }
            if (event.scaling_modifier) {
                settings_.scaling_modifier = *event.scaling_modifier;
                LOG_TRACE("Scaling modifier changed to: {}", settings_.scaling_modifier);
            }
            if (event.antialiasing) {
                settings_.antialiasing = *event.antialiasing;
                LOG_TRACE("Antialiasing: {}", settings_.antialiasing ? "enabled" : "disabled");
            }
            if (event.background_color) {
                settings_.background_color = *event.background_color;
                LOG_TRACE("Background color changed");
            }
            markDirty();
        });

        // Window resize
        events::ui::WindowResized::when([this](const auto&) {
            LOG_DEBUG("Window resized, clearing render cache");
            markDirty();
            cached_result_ = {}; // Clear cache on resize since dimensions changed
        });

        // Grid settings
        events::ui::GridSettingsChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.show_grid = event.enabled;
            settings_.grid_plane = event.plane;
            settings_.grid_opacity = event.opacity;
            LOG_TRACE("Grid settings updated - enabled: {}, plane: {}, opacity: {}",
                      event.enabled, event.plane, event.opacity);
            markDirty();
        });

        // Scene changes
        events::state::SceneLoaded::when([this](const auto&) {
            LOG_DEBUG("Scene loaded, marking render dirty");
            markDirty();
        });

        events::state::SceneChanged::when([this](const auto&) {
            markDirty();
        });

        // PLY visibility changes
        events::cmd::SetPLYVisibility::when([this](const auto&) {
            markDirty();
        });

        // PLY added/removed
        events::state::PLYAdded::when([this](const auto&) {
            LOG_DEBUG("PLY added, marking render dirty");
            markDirty();
        });

        events::state::PLYRemoved::when([this](const auto&) {
            LOG_DEBUG("PLY removed, marking render dirty");
            markDirty();
        });

        // Crop box changes
        events::ui::CropBoxChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.crop_min = event.min_bounds;
            settings_.crop_max = event.max_bounds;
            settings_.use_crop_box = event.enabled;
            LOG_TRACE("Crop box updated - enabled: {}", event.enabled);
            markDirty();
        });

        // Point cloud mode changes
        events::ui::PointCloudModeChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.point_cloud_mode = event.enabled;
            settings_.voxel_size = event.voxel_size;
            LOG_DEBUG("Point cloud mode: {}, voxel size: {}",
                      event.enabled ? "enabled" : "disabled", event.voxel_size);
            markDirty();
        });
    }

    void RenderingManager::markDirty() {
        needs_render_ = true;
    }

    void RenderingManager::updateSettings(const RenderSettings& new_settings) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_ = new_settings;
        markDirty();
    }

    RenderSettings RenderingManager::getSettings() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_;
    }

    float RenderingManager::getFovDegrees() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_.fov;
    }

    float RenderingManager::getScalingModifier() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_.scaling_modifier;
    }

    void RenderingManager::setFov(float f) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.fov = f;
        markDirty();
    }

    void RenderingManager::setScalingModifier(float s) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.scaling_modifier = s;
        markDirty();
    }

    void RenderingManager::advanceSplitOffset() {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.split_view_offset++;
        markDirty();
    }

    SplitViewInfo RenderingManager::getSplitViewInfo() const {
        std::lock_guard<std::mutex> lock(split_info_mutex_);
        return current_split_info_;
    }

    gs::rendering::RenderingEngine* RenderingManager::getRenderingEngine() {
        if (!initialized_) {
            initialize();
        }
        return engine_.get();
    }

    void RenderingManager::renderFrame(const RenderContext& context, SceneManager* scene_manager) {
        framerate_controller_.beginFrame();

        if (!initialized_) {
            initialize();
        }

        // Calculate current render size
        glm::ivec2 current_size = context.viewport.windowSize;
        if (context.viewport_region) {
            current_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        // SAFETY CHECK: Don't render with invalid viewport dimensions
        if (current_size.x <= 0 || current_size.y <= 0) {
            LOG_TRACE("Skipping render - invalid viewport size: {}x{}", current_size.x, current_size.y);
            framerate_controller_.endFrame();
            return;
        }

        // Detect viewport size change and invalidate cache
        if (current_size != last_render_size_) {
            LOG_TRACE("Viewport size changed from {}x{} to {}x{}",
                      last_render_size_.x, last_render_size_.y,
                      current_size.x, current_size.y);
            needs_render_ = true;
            cached_result_ = {};
            last_render_size_ = current_size;
        }

        // Get current model
        const SplatData* model = scene_manager ? scene_manager->getModelForRendering() : nullptr;
        size_t model_ptr = reinterpret_cast<size_t>(model);

        // Detect model switch
        if (model_ptr != last_model_ptr_) {
            LOG_TRACE("Model pointer changed, clearing cache");
            needs_render_ = true;
            last_model_ptr_ = model_ptr;
            cached_result_ = {};
        }

        // Always render if split view is enabled
        bool should_render = false;
        bool needs_render_now = needs_render_.load();

        // Check if split view is enabled FIRST
        bool split_view_active = settings_.split_view_enabled && scene_manager &&
                                 scene_manager->getScene().getVisibleNodes().size() >= 2;

        if (!cached_result_.image || needs_render_now || split_view_active) {
            should_render = true;
            needs_render_ = false;
            LOG_TRACE("Forcing render: no cache={}, needs_render={}, split_view={}",
                      !cached_result_.image, needs_render_now, split_view_active);
        } else if (context.has_focus) {
            should_render = true;
        } else if (scene_manager && scene_manager->hasDataset()) {
            const auto* trainer_manager = scene_manager->getTrainerManager();
            if (trainer_manager && trainer_manager->isRunning()) {
                auto now = std::chrono::steady_clock::now();
                if (now - last_training_render_ > std::chrono::seconds(1)) {
                    should_render = true;
                    last_training_render_ = now;
                }
            }
        }

        // Clear and set viewport
        glViewport(0, 0, context.viewport.frameBufferSize.x, context.viewport.frameBufferSize.y);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Set viewport region for rendering
        if (context.viewport_region) {
            glViewport(
                static_cast<GLint>(context.viewport_region->x),
                static_cast<GLint>(context.viewport_region->y),
                static_cast<GLsizei>(context.viewport_region->width),
                static_cast<GLsizei>(context.viewport_region->height));
        }

        if (should_render || !model) {
            doFullRender(context, scene_manager, model);
        } else if (cached_result_.image) {
            glm::ivec2 viewport_pos(0, 0);
            glm::ivec2 render_size = current_size;

            if (context.viewport_region) {
                viewport_pos = glm::ivec2(
                    static_cast<int>(context.viewport_region->x),
                    static_cast<int>(context.viewport_region->y));
            }

            engine_->presentToScreen(cached_result_, viewport_pos, render_size);
            renderOverlays(context);
        }

        framerate_controller_.endFrame();
    }

    void RenderingManager::doFullRender(const RenderContext& context, SceneManager* scene_manager, const SplatData* model) {
        LOG_TIMER_TRACE("Full render pass");

        glm::ivec2 render_size = context.viewport.windowSize;
        if (context.viewport_region) {
            render_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        // Check for split view FIRST
        if (settings_.split_view_enabled && scene_manager) {
            auto visible_nodes = scene_manager->getScene().getVisibleNodes();

            if (visible_nodes.size() >= 2) {
                // Calculate which pair to show
                size_t left_idx = settings_.split_view_offset % visible_nodes.size();
                size_t right_idx = (settings_.split_view_offset + 1) % visible_nodes.size();

                // Update split info
                {
                    std::lock_guard<std::mutex> lock(split_info_mutex_);
                    current_split_info_.enabled = true;
                    current_split_info_.left_name = visible_nodes[left_idx]->name;
                    current_split_info_.right_name = visible_nodes[right_idx]->name;
                }

                renderSplitView(
                    context,
                    visible_nodes[left_idx]->model.get(),
                    visible_nodes[right_idx]->model.get());

                // Render labels on top
                renderSplitLabels(context, visible_nodes[left_idx]->name, visible_nodes[right_idx]->name);

                return;
            }
            // Fall through to single view if < 2 visible
            std::lock_guard<std::mutex> lock(split_info_mutex_);
            current_split_info_ = SplitViewInfo{}; // Clear info
        } else {
            std::lock_guard<std::mutex> lock(split_info_mutex_);
            current_split_info_ = SplitViewInfo{}; // Clear info
        }

        // Render model if available (single view)
        if (model && model->size() > 0) {
            // Use background color from settings
            glm::vec3 bg_color = settings_.background_color;

            // Create viewport data
            gs::rendering::ViewportData viewport_data{
                .rotation = context.viewport.getRotationMatrix(),
                .translation = context.viewport.getTranslation(),
                .size = render_size,
                .fov = settings_.fov};

            // Apply world transform to camera (inverse of model transform)
            if (!settings_.world_transform.isIdentity()) {
                glm::mat3 world_rot = settings_.world_transform.getRotationMat();
                glm::vec3 world_trans = settings_.world_transform.getTranslation();

                // Transform the camera position and rotation (inverse of model transform)
                viewport_data.rotation = glm::transpose(world_rot) * viewport_data.rotation;
                viewport_data.translation = glm::transpose(world_rot) * (viewport_data.translation - world_trans);
            }

            gs::rendering::RenderRequest request{
                .viewport = viewport_data,
                .scaling_modifier = settings_.scaling_modifier,
                .antialiasing = settings_.antialiasing,
                .background_color = bg_color,
                .crop_box = std::nullopt,
                .point_cloud_mode = settings_.point_cloud_mode,
                .voxel_size = settings_.voxel_size,
                .gut = settings_.gut};

            // Add crop box if enabled
            if (settings_.use_crop_box) {
                auto transform = settings_.crop_transform;
                request.crop_box = gs::rendering::BoundingBox{
                    .min = settings_.crop_min,
                    .max = settings_.crop_max,
                    .transform = transform.inv().toMat4()};
            }

            // Render the gaussians
            auto render_result = engine_->renderGaussians(*model, request);
            if (render_result) {
                // Cache the result
                cached_result_ = *render_result;

                // Present to screen
                glm::ivec2 viewport_pos(0, 0);
                if (context.viewport_region) {
                    viewport_pos = glm::ivec2(
                        static_cast<int>(context.viewport_region->x),
                        static_cast<int>(context.viewport_region->y));
                }

                auto present_result = engine_->presentToScreen(
                    cached_result_,
                    viewport_pos,
                    render_size);

                if (!present_result) {
                    LOG_ERROR("Failed to present render result: {}", present_result.error());
                }
            } else {
                LOG_ERROR("Failed to render gaussians: {}", render_result.error());
            }
        }

        // Always render overlays
        renderOverlays(context);
    }

    void RenderingManager::renderSplitView(const RenderContext& context,
                                           const SplatData* left_model,
                                           const SplatData* right_model) {
        // Calculate full viewport size
        glm::ivec2 full_size = context.viewport.windowSize;
        if (context.viewport_region) {
            full_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        int split_x = static_cast<int>(full_size.x * settings_.split_position);

        // Get viewport offset
        glm::ivec2 viewport_offset(0, 0);
        if (context.viewport_region) {
            viewport_offset = glm::ivec2(
                static_cast<int>(context.viewport_region->x),
                static_cast<int>(context.viewport_region->y));
        }

        // Save scissor state
        GLboolean scissor_enabled = glIsEnabled(GL_SCISSOR_TEST);
        GLint scissor_box[4];
        glGetIntegerv(GL_SCISSOR_BOX, scissor_box);

        // Create viewport data for FULL SIZE rendering
        gs::rendering::ViewportData viewport_data{
            .rotation = context.viewport.getRotationMatrix(),
            .translation = context.viewport.getTranslation(),
            .size = full_size, // FULL SIZE for both views
            .fov = settings_.fov};

        // Apply world transform if needed
        if (!settings_.world_transform.isIdentity()) {
            glm::mat3 world_rot = settings_.world_transform.getRotationMat();
            glm::vec3 world_trans = settings_.world_transform.getTranslation();
            viewport_data.rotation = glm::transpose(world_rot) * viewport_data.rotation;
            viewport_data.translation = glm::transpose(world_rot) * (viewport_data.translation - world_trans);
        }

        // Common render request settings
        gs::rendering::RenderRequest base_request{
            .viewport = viewport_data,
            .scaling_modifier = settings_.scaling_modifier,
            .antialiasing = settings_.antialiasing,
            .background_color = settings_.background_color,
            .crop_box = std::nullopt,
            .point_cloud_mode = settings_.point_cloud_mode,
            .voxel_size = settings_.voxel_size,
            .gut = settings_.gut};

        // Add crop box if enabled
        if (settings_.use_crop_box) {
            auto transform = settings_.crop_transform;
            base_request.crop_box = gs::rendering::BoundingBox{
                .min = settings_.crop_min,
                .max = settings_.crop_max,
                .transform = transform.inv().toMat4()};
        }

        // FIRST: Render RIGHT model to full viewport (this will be underneath)
        glViewport(viewport_offset.x, viewport_offset.y, full_size.x, full_size.y);
        if (right_model && right_model->size() > 0) {
            auto result = engine_->renderGaussians(*right_model, base_request);
            if (result) {
                engine_->presentToScreen(*result, viewport_offset, full_size);
            }
        }

        // SECOND: Set scissor to only render LEFT model on the left side (this overlays)
        glEnable(GL_SCISSOR_TEST);
        glScissor(viewport_offset.x, viewport_offset.y, split_x, full_size.y);

        // Render LEFT model to full viewport (but scissored to left side)
        glViewport(viewport_offset.x, viewport_offset.y, full_size.x, full_size.y);
        if (left_model && left_model->size() > 0) {
            auto result = engine_->renderGaussians(*left_model, base_request);
            if (result) {
                engine_->presentToScreen(*result, viewport_offset, full_size);
            }
        }

        // Restore scissor state
        if (!scissor_enabled) {
            glDisable(GL_SCISSOR_TEST);
        } else {
            glScissor(scissor_box[0], scissor_box[1], scissor_box[2], scissor_box[3]);
        }

        // Draw splitter UI elements using OpenGL 3.3+ compatible methods
        renderSplitViewUI(viewport_offset, full_size, split_x);

        // Render overlays on top (grid, axes, etc.)
        renderOverlays(context);
    }

    void RenderingManager::renderSplitViewUI(const glm::ivec2& offset, const glm::ivec2& size, int split_x) {
        // Save state
        GLint prev_viewport[4];
        glGetIntegerv(GL_VIEWPORT, prev_viewport);
        GLboolean depth_test = glIsEnabled(GL_DEPTH_TEST);
        GLboolean blend = glIsEnabled(GL_BLEND);
        GLint blend_src, blend_dst;
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &blend_src);
        glGetIntegerv(GL_BLEND_DST_ALPHA, &blend_dst);

        glViewport(offset.x, offset.y, size.x, size.y);

        // Setup 2D rendering
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, size.x, size.y, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Draw shadow/highlight for depth
        glLineWidth(1.0f);

        // Left shadow
        glBegin(GL_LINES);
        glColor4f(0.0f, 0.0f, 0.0f, 0.3f);
        glVertex2f(split_x - 2, 0);
        glVertex2f(split_x - 2, size.y);
        glEnd();

        // Right highlight
        glBegin(GL_LINES);
        glColor4f(1.0f, 1.0f, 1.0f, 0.3f);
        glVertex2f(split_x + 2, 0);
        glVertex2f(split_x + 2, size.y);
        glEnd();

        // Main splitter line
        glLineWidth(2.0f);
        glBegin(GL_LINES);
        glColor4f(1.0f, 0.85f, 0.0f, 1.0f); // Golden yellow
        glVertex2f(split_x, 0);
        glVertex2f(split_x, size.y);
        glEnd();

        // Draw handle
        float handle_width = 24.0f;
        float handle_height = 48.0f;
        float handle_y = size.y * 0.5f;

        // Handle shadow
        glColor4f(0.0f, 0.0f, 0.0f, 0.3f);
        glBegin(GL_QUADS);
        glVertex2f(split_x - handle_width / 2 + 2, handle_y - handle_height / 2 + 2);
        glVertex2f(split_x + handle_width / 2 + 2, handle_y - handle_height / 2 + 2);
        glVertex2f(split_x + handle_width / 2 + 2, handle_y + handle_height / 2 + 2);
        glVertex2f(split_x - handle_width / 2 + 2, handle_y + handle_height / 2 + 2);
        glEnd();

        // Handle background
        glColor4f(0.15f, 0.15f, 0.15f, 0.9f);
        glBegin(GL_QUADS);
        glVertex2f(split_x - handle_width / 2, handle_y - handle_height / 2);
        glVertex2f(split_x + handle_width / 2, handle_y - handle_height / 2);
        glVertex2f(split_x + handle_width / 2, handle_y + handle_height / 2);
        glVertex2f(split_x - handle_width / 2, handle_y + handle_height / 2);
        glEnd();

        // Handle border
        glLineWidth(2.0f);
        glColor4f(1.0f, 0.85f, 0.0f, 1.0f);
        glBegin(GL_LINE_LOOP);
        glVertex2f(split_x - handle_width / 2, handle_y - handle_height / 2);
        glVertex2f(split_x + handle_width / 2, handle_y - handle_height / 2);
        glVertex2f(split_x + handle_width / 2, handle_y + handle_height / 2);
        glVertex2f(split_x - handle_width / 2, handle_y + handle_height / 2);
        glEnd();

        // Grip dots
        glPointSize(3.0f);
        glColor4f(0.7f, 0.7f, 0.7f, 1.0f);
        glBegin(GL_POINTS);
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                glVertex2f(split_x + i * 6.0f, handle_y + j * 12.0f);
            }
        }
        glEnd();

        // Restore matrices
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        // Restore state
        if (depth_test)
            glEnable(GL_DEPTH_TEST);
        if (!blend)
            glDisable(GL_BLEND);
        glBlendFunc(blend_src, blend_dst);
        glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3]);
    }

    void RenderingManager::renderSplitLabels(const RenderContext& context,
                                             const std::string& left_name,
                                             const std::string& right_name) {
        // We'll render these labels using immediate mode OpenGL for simplicity
        // In a production system, you'd use a proper text renderer

        glm::ivec2 full_size = context.viewport.windowSize;
        if (context.viewport_region) {
            full_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        glm::ivec2 viewport_offset(0, 0);
        if (context.viewport_region) {
            viewport_offset = glm::ivec2(
                static_cast<int>(context.viewport_region->x),
                static_cast<int>(context.viewport_region->y));
        }

        int split_x = static_cast<int>(full_size.x * settings_.split_position);

        // Set up 2D rendering
        glViewport(viewport_offset.x, viewport_offset.y, full_size.x, full_size.y);
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, full_size.x, full_size.y, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Draw background rectangles for labels
        glColor4f(0.0f, 0.0f, 0.0f, 0.7f); // Semi-transparent black

        // Left label background
        glBegin(GL_QUADS);
        glVertex2f(10, 10);
        glVertex2f(10 + left_name.length() * 10, 10);
        glVertex2f(10 + left_name.length() * 10, 35);
        glVertex2f(10, 35);
        glEnd();

        // Right label background
        glBegin(GL_QUADS);
        glVertex2f(split_x + 10, 10);
        glVertex2f(split_x + 10 + right_name.length() * 10, 10);
        glVertex2f(split_x + 10 + right_name.length() * 10, 35);
        glVertex2f(split_x + 10, 35);
        glEnd();

        // Draw text (placeholder - in reality you'd use a text renderer)
        // For now, just draw colored rectangles to indicate where text would be
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f); // White text

        // Restore matrices
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
    }

    void RenderingManager::renderOverlays(const RenderContext& context) {
        glm::ivec2 render_size = context.viewport.windowSize;
        if (context.viewport_region) {
            render_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        if (render_size.x <= 0 || render_size.y <= 0) {
            return;
        }

        gs::rendering::ViewportData viewport{
            .rotation = context.viewport.getRotationMatrix(),
            .translation = context.viewport.getTranslation(),
            .size = render_size,
            .fov = settings_.fov};

        // Grid
        if (settings_.show_grid && engine_) {
            auto grid_result = engine_->renderGrid(
                viewport,
                static_cast<gs::rendering::GridPlane>(settings_.grid_plane),
                settings_.grid_opacity);

            if (!grid_result) {
                LOG_WARN("Failed to render grid: {}", grid_result.error());
            }
        }

        // Crop box wireframe
        if (settings_.show_crop_box && engine_) {
            auto transform = settings_.crop_transform;

            gs::rendering::BoundingBox box{
                .min = settings_.crop_min,
                .max = settings_.crop_max,
                .transform = transform.inv().toMat4()};

            auto bbox_result = engine_->renderBoundingBox(box, viewport, settings_.crop_color, settings_.crop_line_width);
            if (!bbox_result) {
                LOG_WARN("Failed to render bounding box: {}", bbox_result.error());
            }
        }

        // Coordinate axes
        if (settings_.show_coord_axes && engine_) {
            auto axes_result = engine_->renderCoordinateAxes(viewport, settings_.axes_size, settings_.axes_visibility);
            if (!axes_result) {
                LOG_WARN("Failed to render coordinate axes: {}", axes_result.error());
            }
        }

        // Translation gizmo (render last so it's on top)
        if (settings_.show_translation_gizmo && engine_) {
            glm::vec3 gizmo_pos = settings_.world_transform.getTranslation();
            auto gizmo_result = engine_->renderTranslationGizmo(gizmo_pos, viewport, settings_.gizmo_scale);
            if (!gizmo_result) {
                LOG_WARN("Failed to render translation gizmo: {}", gizmo_result.error());
            }
        }
    }
} // namespace gs::visualizer