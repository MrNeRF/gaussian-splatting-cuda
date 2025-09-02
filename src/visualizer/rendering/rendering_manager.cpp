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
            cached_result_ = {};                  // Clear cache on resize
            last_render_size_ = glm::ivec2(0, 0); // Force size update
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

        // Check for split view
        if (auto split_request = createSplitViewRequest(context, scene_manager)) {
            // Update split info
            {
                std::lock_guard<std::mutex> lock(split_info_mutex_);
                current_split_info_.enabled = true;
                if (split_request->panels.size() >= 2) {
                    current_split_info_.left_name = split_request->panels[0].label;
                    current_split_info_.right_name = split_request->panels[1].label;
                }
            }

            auto result = engine_->renderSplitView(*split_request);
            if (result) {
                // Split view already composites to screen, no need to present again

                // Just store a dummy result to prevent re-rendering every frame
                cached_result_ = *result;
            } else {
                LOG_ERROR("Failed to render split view: {}", result.error());
            }

            renderOverlays(context);
            return;
        }

        // Clear split info if not in split view
        {
            std::lock_guard<std::mutex> lock(split_info_mutex_);
            current_split_info_ = SplitViewInfo{};
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

    std::optional<gs::rendering::SplitViewRequest>
    RenderingManager::createSplitViewRequest(const RenderContext& context, SceneManager* scene_manager) {
        if (!settings_.split_view_enabled || !scene_manager) {
            return std::nullopt;
        }

        auto visible_nodes = scene_manager->getScene().getVisibleNodes();
        if (visible_nodes.size() < 2) {
            return std::nullopt;
        }

        // Calculate which pair to show
        size_t left_idx = settings_.split_view_offset % visible_nodes.size();
        size_t right_idx = (settings_.split_view_offset + 1) % visible_nodes.size();

        // Get render size
        glm::ivec2 render_size = context.viewport.windowSize;
        if (context.viewport_region) {
            render_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        // Create viewport data
        gs::rendering::ViewportData viewport_data{
            .rotation = context.viewport.getRotationMatrix(),
            .translation = context.viewport.getTranslation(),
            .size = render_size,
            .fov = settings_.fov};

        // Apply world transform if needed
        if (!settings_.world_transform.isIdentity()) {
            glm::mat3 world_rot = settings_.world_transform.getRotationMat();
            glm::vec3 world_trans = settings_.world_transform.getTranslation();
            viewport_data.rotation = glm::transpose(world_rot) * viewport_data.rotation;
            viewport_data.translation = glm::transpose(world_rot) * (viewport_data.translation - world_trans);
        }

        // Create crop box if enabled
        std::optional<gs::rendering::BoundingBox> crop_box;
        if (settings_.use_crop_box) {
            auto transform = settings_.crop_transform;
            crop_box = gs::rendering::BoundingBox{
                .min = settings_.crop_min,
                .max = settings_.crop_max,
                .transform = transform.inv().toMat4()};
        }

        return gs::rendering::SplitViewRequest{
            .panels = {
                {.model = visible_nodes[left_idx]->model.get(),
                 .label = visible_nodes[left_idx]->name,
                 .start_position = 0.0f,
                 .end_position = settings_.split_position},
                {.model = visible_nodes[right_idx]->model.get(),
                 .label = visible_nodes[right_idx]->name,
                 .start_position = settings_.split_position,
                 .end_position = 1.0f}},
            .viewport = viewport_data,
            .scaling_modifier = settings_.scaling_modifier,
            .antialiasing = settings_.antialiasing,
            .background_color = settings_.background_color,
            .crop_box = crop_box,
            .point_cloud_mode = settings_.point_cloud_mode,
            .voxel_size = settings_.voxel_size,
            .gut = settings_.gut,
            .show_dividers = true,
            .divider_color = glm::vec4(1.0f, 0.85f, 0.0f, 1.0f),
            .show_labels = true};
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

        // Camera frustums
        if (settings_.show_camera_frustums && engine_) {
            LOG_TRACE("Camera frustums enabled, checking for scene_manager...");

            if (!context.scene_manager) {
                LOG_ERROR("Camera frustums enabled but scene_manager is null in render context!");
                return;
            }

            // Get cameras from scene manager's trainer
            std::vector<std::shared_ptr<const Camera>> cameras;
            auto* trainer_manager = context.scene_manager->getTrainerManager();

            if (!trainer_manager) {
                LOG_WARN("Camera frustums enabled but trainer_manager is null");
                return;
            }

            if (!trainer_manager->hasTrainer()) {
                LOG_TRACE("Camera frustums enabled but no trainer is loaded");
                return;
            }

            cameras = trainer_manager->getCamList();
            LOG_TRACE("Retrieved {} cameras from trainer manager", cameras.size());

            if (!cameras.empty()) {
                LOG_DEBUG("Rendering {} camera frustums with scale {}",
                          cameras.size(), settings_.camera_frustum_scale);

                auto frustum_result = engine_->renderCameraFrustums(
                    cameras, viewport,
                    settings_.camera_frustum_scale,
                    settings_.train_camera_color,
                    settings_.eval_camera_color);

                if (!frustum_result) {
                    LOG_ERROR("Failed to render camera frustums: {}", frustum_result.error());
                } else {
                    LOG_TRACE("Successfully rendered camera frustums");
                }
            } else {
                LOG_WARN("Camera frustums enabled but no cameras available");
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
