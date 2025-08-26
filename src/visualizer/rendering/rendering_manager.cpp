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
            // Keep needs_render flag set so we render on next valid frame
            framerate_controller_.endFrame();
            return;
        }

        // Detect viewport size change and invalidate cache
        if (current_size != last_render_size_) {
            LOG_TRACE("Viewport size changed from {}x{} to {}x{}",
                      last_render_size_.x, last_render_size_.y,
                      current_size.x, current_size.y);
            needs_render_ = true;
            cached_result_ = {}; // Clear cache - it's the wrong resolution!
            last_render_size_ = current_size;
        }

        // Get current model
        const SplatData* model = scene_manager ? scene_manager->getModelForRendering() : nullptr;
        size_t model_ptr = reinterpret_cast<size_t>(model);

        // Detect model switch (PLY -> Training, invisible, etc)
        if (model_ptr != last_model_ptr_) {
            LOG_TRACE("Model pointer changed, clearing cache");
            needs_render_ = true;
            last_model_ptr_ = model_ptr;
            cached_result_ = {}; // Always clear cache on model change
        }

        // Simplified render decision logic for immediate rendering
        bool should_render = false;

        // Store needs_render value before clearing it
        bool needs_render_now = needs_render_.load();

        // ALWAYS render if:
        // 1. We don't have a cached result (first render or cache cleared)
        // 2. We need to render (scene changed, settings changed, etc.)
        // 3. Viewport has focus (for interactivity)
        if (!cached_result_.image || needs_render_now) {
            should_render = true;
            needs_render_ = false; // Clear the flag after deciding to render
            LOG_TRACE("Forcing render: no cache={}, needs_render={}",
                      !cached_result_.image, needs_render_now);
        } else if (context.has_focus) {
            // Always render when viewport has focus for smooth interaction
            should_render = true;
        } else if (scene_manager && scene_manager->hasDataset()) {
            // Throttle training renders only when we have a cache and don't need immediate render
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
            // ALWAYS render when we should OR when there's no model (to clear the view)
            doFullRender(context, scene_manager, model);
        } else if (cached_result_.image) {
            // Only use cache if we have a model and a cached result
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

    void RenderingManager::doFullRender(const RenderContext& context, [[maybe_unused]] SceneManager* scene_manager, const SplatData* model) {
        LOG_TIMER_TRACE("Full render pass");

        glm::ivec2 render_size = context.viewport.windowSize;
        if (context.viewport_region) {
            render_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        // Render model if available
        if (model && model->size() > 0) {
            // Use background color from settings
            glm::vec3 bg_color = settings_.background_color;

            // Create viewport data
            gs::rendering::ViewportData viewport_data{
                .rotation = context.viewport.getRotationMatrix(),
                .translation = context.viewport.getTranslation(),
                .size = render_size,
                .fov = settings_.fov};

            gs::rendering::RenderRequest request{
                .viewport = viewport_data,
                .scaling_modifier = settings_.scaling_modifier,
                .antialiasing = settings_.antialiasing,
                .background_color = bg_color,
                .crop_box = std::nullopt,
                .point_cloud_mode = settings_.point_cloud_mode,
                .voxel_size = settings_.voxel_size,
                .model_transform = std::nullopt // Initialize this field
            };

            // IMPORTANT: Apply world transform to MODEL, not view!
            if (!settings_.world_transform.isIdentity()) {
                // Apply transform to the model instead of the camera
                request.model_transform = settings_.world_transform.toMat4();
                // DON'T transform the viewport data anymore!
            }

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
                    // throw std::runtime_error("Failed to present render result: " + present_result.error());
                }
            } else {
                LOG_ERROR("Failed to render gaussians: {}", render_result.error());
                // throw std::runtime_error("Failed to render gaussians: " + render_result.error());
            }
        }

        // Always render overlays
        renderOverlays(context);
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