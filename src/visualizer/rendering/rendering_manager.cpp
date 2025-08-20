#include "rendering_manager.hpp"
#include "core/splat_data.hpp"
#include "geometry/euclidean_transform.hpp"
#include "rendering/rendering.hpp"
#include "scene/scene_manager.hpp"
#include "tools/background_tool.hpp"
#include "training/training_manager.hpp"
#include <print>

namespace gs::visualizer {

    RenderingManager::RenderingManager() {
        setupEventHandlers();
    }

    RenderingManager::~RenderingManager() = default;

    void RenderingManager::initialize() {
        if (initialized_)
            return;

        engine_ = gs::rendering::RenderingEngine::create();
        auto init_result = engine_->initialize();
        if (!init_result) {
            events::notify::Error{
                .message = "Failed to initialize rendering engine",
                .details = init_result.error()}
                .emit();
            throw std::runtime_error("Failed to initialize rendering engine: " + init_result.error());
        }

        initialized_ = true;
    }

    void RenderingManager::setupEventHandlers() {
        // Listen for settings changes
        events::ui::RenderSettingsChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            if (event.fov) {
                settings_.fov = *event.fov;
            }
            if (event.scaling_modifier) {
                settings_.scaling_modifier = *event.scaling_modifier;
            }
            if (event.antialiasing) {
                settings_.antialiasing = *event.antialiasing;
            }
            markDirty();
        });

        // Grid settings
        events::ui::GridSettingsChanged::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.show_grid = event.enabled;
            settings_.grid_plane = event.plane;
            settings_.grid_opacity = event.opacity;
            markDirty();
        });

        // Scene changes
        events::state::SceneLoaded::when([this](const auto&) {
            markDirty();
        });

        events::state::SceneChanged::when([this](const auto&) {
            markDirty();
        });

        // Crop box changes
        events::ui::CropBoxChanged::when([this](const auto&) {
            markDirty();
        });

        // Point cloud mode changes
        events::ui::PointCloudModeChanged::when([this](const auto&) {
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

        // Get current model
        const SplatData* model = scene_manager ? scene_manager->getModelForRendering() : nullptr;
        size_t model_ptr = reinterpret_cast<size_t>(model);

        // Detect model switch (PLY -> Training, etc)
        if (model_ptr != last_model_ptr_) {
            needs_render_ = true;
            last_model_ptr_ = model_ptr;
            cached_result_ = {}; // Clear cache on model switch
        }

        // Check if camera moved
        bool camera_moved = context.has_focus;

        // Decision: should we render?
        bool should_render = false;

        if (needs_render_) {
            should_render = true;
            needs_render_ = false;
        } else if (camera_moved) {
            should_render = true;
        } else if (scene_manager && scene_manager->hasDataset()) {
            // Check if actively training
            const auto* trainer_manager = scene_manager->getTrainerManager();
            if (trainer_manager && trainer_manager->isRunning()) {
                // Throttle training renders to 1 FPS
                auto now = std::chrono::steady_clock::now();
                if (now - last_training_render_ > std::chrono::seconds(1)) {
                    should_render = true;
                    last_training_render_ = now;
                }
            }
        }

        // Setup viewport
        glViewport(0, 0, context.viewport.frameBufferSize.x, context.viewport.frameBufferSize.y);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Set viewport region for all subsequent rendering
        if (context.viewport_region) {
            glViewport(
                static_cast<GLint>(context.viewport_region->x),
                static_cast<GLint>(context.viewport_region->y),
                static_cast<GLsizei>(context.viewport_region->width),
                static_cast<GLsizei>(context.viewport_region->height));
        }

        if (should_render) {
            doFullRender(context, scene_manager, model);
        } else if (cached_result_.image) {
            // Reuse cached model result
            glm::ivec2 viewport_pos(0, 0);
            glm::ivec2 render_size = context.viewport.windowSize;

            if (context.viewport_region) {
                viewport_pos = glm::ivec2(
                    static_cast<int>(context.viewport_region->x),
                    static_cast<int>(context.viewport_region->y));
                render_size = glm::ivec2(
                    static_cast<int>(context.viewport_region->width),
                    static_cast<int>(context.viewport_region->height));
            }

            engine_->presentToScreen(cached_result_, viewport_pos, render_size);

            // CRITICAL FIX: Always render overlays even when using cached model!
            renderOverlays(context);
        }

        framerate_controller_.endFrame();
    }

    void RenderingManager::doFullRender(const RenderContext& context, SceneManager* scene_manager, const SplatData* model) {
        glm::ivec2 render_size = context.viewport.windowSize;
        if (context.viewport_region) {
            render_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        // Clear before starting
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Enable depth testing for model
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);

        // 1. Render model (expensive, cache it)
        if (model && model->size() > 0) {
            // Get background color
            glm::vec3 bg_color = glm::vec3(0.0f, 0.0f, 0.0f);
            if (context.background_tool) {
                bg_color = context.background_tool->getBackgroundColor();
            }

            gs::rendering::RenderRequest request{
                .viewport = {
                    .rotation = context.viewport.getRotationMatrix(),
                    .translation = context.viewport.getTranslation(),
                    .size = render_size,
                    .fov = settings_.fov},
                .scaling_modifier = settings_.scaling_modifier,
                .antialiasing = settings_.antialiasing,
                .background_color = bg_color,
                .crop_box = std::nullopt,
                .point_cloud_mode = settings_.point_cloud_mode,
                .voxel_size = settings_.voxel_size};

            // Add crop box if enabled
            if (settings_.use_crop_box && context.crop_box) {
                auto transform = context.crop_box->getworld2BBox();
                request.crop_box = gs::rendering::BoundingBox{
                    .min = context.crop_box->getMinBounds(),
                    .max = context.crop_box->getMaxBounds(),
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
                    events::notify::Error{
                        .message = "Failed to present render result",
                        .details = present_result.error()}
                        .emit();
                }
            } else {
                events::notify::Error{
                    .message = "Failed to render gaussians",
                    .details = render_result.error()}
                    .emit();
            }
        }

        // 2. Render overlays (with proper depth handling)
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

        // CRITICAL: Clear depth buffer so overlays are always visible on top
        glClear(GL_DEPTH_BUFFER_BIT);

        // Enable depth testing but with special settings for overlays
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);

        // Enable blending for transparency
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // 1. Grid - render first with depth writing
        if (settings_.show_grid && engine_) {
            // Grid should write to depth buffer
            glDepthMask(GL_TRUE);

            auto grid_result = engine_->renderGrid(
                viewport,
                static_cast<gs::rendering::GridPlane>(settings_.grid_plane),
                settings_.grid_opacity);

            if (!grid_result) {
                std::println("Failed to render grid: {}", grid_result.error());
            }
        }

        // 2. Crop box wireframe - render without depth writing
        if (settings_.show_crop_box && context.crop_box && engine_) {
            // Disable depth writing for wireframe
            glDepthMask(GL_FALSE);

            auto transform = context.crop_box->getworld2BBox();

            gs::rendering::BoundingBox box{
                .min = context.crop_box->getMinBounds(),
                .max = context.crop_box->getMaxBounds(),
                .transform = transform.inv().toMat4()};

            glm::vec3 color = context.crop_box->getColor();
            float line_width = context.crop_box->getLineWidth();

            auto bbox_result = engine_->renderBoundingBox(box, viewport, color, line_width);
            if (!bbox_result) {
                std::println("Failed to render bounding box: {}", bbox_result.error());
            }

            // Re-enable depth writing
            glDepthMask(GL_TRUE);
        }

        // 3. Coordinate axes - always on top
        if (settings_.show_coord_axes && context.coord_axes && engine_) {
            // Disable depth testing completely for axes
            glDisable(GL_DEPTH_TEST);

            std::array<bool, 3> visible = {
                context.coord_axes->isAxisVisible(0),
                context.coord_axes->isAxisVisible(1),
                context.coord_axes->isAxisVisible(2)};

            auto axes_result = engine_->renderCoordinateAxes(viewport, 2.0f, visible);
            if (!axes_result) {
                std::println("Failed to render coordinate axes: {}", axes_result.error());
            }

            // Re-enable depth testing
            glEnable(GL_DEPTH_TEST);
        }

        // Restore default OpenGL state
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);
    }
} // namespace gs::visualizer