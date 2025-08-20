#include "rendering_manager.hpp"
#include "core/splat_data.hpp"
#include "geometry/euclidean_transform.hpp"
#include "rendering/rendering.hpp"
#include "scene/scene_manager.hpp"
#include "tools/background_tool.hpp"
#include "training/training_manager.hpp"

namespace gs::visualizer {

    RenderingManager::RenderingManager()
        : prev_viewport_state_(std::make_unique<Viewport>()),
          prev_world_to_usr_inv_(std::make_unique<geometry::EuclideanTransform>()) {
        setupEventHandlers();
    }

    RenderingManager::~RenderingManager() {
        // Unsubscribe from events
        event::bus().remove<events::state::SceneLoaded>(scene_loaded_handler_id_);
        event::bus().remove<events::ui::GridSettingsChanged>(grid_settings_handler_id_);
        event::bus().remove<events::state::SceneChanged>(scene_changed_handler_id_);
    }

    void RenderingManager::initialize() {
        if (initialized_)
            return;

        engine_ = gs::rendering::RenderingEngine::create();
        auto init_result = engine_->initialize();
        if (!init_result) {
            // Log error and throw exception since this is critical
            events::notify::Error{
                .message = "Failed to initialize rendering engine",
                .details = init_result.error()}
                .emit();
            throw std::runtime_error("Failed to initialize rendering engine: " + init_result.error());
        }

        initialized_ = true;
    }

    void RenderingManager::setupEventHandlers() {
        // Subscribe to SceneLoaded events
        scene_loaded_handler_id_ = events::state::SceneLoaded::when([this]([[maybe_unused]] const auto& event) {
            scene_just_loaded_ = true;
            prev_result_ = gs::rendering::RenderResult{}; // Clear cached result
        });

        // Subscribe to GridSettingsChanged events
        grid_settings_handler_id_ = events::ui::GridSettingsChanged::when([this](const auto& event) {
            settings_.show_grid = event.enabled;
            settings_.grid_plane = event.plane;
            settings_.grid_opacity = event.opacity;
        });

        // Subscribe to SceneChanged events to invalidate cache
        scene_changed_handler_id_ = events::state::SceneChanged::when([this]([[maybe_unused]] const auto& event) {
            prev_result_ = gs::rendering::RenderResult{}; // Clear cached result when scene changes
        });

        // Subscribe to CropBoxChanged events to invalidate cache when crop box changes
        events::ui::CropBoxChanged::when([this]([[maybe_unused]] const auto& event) {
            if (event.enabled) {
                prev_result_ = gs::rendering::RenderResult{}; // Clear cached result when crop box changes
            }
        });
    }

    void RenderingManager::renderFrame(const RenderContext& context, SceneManager* scene_manager) {
        // Begin framerate tracking
        framerate_controller_.beginFrame();

        if (!initialized_) {
            initialize();
        }

        bool scene_changed = hasSceneChanged(context);

        // Check if we should skip scene rendering
        bool skip_scene_render = false;
        if (settings_.adaptive_frame_rate && !settings_.use_crop_box && scene_manager) {
            // Get state to check if training
            bool is_training = scene_manager->isTraining();
            if (is_training) {
                auto state = scene_manager->getState();
                if (auto* training_state = std::get_if<SceneManager::TrainingState>(&state)) {
                    is_training = training_state->is_running;
                }
            }

            skip_scene_render = framerate_controller_.shouldSkipSceneRender(
                is_training, scene_changed);
        }

        // Don't skip rendering if scene was just loaded
        if (scene_just_loaded_) {
            skip_scene_render = false;
            scene_just_loaded_ = false;
        }

        // Always clear and setup viewport (this is fast)
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

        // Simplified scene access
        if (!skip_scene_render && scene_manager) {
            const SplatData* model = scene_manager->getModelForRendering();
            if (model && model->size() > 0) {
                // Prepare render request
                glm::ivec2 render_size = context.viewport.windowSize;
                if (context.viewport_region) {
                    render_size = glm::ivec2(
                        static_cast<int>(context.viewport_region->width),
                        static_cast<int>(context.viewport_region->height));
                }

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
                    // Present to screen
                    glm::ivec2 viewport_pos(0, 0);
                    if (context.viewport_region) {
                        viewport_pos = glm::ivec2(
                            static_cast<int>(context.viewport_region->x),
                            static_cast<int>(context.viewport_region->y));
                    }

                    auto present_result = engine_->presentToScreen(
                        *render_result,
                        viewport_pos,
                        render_size);

                    if (!present_result) {
                        events::notify::Error{
                            .message = "Failed to present render result",
                            .details = present_result.error()}
                            .emit();
                    }

                    // Cache the result for potential reuse
                    prev_result_ = *render_result;
                } else {
                    events::notify::Error{
                        .message = "Failed to render gaussians",
                        .details = render_result.error()}
                        .emit();
                }
            }
        } else if (skip_scene_render && prev_result_.image) {
            // Reuse cached result if we're skipping scene render
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

            auto present_result = engine_->presentToScreen(
                prev_result_,
                viewport_pos,
                render_size);

            if (!present_result) {
                // Don't spam errors for cached results
            }
        }

        // Render overlays
        drawOverlays(context);

        // Update last viewport state - use copy assignment with unique_ptr
        *prev_viewport_state_ = context.viewport;

        // End framerate tracking
        framerate_controller_.endFrame();
    }
    void RenderingManager::drawOverlays(const RenderContext& context) {
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

        // Render grid
        if (settings_.show_grid && engine_) {
            auto grid_result = engine_->renderGrid(
                viewport,
                static_cast<gs::rendering::GridPlane>(settings_.grid_plane),
                settings_.grid_opacity);

            if (!grid_result) {
                events::notify::Error{
                    .message = "Failed to render grid",
                    .details = grid_result.error()}
                    .emit();
            }
        }

        // Render crop box
        if (settings_.show_crop_box && context.crop_box && engine_) {
            // Get the actual transform from the crop box
            auto transform = context.crop_box->getworld2BBox();

            // Convert to mat4
            glm::mat4 transform_mat = transform.inv().toMat4();

            // Convert from interface to rendering type
            gs::rendering::BoundingBox box{
                .min = context.crop_box->getMinBounds(),
                .max = context.crop_box->getMaxBounds(),
                .transform = transform_mat};

            // Get color and line width from the crop box
            glm::vec3 color = context.crop_box->getColor();
            float line_width = context.crop_box->getLineWidth();

            auto bbox_result = engine_->renderBoundingBox(box, viewport, color, line_width);
            if (!bbox_result) {
                events::notify::Error{
                    .message = "Failed to render bounding box",
                    .details = bbox_result.error()}
                    .emit();
            }
        }

        // Render coordinate axes
        if (settings_.show_coord_axes && context.coord_axes && engine_) {
            // Get visibility from interface
            std::array<bool, 3> visible = {
                context.coord_axes->isAxisVisible(0),
                context.coord_axes->isAxisVisible(1),
                context.coord_axes->isAxisVisible(2)};

            auto axes_result = engine_->renderCoordinateAxes(viewport, 2.0f, visible);
            if (!axes_result) {
                events::notify::Error{
                    .message = "Failed to render coordinate axes",
                    .details = axes_result.error()}
                    .emit();
            }
        }
    }

    bool RenderingManager::hasCamChanged(const Viewport& current_viewport) {
        // Compare current viewport with last known state
        const float epsilon = 1e-6f;

        bool has_changed = false;
        // Compare viewport parameters that affect rendering
        if (glm::length(current_viewport.getTranslation() - prev_viewport_state_->getTranslation()) > epsilon) {
            has_changed = true;
        }

        if (!has_changed) {
            auto current_rot = current_viewport.getRotationMatrix();
            auto last_rot = prev_viewport_state_->getRotationMatrix();
            auto diff = last_rot - current_rot;

            if (glm::length(diff[0]) + glm::length(diff[1]) + glm::length(diff[2]) > epsilon) {
                has_changed = true;
            }
        }

        if (!has_changed) {
            // Check window size changes
            if (current_viewport.windowSize != prev_viewport_state_->windowSize) {
                has_changed = true;
            }
            if (std::abs(prev_fov_ - settings_.fov) > epsilon) {
                has_changed = true;
            }
        }

        prev_fov_ = settings_.fov;

        return has_changed;
    }

    bool RenderingManager::hasSceneChanged(const RenderContext& context) {
        // Check if viewport has changed since last frame
        bool scene_changed = hasCamChanged(context.viewport);

        if (!scene_changed && context.world_to_user) {
            const auto& w2u = *context.world_to_user;
            if (!(w2u * *prev_world_to_usr_inv_).isIdentity()) {
                scene_changed = true;
                *prev_world_to_usr_inv_ = context.world_to_user->inv();
            }
        }

        // check if user increased window size
        if (!scene_changed && context.viewport_region) {
            glm::ivec2 render_size = context.viewport.windowSize;
            if (context.viewport_region) {
                render_size = glm::ivec2(
                    static_cast<int>(context.viewport_region->width),
                    static_cast<int>(context.viewport_region->height));
            }
            if (render_size != prev_render_size_) {
                scene_changed = true;
                prev_render_size_ = render_size;
            }
        }

        if (!scene_changed && context.background_tool) {
            auto background_color = context.background_tool->getBackgroundColor();
            if (glm::length(background_color - prev_background_color_) > 0) {
                scene_changed = true;
                prev_background_color_ = background_color;
            }
        }

        // Check if point cloud mode or voxel size changed
        if (settings_.point_cloud_mode != prev_point_cloud_mode_ ||
            std::abs(settings_.voxel_size - prev_voxel_size_) > 1e-6f) {
            scene_changed = true;
            prev_point_cloud_mode_ = settings_.point_cloud_mode;
            prev_voxel_size_ = settings_.voxel_size;
        }

        return scene_changed;
    }

} // namespace gs::visualizer