#include "rendering_manager.hpp"
#include "core/splat_data.hpp"
#include "geometry/euclidean_transform.hpp"
#include "rendering/rendering.hpp"
#include "scene/scene_manager.hpp"
#include "tools/background_tool.hpp"
#include "training/training_manager.hpp"

// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

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
        engine_->initialize();

        initialized_ = true;
    }

    void RenderingManager::setupEventHandlers() {
        // Subscribe to SceneLoaded events
        scene_loaded_handler_id_ = events::state::SceneLoaded::when([this]([[maybe_unused]] const auto& event) {
            scene_just_loaded_ = true;
            prev_result_.valid = false; // Invalidate cached result
        });

        // Subscribe to GridSettingsChanged events
        grid_settings_handler_id_ = events::ui::GridSettingsChanged::when([this](const auto& event) {
            settings_.show_grid = event.enabled;
            settings_.grid_plane = event.plane;
            settings_.grid_opacity = event.opacity;
        });

        // Subscribe to SceneChanged events to invalidate cache
        scene_changed_handler_id_ = events::state::SceneChanged::when([this]([[maybe_unused]] const auto& event) {
            prev_result_.valid = false; // Invalidate cached result when scene changes
        });

        // Subscribe to CropBoxChanged events to invalidate cache when crop box changes
        events::ui::CropBoxChanged::when([this]([[maybe_unused]] const auto& event) {
            if (event.enabled) {
                prev_result_.valid = false; // Invalidate cached result when crop box changes
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
        auto state = scene_manager->getCurrentState();
        bool skip_scene_render = false;
        if (settings_.adaptive_frame_rate && !settings_.use_crop_box) {
            skip_scene_render = framerate_controller_.shouldSkipSceneRender(
                state.is_training, scene_changed);
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

        // Render scene first (if we have one)
        if (scene_manager->hasScene()) {
            drawSceneFrame(context, scene_manager, skip_scene_render);
        }

        // Render overlays
        drawOverlays(context);

        // Update last viewport state - use copy assignment with unique_ptr
        *prev_viewport_state_ = context.viewport;

        // Draw focus indicator if viewport has focus
        if (context.has_focus && context.viewport_region) {
            drawFocusIndicator(context);
        }

        // End framerate tracking
        framerate_controller_.endFrame();
    }

    void RenderingManager::drawSceneFrame(const RenderContext& context, SceneManager* scene_manager, bool skip_render) {
        if (!scene_manager->hasScene()) {
            return;
        }

        const Viewport& render_viewport = context.viewport;
        const geometry::BoundingBox* render_crop_box = nullptr;

        // Create a temporary geometry::BoundingBox from the IBoundingBox interface
        std::unique_ptr<geometry::BoundingBox> temp_bbox;
        if (settings_.use_crop_box && context.crop_box) {
            temp_bbox = std::make_unique<geometry::BoundingBox>();
            temp_bbox->setBounds(context.crop_box->getMinBounds(), context.crop_box->getMaxBounds());
            temp_bbox->setworld2BBox(context.crop_box->getworld2BBox());
            render_crop_box = temp_bbox.get();
        }

        auto rot = render_viewport.getRotationMatrix();
        auto trans = render_viewport.getTranslation();
        geometry::BoundingBox bb;

        // shift view matrix and cropbox in case world to user is defined
        if (context.world_to_user) {
            // This is a trick so we dont have to transform the actual gaussians
            glm::mat4 T = glm::mat4(1.0f);
            T[0] = glm::vec4(rot[0], 0.0f);
            T[1] = glm::vec4(rot[1], 0.0f);
            T[2] = glm::vec4(rot[2], 0.0f);
            T[3] = glm::vec4(trans, 1.0f);
            geometry::EuclideanTransform view_cam_to_world(T);

            view_cam_to_world = *context.world_to_user * view_cam_to_world;

            rot = view_cam_to_world.getRotationMat();
            trans = view_cam_to_world.getTranslation();

            if (render_crop_box) {
                bb = *render_crop_box;
                auto world_2_box = bb.getworld2BBox();
                bb.setworld2BBox(world_2_box * context.world_to_user->inv());
                render_crop_box = &bb;
            }
        }

        // Build render request with the viewport region dimensions if available
        glm::ivec2 render_size = render_viewport.windowSize;
        if (context.viewport_region) {
            render_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        // Don't skip render if we're in point cloud mode or if settings changed or if cached result is invalid
        if (prev_result_.valid && skip_render && !settings_.point_cloud_mode) {
            // Use cached result
            if (context.viewport_region) {
                engine_->presentToScreen(
                    prev_result_,
                    glm::ivec2(context.viewport_region->x, context.viewport_region->y),
                    render_size);
            } else {
                engine_->presentToScreen(prev_result_, glm::ivec2(0, 0), render_size);
            }
            return;
        }

        // Get background color
        glm::vec3 background_color(0.0f, 0.0f, 0.0f);
        if (context.background_tool) {
            background_color = context.background_tool->getBackgroundColor();
        }

        // Build render request
        gs::rendering::RenderRequest request{
            .viewport = {
                .rotation = rot,
                .translation = trans,
                .size = render_size,
                .fov = settings_.fov},
            .scaling_modifier = settings_.scaling_modifier,
            .antialiasing = settings_.antialiasing,
            .background_color = background_color,
            .crop_box = std::nullopt,
            .point_cloud_mode = settings_.point_cloud_mode,
            .voxel_size = settings_.voxel_size};

        // Convert crop box if present
        if (render_crop_box) {
            // The transform should be the inverse of world2BBox for rendering
            glm::mat4 transform_mat = render_crop_box->getworld2BBox().inv().toMat4();

            request.crop_box = gs::rendering::BoundingBox{
                .min = render_crop_box->getMinBounds(),
                .max = render_crop_box->getMaxBounds(),
                .transform = transform_mat};
        }

        // Get trainer for potential mutex locking
        auto state = scene_manager->getCurrentState();
        gs::rendering::RenderResult result;

        if (state.is_training && scene_manager->getTrainerManager()) {
            auto trainer = scene_manager->getTrainerManager()->getTrainer();
            if (trainer && trainer->is_running()) {
                std::shared_lock<std::shared_mutex> lock(trainer->getRenderMutex());
                // Get model and render
                if (scene_manager->getScene() && scene_manager->getScene()->hasModel()) {
                    const SplatData* model = scene_manager->getScene()->getModel();
                    if (model) {
                        result = engine_->renderGaussians(*model, request);
                    }
                }
            } else {
                // Get model and render without lock
                if (scene_manager->getScene() && scene_manager->getScene()->hasModel()) {
                    const SplatData* model = scene_manager->getScene()->getModel();
                    if (model) {
                        result = engine_->renderGaussians(*model, request);
                    }
                }
            }
        } else {
            // Get model and render without lock
            if (scene_manager->getScene() && scene_manager->getScene()->hasModel()) {
                const SplatData* model = scene_manager->getScene()->getModel();
                if (model) {
                    result = engine_->renderGaussians(*model, request);
                }
            }
        }

        if (result.valid) {
            if (context.viewport_region) {
                engine_->presentToScreen(
                    result,
                    glm::ivec2(context.viewport_region->x, context.viewport_region->y),
                    render_size);
            } else {
                engine_->presentToScreen(result, glm::ivec2(0, 0), render_size);
            }
            prev_result_ = result;
        }
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
            engine_->renderGrid(
                viewport,
                static_cast<gs::rendering::GridPlane>(settings_.grid_plane),
                settings_.grid_opacity);
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

            engine_->renderBoundingBox(box, viewport, color, line_width);
        }

        // Render coordinate axes
        if (settings_.show_coord_axes && context.coord_axes && engine_) {
            // Get visibility from interface
            std::array<bool, 3> visible = {
                context.coord_axes->isAxisVisible(0),
                context.coord_axes->isAxisVisible(1),
                context.coord_axes->isAxisVisible(2)};
            engine_->renderCoordinateAxes(viewport, 2.0f, visible);
        }
    }

    void RenderingManager::drawFocusIndicator(const RenderContext& context) {
        // Save current OpenGL state
        GLboolean depth_test_enabled = glIsEnabled(GL_DEPTH_TEST);
        GLboolean blend_enabled = glIsEnabled(GL_BLEND);

        // Setup for 2D overlay rendering
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Set viewport to full window for overlay
        glViewport(0, 0, context.viewport.frameBufferSize.x, context.viewport.frameBufferSize.y);

        // Use immediate mode for simple border (or you could use a shader)
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, context.viewport.frameBufferSize.x, context.viewport.frameBufferSize.y, 0, -1, 1);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        // Draw border
        float x = context.viewport_region->x;
        float y = context.viewport_region->y;
        float w = context.viewport_region->width;
        float h = context.viewport_region->height;

        // Animated glow effect
        float time = static_cast<float>(glfwGetTime());
        float glow = (sin(time * 3.0f) + 1.0f) * 0.5f;

        glLineWidth(3.0f);
        glBegin(GL_LINE_LOOP);
        glColor4f(0.2f, 0.6f, 1.0f, 0.5f + glow * 0.3f);
        glVertex2f(x, y);
        glVertex2f(x + w, y);
        glVertex2f(x + w, y + h);
        glVertex2f(x, y + h);
        glEnd();

        // Inner glow
        glLineWidth(1.0f);
        float inset = 1.0f;
        glBegin(GL_LINE_LOOP);
        glColor4f(0.4f, 0.8f, 1.0f, 0.3f + glow * 0.2f);
        glVertex2f(x + inset, y + inset);
        glVertex2f(x + w - inset, y + inset);
        glVertex2f(x + w - inset, y + h - inset);
        glVertex2f(x + inset, y + h - inset);
        glEnd();

        // Restore matrices
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);

        // Restore OpenGL state
        if (depth_test_enabled)
            glEnable(GL_DEPTH_TEST);
        if (!blend_enabled)
            glDisable(GL_BLEND);
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