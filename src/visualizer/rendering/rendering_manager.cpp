#include "rendering/rendering_manager.hpp"
#include "rendering/render_coordinate_axes.hpp"

#include "internal/resource_paths.hpp"
#include "tools/background_tool.hpp"
#include "training/training_manager.hpp"

#ifdef CUDA_GL_INTEROP_ENABLED
#include "rendering/cuda_gl_interop.hpp"
#endif

namespace gs::visualizer {

    RenderingManager::RenderingManager() {
        setupEventHandlers();
    }

    RenderingManager::~RenderingManager() {
        // Unsubscribe from events
        event::bus().remove<events::state::SceneLoaded>(scene_loaded_handler_id_);
    }

    void RenderingManager::initialize() {
        if (initialized_)
            return;

        initializeShaders();

        // Initialize screen renderer with interop support if available
#ifdef CUDA_GL_INTEROP_ENABLED
        screen_renderer_ = std::make_shared<ScreenQuadRendererInterop>(true);
        std::cout << "CUDA-OpenGL interop enabled for rendering" << std::endl;
#else
        screen_renderer_ = std::make_shared<ScreenQuadRenderer>();
        std::cout << "Using CPU copy for rendering (interop not available)" << std::endl;
#endif

        initialized_ = true;
    }

    void RenderingManager::initializeShaders() {
        quad_shader_ = std::make_shared<Shader>(
            (gs::visualizer::getShaderPath("screen_quad.vert")).string().c_str(),
            (gs::visualizer::getShaderPath("screen_quad.frag")).string().c_str(),
            true);
    }

    void RenderingManager::setupEventHandlers() {
        // Subscribe to SceneLoaded events
        scene_loaded_handler_id_ = events::state::SceneLoaded::when([this]([[maybe_unused]] const auto& event) {
            scene_just_loaded_ = true;
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
        if (settings_.adaptive_frame_rate && not settings_.use_crop_box) {
            // at the moment - I dont want to track the crop box too - so if
            // it is enabled - render every frame
            skip_scene_render = framerate_controller_.shouldSkipSceneRender(
                state.is_training, scene_changed);
        }

        // Don't skip rendering if scene was just loaded
        if (scene_just_loaded_) {
            skip_scene_render = false;
            scene_just_loaded_ = false; // Reset the flag after using it
        }

        // Always clear and setup viewport (this is fast)
        glViewport(0, 0, context.viewport.frameBufferSize.x, context.viewport.frameBufferSize.y);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Set viewport region
        if (context.viewport_region) {
            glViewport(
                static_cast<GLint>(context.viewport_region->x),
                static_cast<GLint>(context.viewport_region->y),
                static_cast<GLsizei>(context.viewport_region->width),
                static_cast<GLsizei>(context.viewport_region->height));
        }

        // Render scene only if not skipping
        drawSceneFrame(context, scene_manager, skip_scene_render);

        // Update last viewport state
        prev_viewport_state_ = context.viewport;

        // Always render UI overlays (these are typically fast)
        if (settings_.show_crop_box && context.crop_box) {
            drawCropBox(context);
        }
        if (settings_.show_coord_axes && context.coord_axes) {
            drawCoordAxes(context);
        }

        // Draw focus indicator if viewport has focus
        if (context.has_focus && context.viewport_region) {
            drawFocusIndicator(context);
        }

        // End framerate tracking
        framerate_controller_.endFrame();
    }

    // case world to user is defined - we shift view matrix and crop box
    // this is a trick so we dont have to transform the actual gaussians
    void TransformViewAndCropBox(glm::mat3& rot, glm::vec3& trans, geometry::BoundingBox& bb,
                                 const geometry::BoundingBox* render_crop_box, const geometry::EuclideanTransform& world_to_user) {
        glm::mat4 T = glm::mat4(1.0f); // identity
        T[0] = glm::vec4(rot[0], 0.0f);
        T[1] = glm::vec4(rot[1], 0.0f);
        T[2] = glm::vec4(rot[2], 0.0f);
        T[3] = glm::vec4(trans, 1.0f); // translation
        geometry::EuclideanTransform view_cam_to_world(T);

        view_cam_to_world = world_to_user * view_cam_to_world;

        rot = view_cam_to_world.getRotationMat();
        trans = view_cam_to_world.getTranslation();

        if (render_crop_box) {
            bb = *render_crop_box;
            auto world_2_box = bb.getworld2BBox();
            bb.setworld2BBox(world_2_box * world_to_user.inv());
            render_crop_box = &bb;
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
        glColor4f(0.2f, 0.6f, 1.0f, 0.5f + glow * 0.3f); // Blue glow
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

    void RenderingManager::drawSceneFrame(const RenderContext& context, SceneManager* scene_manager, bool skip_render) {
        if (!scene_manager->hasScene()) {
            return;
        }
        const Viewport& render_viewport = context.viewport;
        const geometry::BoundingBox* render_crop_box = nullptr;
        if (settings_.use_crop_box && context.crop_box) {
            render_crop_box = const_cast<RenderBoundingBox*>(context.crop_box);
        }

        auto rot = render_viewport.getRotationMatrix();
        auto trans = render_viewport.getTranslation();
        geometry::BoundingBox bb;

        // shift view matrix and cropbox in case world to user is defined
        if (context.world_to_user) {
            TransformViewAndCropBox(rot, trans, bb, render_crop_box, *context.world_to_user);
            if (render_crop_box) {
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

        // Don't skip render if we're in point cloud mode or if settings changed
        if (prev_result_.valid && skip_render && !settings_.point_cloud_mode) {
            RenderingPipeline::uploadToScreen(prev_result_, *screen_renderer_, render_size);
            screen_renderer_->render(quad_shader_);
            return;
        }

        // Get background color
        glm::vec3 background_color(0.0f, 0.0f, 0.0f); // Default black
        if (context.background_tool) {
            background_color = context.background_tool->getBackgroundColor();
        }

        RenderingPipeline::RenderRequest request{
            .view_rotation = rot,
            .view_translation = trans,
            .viewport_size = render_size,
            .fov = settings_.fov,
            .scaling_modifier = settings_.scaling_modifier,
            .antialiasing = settings_.antialiasing,
            .render_mode = RenderMode::RGB,
            .crop_box = render_crop_box,
            .background_color = background_color,
            .point_cloud_mode = settings_.point_cloud_mode, // Pass point cloud settings
            .voxel_size = settings_.voxel_size};

        // Get trainer for potential mutex locking
        auto state = scene_manager->getCurrentState();
        RenderingPipeline::RenderResult result;

        if (state.is_training && scene_manager->getTrainerManager()) {
            auto trainer = scene_manager->getTrainerManager()->getTrainer();
            if (trainer && trainer->is_running()) {
                std::shared_lock<std::shared_mutex> lock(trainer->getRenderMutex());
                result = scene_manager->render(request);
            } else {
                result = scene_manager->render(request);
            }
        } else {
            result = scene_manager->render(request);
        }

        if (result.valid) {
            RenderingPipeline::uploadToScreen(result, *screen_renderer_, render_size);
            screen_renderer_->render(quad_shader_);
            prev_result_ = result;
        }
    }

    void RenderingManager::drawCropBox(const RenderContext& context) {

        glm::ivec2 render_size = context.viewport.windowSize;
        if (context.viewport_region) {
            render_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        if (render_size.x <= 0 || render_size.y <= 0) {
            return;
        }

        auto crop_box = const_cast<RenderBoundingBox*>(context.crop_box);

        if (!crop_box->isInitilized()) {
            crop_box->init();
        }

        if (crop_box->isInitialized()) {
            auto fov_rad = glm::radians(settings_.fov);
            auto projection = glm::perspective(
                static_cast<float>(fov_rad),
                static_cast<float>(render_size.x) / render_size.y,
                0.1f,
                1000.0f);

            glm::mat4 view = context.viewport.getViewMatrix();
            crop_box->render(view, projection);
        }
    }

    void RenderingManager::drawCoordAxes(const RenderContext& context) {

        glm::ivec2 render_size = context.viewport.windowSize;
        if (context.viewport_region) {
            render_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        if (render_size.x <= 0 || render_size.y <= 0) {
            return;
        }

        auto coord_axes = const_cast<RenderCoordinateAxes*>(context.coord_axes);

        if (!coord_axes->isInitialized()) {
            coord_axes->init();
        }

        if (coord_axes->isInitialized()) {

            auto fov_rad = glm::radians(settings_.fov);
            auto projection = glm::perspective(
                static_cast<float>(fov_rad),
                static_cast<float>(render_size.x) / render_size.y,
                0.1f,
                1000.0f);
            glm::mat4 view = context.viewport.getViewMatrix();

            coord_axes->render(view, projection);
        }
    }

    bool RenderingManager::hasCamChanged(const Viewport& current_viewport) {
        // Compare current viewport with last known state
        const float epsilon = 1e-6f;

        bool has_changed = false;
        // Compare viewport parameters that affect rendering
        if (glm::length(current_viewport.getTranslation() - prev_viewport_state_.getTranslation()) > epsilon) {
            has_changed = true;
        }

        if (!has_changed) {
            auto current_rot = current_viewport.getRotationMatrix();
            auto last_rot = prev_viewport_state_.getRotationMatrix();
            auto diff = last_rot - current_rot;

            if (glm::length(diff[0]) + glm::length(diff[1]) + glm::length(diff[2]) > epsilon) {
                has_changed = true;
            }
        }

        if (!has_changed) {
            // Check window size changes
            if (current_viewport.windowSize != prev_viewport_state_.windowSize) {
                has_changed = true;
            }
            if (std::abs(prev_fov_ - settings_.fov) > epsilon) {
                has_changed = true;
            }
        }

        prev_viewport_state_ = current_viewport;
        prev_fov_ = settings_.fov;

        return has_changed;
    }

    bool RenderingManager::hasSceneChanged(const RenderContext& context) {
        // Check if viewport has changed since last frame
        bool scene_changed = hasCamChanged(context.viewport);

        if (!scene_changed && context.world_to_user) {
            const auto& w2u = *context.world_to_user;
            if (!(w2u * prev_world_to_usr_inv_).isIdentity()) {
                scene_changed = true;
                prev_world_to_usr_inv_ = (*context.world_to_user).inv();
            }
        }
        // check is user increased window size
        if (!scene_changed && context.viewport_region) {
            glm::ivec2 render_size = context.viewport.windowSize;
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