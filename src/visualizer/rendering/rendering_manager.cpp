#include "rendering/rendering_manager.hpp"
#include "rendering/render_coordinate_axes.hpp"

#include "internal/resource_paths.hpp"
#include "training/training_manager.hpp"

#ifdef CUDA_GL_INTEROP_ENABLED
#include "rendering/cuda_gl_interop.hpp"
#endif

namespace gs::visualizer {

    RenderingManager::RenderingManager() = default;
    RenderingManager::~RenderingManager() = default;

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

    void RenderingManager::renderFrame(const RenderContext& context, SceneManager* scene_manager) {
        if (!initialized_) {
            initialize();
        }

        // Clear with a dark background
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        drawSceneFrame(context, scene_manager);

        if (settings_.show_crop_box && context.crop_box) {
            drawCropBox(context);
        }
        if (settings_.show_coord_axes && context.coord_axes) {
            drawCoordAxes(context);
        }
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

    void RenderingManager::drawSceneFrame(const RenderContext& context, SceneManager* scene_manager) {
        if (!scene_manager->hasScene()) {
            return;
        }

        const geometry::BoundingBox* render_crop_box = nullptr;
        if (settings_.use_crop_box && context.crop_box) {
            render_crop_box = const_cast<RenderBoundingBox*>(context.crop_box);
        }

        auto rot = context.viewport.getRotationMatrix();
        auto trans = context.viewport.getTranslation();
        geometry::BoundingBox bb;

        // shift view matrix and cropbox in case world to user is defined
        if (context.world_to_user) {
            TransformViewAndCropBox(rot, trans, bb, render_crop_box, *context.world_to_user);
            if (render_crop_box) {
                render_crop_box = &bb;
            }
        }

        // Build render request
        RenderingPipeline::RenderRequest request{
            .view_rotation = rot,
            .view_translation = trans,
            .viewport_size = context.viewport.windowSize,
            .fov = settings_.fov,
            .scaling_modifier = settings_.scaling_modifier,
            .antialiasing = settings_.antialiasing,
            .render_mode = RenderMode::RGB,
            .crop_box = render_crop_box};

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
            RenderingPipeline::uploadToScreen(result, *screen_renderer_, context.viewport.windowSize);
            screen_renderer_->render(quad_shader_, context.viewport);
        }
    }

    void RenderingManager::drawCropBox(const RenderContext& context) {
        auto& reso = context.viewport.windowSize;

        if (reso.x <= 0 || reso.y <= 0) {
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
                static_cast<float>(reso.x) / reso.y,
                0.1f,
                1000.0f);
            glm::mat4 view = context.viewport.getViewMatrix();

            crop_box->render(view, projection);
        }
    }

    void RenderingManager::drawCoordAxes(const RenderContext& context) {
        auto& reso = context.viewport.windowSize;

        if (reso.x <= 0 || reso.y <= 0) {
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
                static_cast<float>(reso.x) / reso.y,
                0.1f,
                1000.0f);
            glm::mat4 view = context.viewport.getViewMatrix();

            coord_axes->render(view, projection);
        }
    }

} // namespace gs::visualizer