#include <cstdlib>

#include "rendering/rendering_manager.hpp"

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
        constexpr int max_buffer_size = 512;

        auto convertPath = [max_buffer_size](const std::wstring& path, const std::string& shader_name, char* buffer) {
            const wchar_t* wstr = path.c_str();
            auto len = wcslen(wstr) + 1;

            if (len > max_buffer_size)
                throw std::runtime_error(std::format("{} path is too long {} > {}", shader_name, len, max_buffer_size));

            std::wcstombs(buffer, wstr, len);
            return buffer;
        };

        char buffer1[max_buffer_size] = {0};
        char buffer2[max_buffer_size] = {0};

        convertPath(gs::visualizer::getShaderPath("screen_quad.vert"), "screen_quad.vert", buffer1);
        convertPath(gs::visualizer::getShaderPath("screen_quad.frag"), "screen_quad.frag", buffer2);

        quad_shader_ = std::make_shared<Shader>(buffer1, buffer2, true);
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
    }

    void RenderingManager::drawSceneFrame(const RenderContext& context, SceneManager* scene_manager) {
        if (!scene_manager->hasScene()) {
            return;
        }

        RenderBoundingBox* render_crop_box = nullptr;
        if (settings_.use_crop_box && context.crop_box) {
            render_crop_box = const_cast<RenderBoundingBox*>(context.crop_box);
        }

        // Build render request
        RenderingPipeline::RenderRequest request{
            .view_rotation = context.viewport.getRotationMatrix(),
            .view_translation = context.viewport.getTranslation(),
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

} // namespace gs::visualizer