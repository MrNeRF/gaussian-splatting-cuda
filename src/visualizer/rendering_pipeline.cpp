#include "visualizer/rendering_pipeline.hpp"
#include <print>

#ifdef CUDA_GL_INTEROP_ENABLED
#include "visualizer/cuda_gl_interop.hpp"
#endif

namespace gs {

    RenderingPipeline::RenderingPipeline()
        : background_(torch::zeros({3}, torch::kFloat32).to(torch::kCUDA)) {
    }

    RenderingPipeline::RenderResult RenderingPipeline::render(
        const SplatData& model,
        const RenderRequest& request) {

        RenderResult result;

        // Validate dimensions
        if (request.viewport_size.x <= 0 || request.viewport_size.y <= 0 ||
            request.viewport_size.x > 16384 || request.viewport_size.y > 16384) {
            result.valid = false;
            return result;
        }

        // Create camera for this frame
        Camera cam = createCamera(request);

        // Perform rendering
        auto output = gs::rasterize(
            cam,
            model,
            background_,
            request.scaling_modifier,
            false, // train
            request.antialiasing,
            request.render_mode,
            request.crop_box);

        result.image = output.image;
        result.depth = output.depth;
        result.valid = true;

        return result;
    }

    void RenderingPipeline::uploadToScreen(
        const RenderResult& result,
        ScreenQuadRenderer& renderer,
        const glm::ivec2& viewport_size) {

        if (!result.valid || !result.image.defined()) {
            return;
        }

#ifdef CUDA_GL_INTEROP_ENABLED
        auto interop_renderer = dynamic_cast<ScreenQuadRendererInterop*>(&renderer);

        if (interop_renderer && interop_renderer->isInteropEnabled()) {
            // Keep data on GPU - convert [C, H, W] to [H, W, C] format
            auto image_hwc = result.image.permute({1, 2, 0}).contiguous();

            if (image_hwc.size(0) == viewport_size.y && image_hwc.size(1) == viewport_size.x) {
                interop_renderer->uploadFromCUDA(image_hwc, viewport_size.x, viewport_size.y);
                return;
            }
        }
#endif

        // Fallback to CPU copy
        auto image = (result.image * 255)
                         .to(torch::kCPU)
                         .to(torch::kU8)
                         .permute({1, 2, 0})
                         .contiguous();

        if (image.size(0) == viewport_size.y &&
            image.size(1) == viewport_size.x &&
            image.data_ptr<unsigned char>()) {
            renderer.uploadData(image.data_ptr<unsigned char>(),
                                viewport_size.x, viewport_size.y);
        }
    }

    Camera RenderingPipeline::createCamera(const RenderRequest& request) {
        // Convert view matrix to camera matrix
        torch::Tensor R_tensor = torch::tensor({request.view_rotation[0][0], request.view_rotation[1][0], request.view_rotation[2][0],
                                                request.view_rotation[0][1], request.view_rotation[1][1], request.view_rotation[2][1],
                                                request.view_rotation[0][2], request.view_rotation[1][2], request.view_rotation[2][2]},
                                               torch::TensorOptions().dtype(torch::kFloat32))
                                     .reshape({3, 3});

        torch::Tensor t_tensor = torch::tensor({request.view_translation[0],
                                                request.view_translation[1],
                                                request.view_translation[2]},
                                               torch::TensorOptions().dtype(torch::kFloat32))
                                     .reshape({3, 1});

        // Convert from view to camera space
        R_tensor = R_tensor.transpose(0, 1);
        t_tensor = -R_tensor.mm(t_tensor).squeeze();

        // Compute field of view
        glm::vec2 fov = computeFov(request.fov,
                                   request.viewport_size.x,
                                   request.viewport_size.y);

        return Camera(
            R_tensor,
            t_tensor,
            fov2focal(fov.x, request.viewport_size.x),
            fov2focal(fov.y, request.viewport_size.y),
            request.viewport_size.x / 2.0f,
            request.viewport_size.y / 2.0f,
            torch::empty({0}, torch::kFloat32),
            torch::empty({0}, torch::kFloat32),
            gsplat::CameraModelType::PINHOLE,
            "render_camera",
            "none",
            "none",
            request.viewport_size.x,
            request.viewport_size.y,
            -1);
    }

    glm::vec2 RenderingPipeline::computeFov(float fov_degrees, int width, int height) {
        float fov_rad = glm::radians(fov_degrees);
        float aspect = static_cast<float>(width) / height;

        return glm::vec2(
            atan(tan(fov_rad / 2.0f) * aspect) * 2.0f,
            fov_rad);
    }

} // namespace gs