#include "rendering_engine_impl.hpp"
#include "core/splat_data.hpp"
#include <print>

namespace gs::rendering {

    std::unique_ptr<RenderingEngine> RenderingEngine::create() {
        return std::make_unique<RenderingEngineImpl>();
    }

    RenderingEngineImpl::RenderingEngineImpl() = default;

    RenderingEngineImpl::~RenderingEngineImpl() {
        if (initialized_) {
            shutdown();
        }
    }

    void RenderingEngineImpl::initialize() {
        if (initialized_)
            return;

        std::println("Initializing rendering engine...");

        // Initialize components
        pipeline_ = std::make_unique<RenderingPipeline>();
        point_cloud_renderer_ = std::make_unique<PointCloudRenderer>();
        point_cloud_renderer_->initialize();

#ifdef CUDA_GL_INTEROP_ENABLED
        screen_renderer_ = std::make_shared<ScreenQuadRendererInterop>(true);
#else
        screen_renderer_ = std::make_shared<ScreenQuadRenderer>();
#endif

        grid_renderer_ = std::make_unique<RenderInfiniteGrid>();
        grid_renderer_->init();

        bbox_renderer_ = std::make_unique<RenderBoundingBox>();
        bbox_renderer_->init();

        axes_renderer_ = std::make_unique<RenderCoordinateAxes>();
        axes_renderer_->init();

        initializeShaders();

        initialized_ = true;
        std::println("Rendering engine initialized successfully");
    }

    void RenderingEngineImpl::shutdown() {
        if (!initialized_)
            return;

        // Cleanup in reverse order
        axes_renderer_.reset();
        bbox_renderer_.reset();
        grid_renderer_.reset();
        quad_shader_.reset();
        screen_renderer_.reset();
        point_cloud_renderer_.reset();
        pipeline_.reset();

        initialized_ = false;
    }

    void RenderingEngineImpl::initializeShaders() {
        std::string shader_path = std::string(SHADER_PATH) + "/";
        quad_shader_ = std::make_shared<Shader>(
            (shader_path + "screen_quad.vert").c_str(),
            (shader_path + "screen_quad.frag").c_str(),
            true);
    }

    RenderResult RenderingEngineImpl::renderGaussians(
        const SplatData& splat_data,
        const RenderRequest& request) {

        if (!initialized_) {
            return RenderResult{.valid = false};
        }

        // Convert to internal pipeline request
        RenderingPipeline::RenderRequest pipeline_req{
            .view_rotation = request.viewport.rotation,
            .view_translation = request.viewport.translation,
            .viewport_size = request.viewport.size,
            .fov = request.viewport.fov,
            .scaling_modifier = request.scaling_modifier,
            .antialiasing = request.antialiasing,
            .render_mode = RenderMode::RGB,
            .crop_box = nullptr, // Handle below
            .background_color = request.background_color,
            .point_cloud_mode = request.point_cloud_mode,
            .voxel_size = request.voxel_size};

        // Convert crop box if present
        std::unique_ptr<geometry::BoundingBox> temp_crop_box;
        if (request.crop_box) {
            temp_crop_box = std::make_unique<geometry::BoundingBox>();
            temp_crop_box->setBounds(request.crop_box->min, request.crop_box->max);
            // Note: transform handling would need to be added to geometry::BoundingBox
            pipeline_req.crop_box = temp_crop_box.get();
        }

        auto pipeline_result = pipeline_->render(splat_data, pipeline_req);

        // Convert result
        RenderResult result;
        result.valid = pipeline_result.valid;
        if (pipeline_result.valid) {
            result.image = std::make_shared<torch::Tensor>(pipeline_result.image);
            result.depth = std::make_shared<torch::Tensor>(pipeline_result.depth);
        }

        return result;
    }

    void RenderingEngineImpl::presentToScreen(
        const RenderResult& result,
        const glm::ivec2& viewport_pos,
        const glm::ivec2& viewport_size) {

        if (!initialized_ || !result.valid || !result.image)
            return;

        // Convert back to internal result type
        RenderingPipeline::RenderResult internal_result;
        internal_result.valid = result.valid;
        internal_result.image = *result.image;
        if (result.depth) {
            internal_result.depth = *result.depth;
        }

        RenderingPipeline::uploadToScreen(internal_result, *screen_renderer_, viewport_size);

        // Set viewport for rendering
        glViewport(viewport_pos.x, viewport_pos.y, viewport_size.x, viewport_size.y);
        screen_renderer_->render(quad_shader_);
    }

    void RenderingEngineImpl::renderGrid(
        const ViewportData& viewport,
        GridPlane plane,
        float opacity) {

        if (!initialized_ || !grid_renderer_)
            return;

        auto view = createViewMatrix(viewport);
        auto proj = createProjectionMatrix(viewport);

        grid_renderer_->setPlane(static_cast<RenderInfiniteGrid::GridPlane>(plane));
        grid_renderer_->setOpacity(opacity);
        grid_renderer_->render(view, proj);
    }

    void RenderingEngineImpl::renderBoundingBox(
        const BoundingBox& box,
        const ViewportData& viewport,
        const glm::vec3& color,
        float line_width) {

        if (!initialized_ || !bbox_renderer_)
            return;

        bbox_renderer_->setBounds(box.min, box.max);
        bbox_renderer_->setColor(color);
        bbox_renderer_->setLineWidth(line_width);

        auto view = createViewMatrix(viewport);
        auto proj = createProjectionMatrix(viewport);

        bbox_renderer_->render(view, proj);
    }

    void RenderingEngineImpl::renderCoordinateAxes(
        const ViewportData& viewport,
        float size,
        const std::array<bool, 3>& visible) {

        if (!initialized_ || !axes_renderer_)
            return;

        axes_renderer_->setSize(size);
        for (int i = 0; i < 3; ++i) {
            axes_renderer_->setAxisVisible(i, visible[i]);
        }

        auto view = createViewMatrix(viewport);
        auto proj = createProjectionMatrix(viewport);

        axes_renderer_->render(view, proj);
    }

    glm::mat4 RenderingEngineImpl::createViewMatrix(const ViewportData& viewport) const {
        glm::mat3 flip_yz = glm::mat3(1, 0, 0, 0, -1, 0, 0, 0, -1);
        glm::mat3 R_inv = glm::transpose(viewport.rotation);
        glm::vec3 t_inv = -R_inv * viewport.translation;

        R_inv = flip_yz * R_inv;
        t_inv = flip_yz * t_inv;

        glm::mat4 view(1.0f);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                view[i][j] = R_inv[i][j];
            }
        }
        view[3][0] = t_inv.x;
        view[3][1] = t_inv.y;
        view[3][2] = t_inv.z;

        return view;
    }

    glm::mat4 RenderingEngineImpl::createProjectionMatrix(const ViewportData& viewport) const {
        float aspect = static_cast<float>(viewport.size.x) / viewport.size.y;
        float fov_rad = glm::radians(viewport.fov);
        return glm::perspective(fov_rad, aspect, 0.1f, 1000.0f);
    }

} // namespace gs::rendering
