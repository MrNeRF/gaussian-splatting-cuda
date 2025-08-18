#include "scene/scene_manager.hpp"
#include "core/model_providers.hpp"
#include "core/training_setup.hpp"
#include "loader/loader.hpp"
#include "rendering/rendering_pipeline.hpp"
#include "training/training_manager.hpp"
#include <chrono>
#include <print>

namespace gs {

    SceneManager::SceneManager() {
        setupEventHandlers();
    }

    SceneManager::~SceneManager() {
        // Cleanup handled automatically
    }

    void SceneManager::setupEventHandlers() {
        using namespace events;

        cmd::ClearScene::when([this](const auto&) {
            clearScene();
        });

        // Handle add PLY command
        cmd::AddPLY::when([this](const auto& cmd) {
            addPLYInternal(cmd.path);
        });

        // Query handlers
        query::GetSceneInfo::when([this](const auto&) {
            auto state = getCurrentState();

            query::SceneInfo response;

            // Map internal state to response
            switch (state.type) {
            case SceneType::None:
                response.type = query::SceneInfo::Type::None;
                break;
            case SceneType::PLY:
                response.type = query::SceneInfo::Type::PLY;
                break;
            case SceneType::Dataset:
                response.type = query::SceneInfo::Type::Dataset;
                break;
            }

            response.source_path = state.source_path;
            response.num_gaussians = state.num_gaussians;
            response.is_training = state.is_training;
            response.has_model = hasScene();

            response.emit();
        });

        query::GetRenderCapabilities::when([this](const auto&) {
            query::RenderCapabilities response;

            response.modes = {"RGB", "D", "ED", "RGB_D", "RGB_ED"};
            response.supports_antialiasing = true;
            response.supports_depth = true;
            response.max_width = 4096;
            response.max_height = 4096;

            response.emit();
        });

        // Training event handlers
        state::TrainingStarted::when([this](const auto&) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (current_state_.type == SceneType::Dataset) {
                current_state_.is_training = true;
                current_state_.training_iteration = 0;
            }
        });

        state::TrainingCompleted::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (current_state_.type == SceneType::Dataset) {
                current_state_.is_training = false;
                current_state_.training_iteration = event.iteration;
            }
        });

        state::ModelUpdated::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            current_state_.num_gaussians = event.num_gaussians;
            if (current_state_.is_training) {
                current_state_.training_iteration = event.iteration;
            }
        });

        // Handle PLY events to update state
        state::PLYAdded::when([this](const auto& event) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (current_state_.type == SceneType::PLY) {
                current_state_.num_gaussians = event.total_gaussians;
                current_state_.num_plys = scene_ ? scene_->getSceneNodes().size() : 0;
            }
        });

        state::PLYRemoved::when([this](const auto&) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (current_state_.type == SceneType::PLY && scene_) {
                current_state_.num_gaussians = scene_->getTotalGaussianCount();
                current_state_.num_plys = scene_->getSceneNodes().size();
            }
        });

        // Handle render requests
        internal::RenderRequest::when([this](const auto& cmd) {
            gs::rendering::RenderingPipeline::RenderRequest request{
                .view_rotation = cmd.view_rotation,
                .view_translation = cmd.view_translation,
                .viewport_size = cmd.viewport_size,
                .fov = cmd.fov,
                .scaling_modifier = cmd.scaling_modifier,
                .antialiasing = cmd.antialiasing,
                .render_mode = static_cast<RenderMode>(cmd.render_mode),
                .crop_box = static_cast<const geometry::BoundingBox*>(cmd.crop_box)};

            auto result = render(request);

            internal::RenderComplete{
                .request_id = cmd.request_id,
                .success = result.valid,
                .render_ms = 0.0f // TODO: Add timing
            }
                .emit();
        });
    }

    void SceneManager::setScene(std::unique_ptr<Scene> scene) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        scene_ = std::move(scene);
        updateSceneState();
    }

    void SceneManager::loadPLY(const std::filesystem::path& path) {
        try {
            loadPLYInternal(path);
        } catch (const std::exception& e) {
            events::notify::Error{
                .message = std::format("Failed to load PLY: {}", e.what()),
                .details = std::format("Path: {}", path.string())}
                .emit();
        }
    }

    void SceneManager::addPLY(const std::filesystem::path& path) {
        try {
            addPLYInternal(path);
        } catch (const std::exception& e) {
            events::notify::Error{
                .message = std::format("Failed to add PLY: {}", e.what()),
                .details = std::format("Path: {}", path.string())}
                .emit();
        }
    }

    void SceneManager::loadDataset(const std::filesystem::path& path,
                                   const param::TrainingParameters& params) {
        try {
            trainer_manager_->clearTrainer(); // mainly to stop training thread
            cached_params_ = params;          // Cache for potential reloads
            loadDatasetInternal(path, params);
        } catch (const std::exception& e) {
            events::notify::Error{
                .message = std::format("Failed to load dataset: {}", e.what()),
                .details = std::format("Path: {}", path.string())}
                .emit();
        }
    }

    void SceneManager::clearScene() {
        // because if we are training while clearing scene - everything crashes...
        events::cmd::StopTraining{}.emit();

        std::lock_guard<std::mutex> lock(state_mutex_);

        // Clear trainer if we're in dataset mode
        if (current_state_.type == SceneType::Dataset && trainer_manager_) {
            trainer_manager_->clearTrainer();
        }

        // Clear scene
        if (scene_) {
            scene_->clearModel();
        }

        // Update state
        current_state_ = SceneState{};

        // Notify
        events::state::SceneCleared{}.emit();
    }

    SceneManager::SceneState SceneManager::getCurrentState() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return current_state_;
    }

    gs::rendering::RenderingPipeline::RenderResult SceneManager::render(
        const gs::rendering::RenderingPipeline::RenderRequest& request) {

        if (!scene_) {
            return gs::rendering::RenderingPipeline::RenderResult(false);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        auto result = scene_->render(request);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto render_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();

        // Get actual gaussian count from scene
        size_t actual_gaussians = 0;
        if (scene_->hasModel()) {
            const SplatData* model = scene_->getModel();
            if (model) {
                actual_gaussians = model->size();
            }
        }

        // Publish render completed event
        events::state::FrameRendered{
            .render_ms = render_time,
            .fps = 1000.0f / render_time,
            .num_gaussians = static_cast<int>(actual_gaussians)}
            .emit();

        return result;
    }

    void SceneManager::loadPLYInternal(const std::filesystem::path& path) {
        std::println("SceneManager: Loading PLY file: {}", path.string());

        // Clear any existing scene
        clearScene();

        // Create scene if needed
        if (!scene_) {
            scene_ = std::make_unique<Scene>();
        }

        // Use the public loader interface
        auto loader = gs::loader::Loader::create();

        // Set up load options
        gs::loader::LoadOptions options{
            .resize_factor = -1,
            .images_folder = "images",
            .validate_only = false,
            .progress = [this](float percent, const std::string& msg) {
                std::println("[{:5.1f}%] {}", percent, msg);
            }};

        // Load the PLY file
        auto load_result = loader->load(path, options);
        if (!load_result) {
            throw std::runtime_error(load_result.error());
        }

        // Extract SplatData from the result
        auto* splat_data = std::get_if<std::shared_ptr<gs::SplatData>>(&load_result->data);
        if (!splat_data || !*splat_data) {
            throw std::runtime_error("Expected PLY file but loader returned different data type");
        }

        // Get gaussian count before moving
        size_t gaussian_count = static_cast<size_t>((*splat_data)->size());
        std::string name = path.stem().string();

        // Update state BEFORE adding to scene
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            current_state_.type = SceneType::PLY;
            current_state_.source_path = path;
            current_state_.num_gaussians = gaussian_count;
            current_state_.is_training = false;
            current_state_.training_iteration.reset();
            current_state_.num_plys = 1;
        }

        // Emit SceneLoaded BEFORE adding PLY
        // This will set the panel to PLY mode and clear any existing nodes
        events::state::SceneLoaded{
            .scene = scene_.get(),
            .path = path,
            .type = events::state::SceneLoaded::Type::PLY,
            .num_gaussians = gaussian_count}
            .emit();

        // Now add the PLY - this will emit PLYAdded event
        scene_->addPLY(name, std::make_unique<SplatData>(std::move(**splat_data)));

        events::notify::Log{
            .level = events::notify::Log::Level::Info,
            .message = std::format("Loaded PLY '{}' with {} Gaussians", name, gaussian_count),
            .source = "SceneManager"}
            .emit();
    }

    void SceneManager::addPLYInternal(const std::filesystem::path& path) {
        std::println("SceneManager: Adding PLY file to scene: {}", path.string());

        // Ensure we have a scene
        if (!scene_) {
            scene_ = std::make_unique<Scene>();
        }

        // If not in PLY mode, do a regular load instead
        if (current_state_.type != SceneType::PLY) {
            loadPLYInternal(path);
            return;
        }

        // Use the public loader interface
        auto loader = gs::loader::Loader::create();

        // Set up load options
        gs::loader::LoadOptions options{
            .resize_factor = -1,
            .images_folder = "images",
            .validate_only = false,
            .progress = [this](float percent, const std::string& msg) {
                std::println("[{:5.1f}%] {}", percent, msg);
            }};

        // Load the PLY file
        auto load_result = loader->load(path, options);
        if (!load_result) {
            throw std::runtime_error(load_result.error());
        }

        // Extract SplatData from the result
        auto* splat_data = std::get_if<std::shared_ptr<gs::SplatData>>(&load_result->data);
        if (!splat_data || !*splat_data) {
            throw std::runtime_error("Expected PLY file but loader returned different data type");
        }

        // Generate unique name if needed
        std::string base_name = path.stem().string();
        std::string name = base_name;
        int counter = 1;

        // Check if name already exists
        auto nodes = scene_->getSceneNodes();
        while (std::any_of(nodes.begin(), nodes.end(),
                           [&name](const auto* node) { return node->name == name; })) {
            name = std::format("{}_{}", base_name, counter++);
        }

        // Get gaussian count before moving
        size_t gaussian_count = static_cast<size_t>((*splat_data)->size());

        // Add to scene graph
        scene_->addPLY(name, std::make_unique<SplatData>(std::move(**splat_data)));

        // Update state AFTER adding to scene
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            current_state_.num_gaussians = scene_->getTotalGaussianCount();
            current_state_.num_plys = scene_->getSceneNodes().size();
        }

        events::notify::Log{
            .level = events::notify::Log::Level::Info,
            .message = std::format("Added PLY '{}' to scene. Total: {} Gaussians in {} models",
                                   name, current_state_.num_gaussians, current_state_.num_plys),
            .source = "SceneManager"}
            .emit();
    }

    void SceneManager::loadDatasetInternal(const std::filesystem::path& path,
                                           const param::TrainingParameters& params) {
        std::println("SceneManager: Loading dataset: {}", path.string());

        // Clear any existing scene
        clearScene();

        // Setup training
        auto dataset_params = params;
        dataset_params.dataset.data_path = path;

        auto setup_result = gs::setupTraining(dataset_params);
        if (!setup_result) {
            throw std::runtime_error(setup_result.error());
        }

        // Pass trainer to manager
        if (trainer_manager_) {
            trainer_manager_->setTrainer(std::move(setup_result->trainer));
        } else {
            throw std::runtime_error("No trainer manager available");
        }

        // Create scene if needed
        if (!scene_) {
            scene_ = std::make_unique<Scene>();
        }

        // Link scene to trainer
        scene_->linkToTrainer(trainer_manager_->getTrainer());

        // Update state
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            current_state_.type = SceneType::Dataset;
            current_state_.source_path = path;
            current_state_.num_gaussians = trainer_manager_->getTrainer()
                                               ->get_strategy()
                                               .get_model()
                                               .size();
            current_state_.is_training = false;
            current_state_.training_iteration = 0;
            current_state_.num_plys = 0;
        }

        // Notify
        size_t num_images = setup_result->dataset->size().value();

        events::state::SceneLoaded{
            .scene = scene_.get(),
            .path = path,
            .type = events::state::SceneLoaded::Type::Dataset,
            .num_gaussians = current_state_.num_gaussians}
            .emit();

        events::state::DatasetLoadCompleted{
            .path = path,
            .success = true,
            .error = std::nullopt,
            .num_images = num_images,
            .num_points = current_state_.num_gaussians}
            .emit();

        events::notify::Log{
            .level = events::notify::Log::Level::Info,
            .message = std::format("Loaded dataset with {} images and {} initial Gaussians",
                                   num_images, current_state_.num_gaussians),
            .source = "SceneManager"}
            .emit();
    }

    void SceneManager::updateSceneState() {
        // This would be called when scene changes internally
        // For now, we handle updates through events
    }

} // namespace gs