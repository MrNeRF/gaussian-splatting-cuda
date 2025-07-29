#include "visualizer/scene_manager.hpp"
#include "core/model_providers.hpp"
#include "loader/formats/ply.hpp"
#include "core/training_setup.hpp"
#include "visualizer/event_response_handler.hpp"
#include "visualizer/training_manager.hpp"
#include <chrono>
#include <print>

namespace gs {

    SceneManager::SceneManager(std::shared_ptr<EventBus> event_bus)
        : event_bus_(event_bus) {
        setupEventHandlers();
    }

    SceneManager::~SceneManager() {
        // Event handlers are automatically cleaned up when event bus is destroyed
    }

    void SceneManager::setupEventHandlers() {
        if (!event_bus_)
            return;

        // Command handlers
        event_handler_ids_.push_back(
            event_bus_->subscribe<LoadFileCommand>(
                [this](const LoadFileCommand& cmd) { handleLoadFileCommand(cmd); }));

        event_handler_ids_.push_back(
            event_bus_->subscribe<ClearSceneCommand>(
                [this](const ClearSceneCommand& cmd) { handleClearSceneCommand(cmd); }));

        event_handler_ids_.push_back(
            event_bus_->subscribe<RenderRequestCommand>(
                [this](const RenderRequestCommand& cmd) { handleRenderRequestCommand(cmd); }));

        // Query handlers
        event_handler_ids_.push_back(
            event_bus_->subscribe<QuerySceneStateRequest>(
                [this](const QuerySceneStateRequest& req) { handleSceneStateQuery(req); }));

        event_handler_ids_.push_back(
            event_bus_->subscribe<QueryRenderCapabilitiesRequest>(
                [this](const QueryRenderCapabilitiesRequest& req) { handleRenderCapabilitiesQuery(req); }));

        // Training event handlers
        event_handler_ids_.push_back(
            event_bus_->subscribe<TrainingStartedEvent>(
                [this](const TrainingStartedEvent& event) { handleTrainingStarted(event); }));

        event_handler_ids_.push_back(
            event_bus_->subscribe<TrainingCompletedEvent>(
                [this](const TrainingCompletedEvent& event) { handleTrainingCompleted(event); }));

        event_handler_ids_.push_back(
            event_bus_->subscribe<ModelUpdatedEvent>(
                [this](const ModelUpdatedEvent& event) { handleModelUpdated(event); }));
    }

    void SceneManager::setScene(std::unique_ptr<Scene> scene) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        scene_ = std::move(scene);
        if (scene_) {
            scene_->setEventBus(event_bus_);
        }
        updateSceneState();
    }

    void SceneManager::loadPLY(const std::filesystem::path& path) {
        try {
            loadPLYInternal(path);
        } catch (const std::exception& e) {
            event_bus_->publish(ErrorOccurredEvent{
                ErrorOccurredEvent::Severity::Error,
                ErrorOccurredEvent::Category::IO,
                std::format("Failed to load PLY: {}", e.what()),
                std::format("Path: {}", path.string()),
                "Check if the file exists and is a valid PLY file"});
        }
    }

    void SceneManager::loadDataset(const std::filesystem::path& path,
                                   const param::TrainingParameters& params) {
        try {
            cached_params_ = params; // Cache for potential reloads
            loadDatasetInternal(path, params);
        } catch (const std::exception& e) {
            event_bus_->publish(ErrorOccurredEvent{
                ErrorOccurredEvent::Severity::Error,
                ErrorOccurredEvent::Category::IO,
                std::format("Failed to load dataset: {}", e.what()),
                std::format("Path: {}", path.string()),
                "Check if the dataset format is correct"});
        }
    }

    void SceneManager::clearScene() {
        std::lock_guard<std::mutex> lock(state_mutex_);

        auto old_state = current_state_;

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
        publishSceneStateChanged(old_state, current_state_);
        event_bus_->publish(SceneClearedEvent{});
    }

    SceneManager::SceneState SceneManager::getCurrentState() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return current_state_;
    }

    RenderingPipeline::RenderResult SceneManager::render(
        const RenderingPipeline::RenderRequest& request) {

        if (!scene_) {
            return RenderingPipeline::RenderResult{.valid = false};
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        auto result = scene_->render(request);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto render_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();

        // Publish render completed event
        event_bus_->publish(FrameRenderedEvent{
            render_time,
            1000.0f / render_time, // FPS
            static_cast<int>(current_state_.num_gaussians)});

        return result;
    }

    void SceneManager::handleLoadFileCommand(const LoadFileCommand& cmd) {
        if (cmd.is_dataset && cached_params_) {
            loadDataset(cmd.path, *cached_params_);
        } else if (!cmd.is_dataset) {
            loadPLY(cmd.path);
        } else {
            event_bus_->publish(LogMessageEvent{
                LogMessageEvent::Level::Error,
                "Cannot load dataset without parameters",
                "SceneManager"});
        }
    }

    void SceneManager::handleClearSceneCommand(const ClearSceneCommand& cmd) {
        clearScene();
    }

    void SceneManager::handleRenderRequestCommand(const RenderRequestCommand& cmd) {
        RenderingPipeline::RenderRequest request{
            .view_rotation = cmd.view_rotation,
            .view_translation = cmd.view_translation,
            .viewport_size = cmd.viewport_size,
            .fov = cmd.fov,
            .scaling_modifier = cmd.scaling_modifier,
            .antialiasing = cmd.antialiasing,
            .render_mode = static_cast<RenderMode>(cmd.render_mode),
            .crop_box = static_cast<const BoundingBox*>(cmd.crop_box)};

        auto result = render(request);

        event_bus_->publish(RenderCompletedEvent{
            cmd.request_id,
            result.valid,
            result.valid ? std::nullopt : std::optional<std::string>("Render failed"),
            0.0f // TODO: Add timing
        });
    }

    void SceneManager::handleSceneStateQuery(const QuerySceneStateRequest& request) {
        auto state = getCurrentState();

        QuerySceneStateResponse response;

        // Map internal state to response
        switch (state.type) {
        case SceneType::None:
            response.type = QuerySceneStateResponse::SceneType::None;
            break;
        case SceneType::PLY:
            response.type = QuerySceneStateResponse::SceneType::PLY;
            break;
        case SceneType::Dataset:
            response.type = QuerySceneStateResponse::SceneType::Dataset;
            break;
        }

        response.source_path = state.source_path;
        response.num_gaussians = state.num_gaussians;
        response.is_training = state.is_training;
        response.training_iteration = state.training_iteration;
        response.has_model = hasScene();

        event_bus_->publish(response);
    }

    void SceneManager::handleRenderCapabilitiesQuery(const QueryRenderCapabilitiesRequest& request) {
        QueryRenderCapabilitiesResponse response;

        response.supported_render_modes = {"RGB", "D", "ED", "RGB_D", "RGB_ED"};
        response.supports_antialiasing = true;
        response.supports_depth = true;
        response.max_viewport_width = 4096;
        response.max_viewport_height = 4096;

        event_bus_->publish(response);
    }

    void SceneManager::handleTrainingStarted(const TrainingStartedEvent& event) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (current_state_.type == SceneType::Dataset) {
            current_state_.is_training = true;
            current_state_.training_iteration = 0;
        }
    }

    void SceneManager::handleTrainingCompleted(const TrainingCompletedEvent& event) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (current_state_.type == SceneType::Dataset) {
            current_state_.is_training = false;
            current_state_.training_iteration = event.final_iteration;
        }
    }

    void SceneManager::handleModelUpdated(const ModelUpdatedEvent& event) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        current_state_.num_gaussians = event.num_gaussians;
        if (current_state_.is_training) {
            current_state_.training_iteration = event.iteration;
        }
    }

    void SceneManager::loadPLYInternal(const std::filesystem::path& path) {
        std::println("SceneManager: Loading PLY file: {}", path.string());

        // Clear any existing scene
        clearScene();

        // Load PLY
        auto splat_result = gs::load_ply(path);
        if (!splat_result) {
            throw std::runtime_error(splat_result.error());
        }

        // Create scene if needed
        if (!scene_) {
            scene_ = std::make_unique<Scene>();
            scene_->setEventBus(event_bus_);
        }

        // Set model
        scene_->setStandaloneModel(std::make_unique<SplatData>(std::move(*splat_result)));

        // Update state
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            auto old_state = current_state_;

            current_state_.type = SceneType::PLY;
            current_state_.source_path = path;
            current_state_.num_gaussians = scene_->getModel()->size();
            current_state_.is_training = false;
            current_state_.training_iteration.reset();

            publishSceneStateChanged(old_state, current_state_);
        }

        // Notify
        event_bus_->publish(SceneLoadedEvent{
            scene_.get(),
            path,
            SceneLoadedEvent::SourceType::PLY,
            current_state_.num_gaussians});

        event_bus_->publish(LogMessageEvent{
            LogMessageEvent::Level::Info,
            std::format("Loaded PLY with {} Gaussians", current_state_.num_gaussians),
            "SceneManager"});
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
            scene_->setEventBus(event_bus_);
        }

        // Link scene to trainer
        scene_->linkToTrainer(trainer_manager_->getTrainer());

        // Update state
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            auto old_state = current_state_;

            current_state_.type = SceneType::Dataset;
            current_state_.source_path = path;
            current_state_.num_gaussians = trainer_manager_->getTrainer()
                                               ->get_strategy()
                                               .get_model()
                                               .size();
            current_state_.is_training = false;
            current_state_.training_iteration = 0;

            publishSceneStateChanged(old_state, current_state_);
        }

        // Notify
        size_t num_images = setup_result->dataset->size().value();

        event_bus_->publish(SceneLoadedEvent{
            scene_.get(),
            path,
            SceneLoadedEvent::SourceType::Dataset,
            current_state_.num_gaussians});

        event_bus_->publish(DatasetLoadCompletedEvent{
            path,
            true,
            std::nullopt,
            num_images,
            current_state_.num_gaussians});

        event_bus_->publish(LogMessageEvent{
            LogMessageEvent::Level::Info,
            std::format("Loaded dataset with {} images and {} initial Gaussians",
                        num_images, current_state_.num_gaussians),
            "SceneManager"});
    }

    void SceneManager::updateSceneState() {
        // This would be called when scene changes internally
        // For now, we handle updates through events
    }

    void SceneManager::publishSceneStateChanged(const SceneState& old_state,
                                                const SceneState& new_state) {
        if (!event_bus_)
            return;

        SceneStateChangedEvent event;

        // Map types
        auto mapType = [](SceneType type) -> QuerySceneStateResponse::SceneType {
            switch (type) {
            case SceneType::None: return QuerySceneStateResponse::SceneType::None;
            case SceneType::PLY: return QuerySceneStateResponse::SceneType::PLY;
            case SceneType::Dataset: return QuerySceneStateResponse::SceneType::Dataset;
            default: return QuerySceneStateResponse::SceneType::None;
            }
        };

        event.old_type = mapType(old_state.type);
        event.new_type = mapType(new_state.type);
        event.source_path = new_state.source_path;
        event.num_gaussians = new_state.num_gaussians;

        if (old_state.type != new_state.type) {
            event.change_reason = std::format("Scene type changed from {} to {}",
                                              static_cast<int>(old_state.type),
                                              static_cast<int>(new_state.type));
        }

        event_bus_->publish(event);
    }

} // namespace gs
