#include "scene/scene_manager.hpp"
#include "core/training_setup.hpp"
#include "loader/loader.hpp"
#include "training/training_manager.hpp"
#include <print>

namespace gs {

    SceneManager::SceneManager() {
        setupEventHandlers();
    }

    SceneManager::~SceneManager() = default;

    void SceneManager::setupEventHandlers() {
        using namespace events;

        // Handle PLY commands
        cmd::AddPLY::when([this](const auto& cmd) {
            addPLY(cmd.path, cmd.name);
        });

        cmd::RemovePLY::when([this](const auto& cmd) {
            removePLY(cmd.name);
        });

        cmd::SetPLYVisibility::when([this](const auto& cmd) {
            setPLYVisibility(cmd.name, cmd.visible);
        });

        cmd::ClearScene::when([this](const auto&) {
            clear();
        });
<<<<<<< Updated upstream

        // Training state updates
        state::TrainingStarted::when([this](const auto&) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (auto* training = std::get_if<TrainingState>(&state_)) {
                training->is_running = true;
                training->current_iteration = 0;
            }
        });

        state::TrainingProgress::when([this](const auto& e) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (auto* training = std::get_if<TrainingState>(&state_)) {
                training->current_iteration = e.iteration;
            }
        });

        state::TrainingCompleted::when([this](const auto& e) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (auto* training = std::get_if<TrainingState>(&state_)) {
                training->is_running = false;
                training->current_iteration = e.iteration;
            }
        });

        state::TrainingStopped::when([this](const auto&) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (auto* training = std::get_if<TrainingState>(&state_)) {
                training->is_running = false;
            }
        });
=======
>>>>>>> Stashed changes
    }

    void SceneManager::loadPLY(const std::filesystem::path& path) {
        try {
            std::println("SceneManager: Loading PLY file: {}", path.string());

            // Clear existing scene
            clear();

            // Load the PLY
            auto loader = gs::loader::Loader::create();
            gs::loader::LoadOptions options{
                .resize_factor = -1,
                .images_folder = "images",
                .validate_only = false};

            auto load_result = loader->load(path, options);
            if (!load_result) {
                throw std::runtime_error(load_result.error());
            }

            auto* splat_data = std::get_if<std::shared_ptr<gs::SplatData>>(&load_result->data);
            if (!splat_data || !*splat_data) {
                throw std::runtime_error("Expected PLY file but got different data type");
            }

            // Add to scene
            std::string name = path.stem().string();
            scene_.addNode(name, std::make_unique<SplatData>(std::move(**splat_data)));

            // Transition to viewing state
            ViewingState new_state;
            new_state.ply_paths.push_back(path);
            transitionTo(new_state);

            // Emit events
            events::state::SceneLoaded{
                .scene = nullptr, // Not used anymore
                .path = path,
                .type = events::state::SceneLoaded::Type::PLY,
                .num_gaussians = scene_.getTotalGaussianCount()}
                .emit();

            events::state::PLYAdded{
                .name = name,
                .total_gaussians = scene_.getTotalGaussianCount()}
                .emit();

            emitSceneChanged();

        } catch (const std::exception& e) {
            events::notify::Error{
                .message = std::format("Failed to load PLY: {}", e.what()),
                .details = path.string()}
                .emit();
        }
    }

    void SceneManager::addPLY(const std::filesystem::path& path, const std::string& name_hint) {
        try {
            // If not in viewing state, switch to it
            if (!isViewing()) {
                loadPLY(path);
                return;
            }

            std::println("SceneManager: Adding PLY to scene: {}", path.string());

            // Load the PLY
            auto loader = gs::loader::Loader::create();
            gs::loader::LoadOptions options{
                .resize_factor = -1,
                .images_folder = "images",
                .validate_only = false};

            auto load_result = loader->load(path, options);
            if (!load_result) {
                throw std::runtime_error(load_result.error());
            }

            auto* splat_data = std::get_if<std::shared_ptr<gs::SplatData>>(&load_result->data);
            if (!splat_data || !*splat_data) {
                throw std::runtime_error("Expected PLY file");
            }

            // Generate unique name
            std::string base_name = name_hint.empty() ? path.stem().string() : name_hint;
            std::string name = base_name;
            int counter = 1;

            while (scene_.getNode(name) != nullptr) {
                name = std::format("{}_{}", base_name, counter++);
            }

            scene_.addNode(name, std::make_unique<SplatData>(std::move(**splat_data)));

            // Update state
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                if (auto* viewing = std::get_if<ViewingState>(&state_)) {
                    viewing->ply_paths.push_back(path);
                }
            }

            events::state::PLYAdded{
                .name = name,
                .total_gaussians = scene_.getTotalGaussianCount()}
                .emit();

            emitSceneChanged();

        } catch (const std::exception& e) {
            events::notify::Error{
                .message = std::format("Failed to add PLY: {}", e.what()),
                .details = path.string()}
                .emit();
        }
    }

    void SceneManager::removePLY(const std::string& name) {
        scene_.removeNode(name);

        // If no nodes left, transition to empty
        if (scene_.getNodeCount() == 0) {
            transitionTo(EmptyState{});
        }

        events::state::PLYRemoved{.name = name}.emit();
        emitSceneChanged();
    }

    void SceneManager::setPLYVisibility(const std::string& name, bool visible) {
        scene_.setNodeVisibility(name, visible);
        emitSceneChanged();
    }

    void SceneManager::loadDataset(const std::filesystem::path& path,
                                   const param::TrainingParameters& params) {
        try {
            std::println("SceneManager: Loading dataset: {}", path.string());

            // Stop any existing training
            if (trainer_manager_) {
                trainer_manager_->clearTrainer();
            }

            // Clear scene
            clear();

            // Setup training
            auto dataset_params = params;
            dataset_params.dataset.data_path = path;
            cached_params_ = dataset_params;

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

            // Transition to training state - only store the dataset path
            TrainingState new_state;
            new_state.dataset_path = path;
            transitionTo(new_state);

            // Emit events
            size_t num_images = setup_result->dataset->size().value();
            size_t num_gaussians = trainer_manager_->getTrainer()
                                       ->get_strategy()
                                       .get_model()
                                       .size();

            events::state::SceneLoaded{
                .scene = nullptr,
                .path = path,
                .type = events::state::SceneLoaded::Type::Dataset,
                .num_gaussians = num_gaussians}
                .emit();

            events::state::DatasetLoadCompleted{
                .path = path,
                .success = true,
                .error = std::nullopt,
                .num_images = num_images,
                .num_points = num_gaussians}
                .emit();

            emitSceneChanged();

        } catch (const std::exception& e) {
            events::notify::Error{
                .message = std::format("Failed to load dataset: {}", e.what()),
                .details = path.string()}
                .emit();
        }
    }

    void SceneManager::clear() {
        // Stop training if active
        if (trainer_manager_ && isTraining()) {
            events::cmd::StopTraining{}.emit();
            trainer_manager_->clearTrainer();
        }

        scene_.clear();
        transitionTo(EmptyState{});

        events::state::SceneCleared{}.emit();
        emitSceneChanged();
    }

    const SplatData* SceneManager::getModelForRendering() const {
        std::lock_guard<std::mutex> lock(state_mutex_);

        if (std::holds_alternative<ViewingState>(state_)) {
            return scene_.getCombinedModel();
        } else if (std::holds_alternative<TrainingState>(state_)) {
            if (trainer_manager_ && trainer_manager_->getTrainer()) {
                // For training, we need to be careful about thread safety
                // Return the model directly from trainer
                return &trainer_manager_->getTrainer()->get_strategy().get_model();
            }
        }

        return nullptr;
    }

    SceneManager::SceneInfo SceneManager::getSceneInfo() const {
        std::lock_guard<std::mutex> lock(state_mutex_);

        SceneInfo info;

        if (std::holds_alternative<EmptyState>(state_)) {
            info.source_type = "Empty";
        } else if (auto* viewing = std::get_if<ViewingState>(&state_)) {
            info.has_model = scene_.hasNodes();
            info.num_gaussians = scene_.getTotalGaussianCount();
            info.num_nodes = scene_.getNodeCount();
            info.source_type = "PLY";
            if (!viewing->ply_paths.empty()) {
                info.source_path = viewing->ply_paths.back();
            }
        } else if (auto* training = std::get_if<TrainingState>(&state_)) {
            // Query trainer manager for current state
            info.has_model = trainer_manager_ && trainer_manager_->getTrainer();
            if (info.has_model) {
                info.num_gaussians = trainer_manager_->getTrainer()
                                         ->get_strategy()
                                         .get_model()
                                         .size();
            }
            info.num_nodes = 1;
            info.source_type = "Dataset";
            info.source_path = training->dataset_path;
        }

        return info;
    }

    void SceneManager::transitionTo(State new_state) {
        std::lock_guard<std::mutex> lock(state_mutex_);

        // Get old mode for event
        std::string old_mode = "Unknown";
        if (std::holds_alternative<EmptyState>(state_))
            old_mode = "Empty";
        else if (std::holds_alternative<ViewingState>(state_))
            old_mode = "Viewing";
        else if (std::holds_alternative<TrainingState>(state_))
            old_mode = "Training";

        state_ = std::move(new_state);

        // Get new mode for event
        std::string new_mode = "Unknown";
        if (std::holds_alternative<EmptyState>(state_))
            new_mode = "Empty";
        else if (std::holds_alternative<ViewingState>(state_))
            new_mode = "Viewing";
        else if (std::holds_alternative<TrainingState>(state_))
            new_mode = "Training";

        if (old_mode != new_mode) {
            events::ui::RenderModeChanged{
                .old_mode = old_mode,
                .new_mode = new_mode}
                .emit();
        }
    }

    void SceneManager::emitSceneChanged() {
        events::state::SceneChanged{}.emit();
    }

} // namespace gs