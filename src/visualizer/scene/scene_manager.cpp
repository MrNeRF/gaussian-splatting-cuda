#include "scene/scene_manager.hpp"
#include "core/logger.hpp"
#include "loader/loader.hpp"
#include "training/training_manager.hpp"
#include "training_setup.hpp"
#include <stdexcept>

namespace gs {

    SceneManager::SceneManager() {
        setupEventHandlers();
        LOG_DEBUG("SceneManager initialized");
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
    }

    void SceneManager::changeContentType(const ContentType& type) {
        std::lock_guard<std::mutex> lock(state_mutex_);

        const char* type_str = (type == ContentType::Empty) ? "Empty" : (type == ContentType::PLYFiles) ? "PLYFiles"
                                                                                                        : "Dataset";
        LOG_DEBUG("Changing content type to: {}", type_str);

        content_type_ = type;
    }

    void SceneManager::loadPLY(const std::filesystem::path& path) {
        LOG_TIMER("SceneManager::loadPLY");

        try {
            LOG_INFO("Loading PLY file: {}", path.string());

            // Clear existing scene
            clear();

            // Load the PLY
            LOG_DEBUG("Creating loader for PLY file");
            auto loader = gs::loader::Loader::create();
            gs::loader::LoadOptions options{
                .resize_factor = -1,
                .images_folder = "images",
                .validate_only = false};

            LOG_TRACE("Loading PLY with loader");
            auto load_result = loader->load(path, options);
            if (!load_result) {
                LOG_ERROR("Failed to load PLY: {}", load_result.error());
                throw std::runtime_error(load_result.error());
            }

            auto* splat_data = std::get_if<std::shared_ptr<gs::SplatData>>(&load_result->data);
            if (!splat_data || !*splat_data) {
                LOG_ERROR("Expected PLY file but got different data type from: {}", path.string());
                throw std::runtime_error("Expected PLY file but got different data type");
            }

            // Add to scene
            std::string name = path.stem().string();
            size_t gaussian_count = (*splat_data)->size();
            LOG_DEBUG("Adding PLY '{}' to scene with {} gaussians", name, gaussian_count);

            scene_.addNode(name, std::make_unique<SplatData>(std::move(**splat_data)));

            // Update content state
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                content_type_ = ContentType::PLYFiles;
                ply_paths_.clear();
                ply_paths_.push_back(path);
            }

            // Emit events
            events::state::SceneLoaded{
                .scene = nullptr,
                .path = path,
                .type = events::state::SceneLoaded::Type::PLY,
                .num_gaussians = scene_.getTotalGaussianCount()}
                .emit();

            events::state::PLYAdded{
                .name = name,
                .node_gaussians = gaussian_count,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = true}
                .emit();

            emitSceneChanged();

            LOG_INFO("Successfully loaded PLY '{}' with {} gaussians", name, gaussian_count);

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load PLY: {} (path: {})", e.what(), path.string());
            throw;
        }
    }

    void SceneManager::addPLY(const std::filesystem::path& path, const std::string& name_hint,
                              bool is_visible) {
        LOG_TIMER_TRACE("SceneManager::addPLY");

        try {
            // If not in PLY mode, switch to it
            if (content_type_ != ContentType::PLYFiles) {
                LOG_DEBUG("Not in PLY mode, switching to PLY mode and loading");
                loadPLY(path);
                return;
            }

            LOG_INFO("Adding PLY to scene: {}", path.string());

            // Load the PLY
            auto loader = gs::loader::Loader::create();
            gs::loader::LoadOptions options{
                .resize_factor = -1,
                .images_folder = "images",
                .validate_only = false};

            LOG_TRACE("Loading PLY data");
            auto load_result = loader->load(path, options);
            if (!load_result) {
                LOG_ERROR("Failed to load PLY: {}", load_result.error());
                throw std::runtime_error(load_result.error());
            }

            auto* splat_data = std::get_if<std::shared_ptr<gs::SplatData>>(&load_result->data);
            if (!splat_data || !*splat_data) {
                LOG_ERROR("Expected PLY file from: {}", path.string());
                throw std::runtime_error("Expected PLY file");
            }

            // Generate unique name
            std::string base_name = name_hint.empty() ? path.stem().string() : name_hint;
            std::string name = base_name;
            int counter = 1;

            while (scene_.getNode(name) != nullptr) {
                name = std::format("{}_{}", base_name, counter++);
                LOG_TRACE("Name '{}' already exists, trying '{}'", base_name, name);
            }

            size_t gaussian_count = (*splat_data)->size();
            LOG_DEBUG("Adding node '{}' with {} gaussians", name, gaussian_count);

            scene_.addNode(name, std::make_unique<SplatData>(std::move(**splat_data)));

            // Update paths
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                ply_paths_.push_back(path);
            }

            events::state::PLYAdded{
                .name = name,
                .node_gaussians = gaussian_count,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = is_visible}
                .emit();

            emitSceneChanged();

            LOG_INFO("Added PLY '{}' to scene ({} gaussians, visible: {})",
                     name, gaussian_count, is_visible);

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to add PLY: {} (path: {})", e.what(), path.string());
            throw;
        }
    }

    void SceneManager::removePLY(const std::string& name) {
        LOG_DEBUG("Removing PLY '{}' from scene", name);

        scene_.removeNode(name);

        // If no nodes left, transition to empty
        if (scene_.getNodeCount() == 0) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            content_type_ = ContentType::Empty;
            ply_paths_.clear();
            LOG_DEBUG("No nodes remaining, transitioning to empty state");
        }

        events::state::PLYRemoved{.name = name}.emit();
        emitSceneChanged();

        LOG_INFO("Removed PLY '{}' from scene", name);
    }

    void SceneManager::setPLYVisibility(const std::string& name, bool visible) {
        LOG_TRACE("Setting PLY '{}' visibility to: {}", name, visible);
        scene_.setNodeVisibility(name, visible);
        emitSceneChanged();
    }

    void SceneManager::loadDataset(const std::filesystem::path& path,
                                   const param::TrainingParameters& params) {
        LOG_TIMER("SceneManager::loadDataset");

        try {
            LOG_INFO("Loading dataset: {}", path.string());

            // Stop any existing training
            if (trainer_manager_) {
                LOG_DEBUG("Clearing existing trainer");
                trainer_manager_->clearTrainer();
            }

            // Clear scene
            clear();

            // Setup training
            auto dataset_params = params;
            dataset_params.dataset.data_path = path;
            cached_params_ = dataset_params;

            LOG_DEBUG("Setting up training with parameters");
            LOG_TRACE("Dataset path: {}", path.string());
            LOG_TRACE("Iterations: {}", dataset_params.optimization.iterations);

            auto setup_result = gs::setupTraining(dataset_params);
            if (!setup_result) {
                LOG_ERROR("Failed to setup training: {}", setup_result.error());
                throw std::runtime_error(setup_result.error());
            }

            // Pass trainer to manager
            if (trainer_manager_) {
                LOG_DEBUG("Setting trainer in manager");
                trainer_manager_->setTrainer(std::move(setup_result->trainer));
            } else {
                LOG_ERROR("No trainer manager available");
                throw std::runtime_error("No trainer manager available");
            }

            // Update content state
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                content_type_ = ContentType::Dataset;
                dataset_path_ = path;
            }

            // Emit events
            const size_t num_gaussians = trainer_manager_->getTrainer()
                                             ->get_strategy()
                                             .get_model()
                                             .size();

            LOG_INFO("Dataset loaded successfully - {} images, {} initial gaussians",
                     setup_result->dataset->size().value(), num_gaussians);

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
                .num_images = setup_result->dataset->size().value(),
                .num_points = num_gaussians}
                .emit();

            emitSceneChanged();

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load dataset: {} (path: {})", e.what(), path.string());
            throw;
        }
    }

    void SceneManager::clear() {
        LOG_DEBUG("Clearing scene");

        // Stop training if active
        if (trainer_manager_ && content_type_ == ContentType::Dataset) {
            LOG_DEBUG("Stopping training before clearing");
            events::cmd::StopTraining{}.emit();
            trainer_manager_->clearTrainer();
        }

        scene_.clear();

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            content_type_ = ContentType::Empty;
            ply_paths_.clear();
            dataset_path_.clear();
        }

        events::state::SceneCleared{}.emit();
        emitSceneChanged();

        LOG_INFO("Scene cleared");
    }

    const SplatData* SceneManager::getModelForRendering() const {
        std::lock_guard<std::mutex> lock(state_mutex_);

        if (content_type_ == ContentType::PLYFiles) {
            return scene_.getCombinedModel();
        } else if (content_type_ == ContentType::Dataset) {
            if (trainer_manager_ && trainer_manager_->getTrainer()) {
                return &trainer_manager_->getTrainer()->get_strategy().get_model();
            }
        }

        return nullptr;
    }

    SceneManager::SceneInfo SceneManager::getSceneInfo() const {
        std::lock_guard<std::mutex> lock(state_mutex_);

        SceneInfo info;

        switch (content_type_) {
        case ContentType::Empty:
            info.source_type = "Empty";
            break;

        case ContentType::PLYFiles:
            info.has_model = scene_.hasNodes();
            info.num_gaussians = scene_.getTotalGaussianCount();
            info.num_nodes = scene_.getNodeCount();
            info.source_type = "PLY";
            if (!ply_paths_.empty()) {
                info.source_path = ply_paths_.back();
            }
            break;

        case ContentType::Dataset:
            info.has_model = trainer_manager_ && trainer_manager_->getTrainer();
            if (info.has_model) {
                info.num_gaussians = trainer_manager_->getTrainer()
                                         ->get_strategy()
                                         .get_model()
                                         .size();
            }
            info.num_nodes = 1;
            info.source_type = "Dataset";
            info.source_path = dataset_path_;
            break;
        }

        LOG_TRACE("Scene info - type: {}, gaussians: {}, nodes: {}",
                  info.source_type, info.num_gaussians, info.num_nodes);

        return info;
    }

    void SceneManager::emitSceneChanged() {
        events::state::SceneChanged{}.emit();
    }

} // namespace gs