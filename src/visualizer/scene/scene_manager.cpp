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
    }

    void SceneManager::changeContentType(const ContentType& type) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        content_type_ = type;
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
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = true}
                .emit();

            emitSceneChanged();

        } catch (const std::exception& e) {
            events::notify::Error{
                .message = std::format("Failed to load PLY: {}", e.what()),
                .details = path.string()}
                .emit();
        }
    }

    void SceneManager::addPLY(const std::filesystem::path& path, const std::string& name_hint,
                              bool is_visible) {
        try {
            // If not in PLY mode, switch to it
            if (content_type_ != ContentType::PLYFiles) {
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

            // Update paths
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                ply_paths_.push_back(path);
            }

            events::state::PLYAdded{
                .name = name,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = is_visible}
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
            std::lock_guard<std::mutex> lock(state_mutex_);
            content_type_ = ContentType::Empty;
            ply_paths_.clear();
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
            events::notify::Error{
                .message = std::format("Failed to load dataset: {}", e.what()),
                .details = path.string()}
                .emit();
        }
    }

    void SceneManager::clear() {
        // Stop training if active
        if (trainer_manager_ && content_type_ == ContentType::Dataset) {
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

        return info;
    }

    void SceneManager::emitSceneChanged() {
        events::state::SceneChanged{}.emit();
    }

} // namespace gs