#include "scene.hpp"
#include "core/model_providers.hpp"
#include <format>

namespace gs {

    Scene::Scene() {
        // Initialize rendering pipeline
        pipeline_ = std::make_unique<RenderingPipeline>();

        // Subscribe to query events
        events::query::GetModelInfo::when([this](const auto&) {
            handleModelInfoQuery();
        });
    }

    void Scene::setModelProvider(std::shared_ptr<IModelProvider> provider) {
        // Legacy single-model method - clear all and add single model
        clearAllModels();

        if (!pipeline_) {
            pipeline_ = std::make_unique<RenderingPipeline>();
        }

        // Update mode based on provider type
        Mode old_mode = mode_;
        if (!provider) {
            mode_ = Mode::Empty;
        } else if (dynamic_cast<TrainerModelProvider*>(provider.get())) {
            mode_ = Mode::Training;
            // Training mode: add single model
            addModel("Training Model", "", provider);
        } else {
            mode_ = Mode::Viewing;
            // Viewing mode: add as standalone model
            addModel("Model", "", provider);
        }

        // Publish mode change event if it changed
        if (old_mode != mode_) {
            publishModeChange(old_mode, mode_);
        }
    }

    void Scene::clearModel() {
        clearAllModels();
    }

    bool Scene::hasModel() const {
        return !models_.empty() && std::any_of(models_.begin(), models_.end(),
                                               [](const auto& pair) { return pair.second.provider && pair.second.provider->hasModel(); });
    }

    std::string Scene::addModel(const std::string& name,
                                const std::filesystem::path& path,
                                std::shared_ptr<IModelProvider> provider) {
        if (!provider || !provider->hasModel()) {
            return "";
        }

        // In training mode, only allow one model
        if (mode_ == Mode::Training && !models_.empty()) {
            return "";
        }

        std::string id = generateModelId();
        ModelEntry entry{
            .id = id,
            .name = name.empty() ? std::format("Model_{}", id) : name,
            .path = path,
            .provider = provider,
            .visible = true,
            .selected = false};

        models_[id] = entry;

        // Clear merged model cache when adding new model
        merged_model_.reset();
        merged_model_dirty_ = true;

        // Update mode if needed
        if (mode_ == Mode::Empty) {
            Mode old_mode = mode_;
            mode_ = Mode::Viewing;
            publishModeChange(old_mode, mode_);
        }

        publishModelAdded(id, entry);
        return id;
    }

    bool Scene::removeModel(const std::string& id) {
        auto it = models_.find(id);
        if (it == models_.end()) {
            return false;
        }

        models_.erase(it);

        // Clear merged model cache when removing model
        merged_model_.reset();
        merged_model_dirty_ = true;

        publishModelRemoved(id);

        // Update mode if needed
        if (models_.empty()) {
            Mode old_mode = mode_;
            mode_ = Mode::Empty;
            publishModeChange(old_mode, mode_);
        }

        return true;
    }

    void Scene::clearAllModels() {
        auto old_mode = mode_;
        models_.clear();
        merged_model_.reset();
        merged_model_dirty_ = true;
        mode_ = Mode::Empty;

        if (old_mode != mode_) {
            publishModeChange(old_mode, mode_);
        }
    }

    std::vector<const Scene::ModelEntry*> Scene::getModels() const {
        std::vector<const ModelEntry*> result;
        result.reserve(models_.size());
        for (const auto& [id, entry] : models_) {
            result.push_back(&entry);
        }
        return result;
    }

    Scene::ModelEntry* Scene::getModel(const std::string& id) {
        auto it = models_.find(id);
        return it != models_.end() ? &it->second : nullptr;
    }

    const Scene::ModelEntry* Scene::getModel(const std::string& id) const {
        auto it = models_.find(id);
        return it != models_.end() ? &it->second : nullptr;
    }

    void Scene::selectModel(const std::string& id, bool exclusive) {
        if (exclusive) {
            deselectAllModels();
        }

        auto it = models_.find(id);
        if (it != models_.end()) {
            it->second.selected = true;
            publishSelectionChanged();
        }
    }

    void Scene::deselectModel(const std::string& id) {
        auto it = models_.find(id);
        if (it != models_.end()) {
            it->second.selected = false;
            publishSelectionChanged();
        }
    }

    void Scene::deselectAllModels() {
        for (auto& [id, entry] : models_) {
            entry.selected = false;
        }
        publishSelectionChanged();
    }

    std::vector<std::string> Scene::getSelectedModelIds() const {
        std::vector<std::string> result;
        for (const auto& [id, entry] : models_) {
            if (entry.selected) {
                result.push_back(id);
            }
        }
        return result;
    }

    void Scene::setModelVisible(const std::string& id, bool visible) {
        auto it = models_.find(id);
        if (it != models_.end()) {
            it->second.visible = visible;
            // Clear merged model cache when visibility changes
            merged_model_.reset();
            merged_model_dirty_ = true;
        }
    }

    bool Scene::isModelVisible(const std::string& id) const {
        auto it = models_.find(id);
        return it != models_.end() ? it->second.visible : false;
    }

    void Scene::setStandaloneModel(std::unique_ptr<SplatData> model) {
        auto provider = std::make_shared<StandaloneModelProvider>(std::move(model));
        setModelProvider(provider);
    }

    void Scene::linkToTrainer(Trainer* trainer) {
        if (trainer) {
            auto provider = std::make_shared<TrainerModelProvider>(trainer);
            setModelProvider(provider);
        } else {
            clearModel();
        }
    }

    void Scene::unlinkFromTrainer() {
        if (mode_ == Mode::Training) {
            clearModel();
        }
    }

    const SplatData* Scene::getModel() const {
        // For training mode or single model, return directly
        if (mode_ == Mode::Training || models_.size() <= 1) {
            for (const auto& [id, entry] : models_) {
                if (entry.visible && entry.provider) {
                    return entry.provider->getModel();
                }
            }
            return nullptr;
        }

        // For multiple models, return merged model
        if (!merged_model_ || merged_model_dirty_) {
            const_cast<Scene*>(this)->updateMergedModel();
        }

        return merged_model_.get();
    }

    void Scene::updateMergedModel() {
        merged_model_.reset();
        merged_model_dirty_ = false;

        // Collect all visible models
        std::vector<const SplatData*> visible_models;
        for (const auto& [id, entry] : models_) {
            if (entry.visible && entry.provider && entry.provider->hasModel()) {
                visible_models.push_back(entry.provider->getModel());
            }
        }

        if (visible_models.empty()) {
            return;
        }

        // TODO: LAZY implementation. That needs to be improved.
        // Always merge, even for single model (to avoid const issues)
        // Calculate total size and max SH degree
        size_t total_gaussians = 0;
        int max_sh_degree = 0;
        for (const auto* model : visible_models) {
            total_gaussians += model->size();
            max_sh_degree = std::max(max_sh_degree, model->get_active_sh_degree());
        }

        // Get device from first model
        auto device = visible_models[0]->get_means().device();
        auto options = torch::TensorOptions().device(device);

        // Allocate merged tensors
        torch::Tensor merged_means = torch::empty({static_cast<long>(total_gaussians), 3}, options);

        // Get SH dimensions from get_shs()
        auto first_shs = visible_models[0]->get_shs();
        int sh_dim = first_shs.size(1); // Total SH dimensions

        torch::Tensor merged_sh0 = torch::empty({static_cast<long>(total_gaussians), 1, 3}, options);
        torch::Tensor merged_shN = torch::empty({static_cast<long>(total_gaussians), sh_dim - 1, 3}, options);
        torch::Tensor merged_scaling = torch::empty({static_cast<long>(total_gaussians), 3}, options);
        torch::Tensor merged_rotation = torch::empty({static_cast<long>(total_gaussians), 4}, options);
        torch::Tensor merged_opacity = torch::empty({static_cast<long>(total_gaussians), 1}, options);

        // Copy data from each model
        size_t offset = 0;
        for (const auto* model : visible_models) {
            size_t model_size = model->size();

            // Copy means
            merged_means.slice(0, offset, offset + model_size) = model->get_means();

            // Get SH coefficients combined
            auto model_shs = model->get_shs(); // [N, total_sh_coeffs, 3]

            // Split into DC (first) and rest
            merged_sh0.slice(0, offset, offset + model_size) = model_shs.slice(1, 0, 1);

            // Handle different SH degrees
            int model_sh_rest = model_shs.size(1) - 1;
            if (model_sh_rest == merged_shN.size(1)) {
                merged_shN.slice(0, offset, offset + model_size) = model_shs.slice(1, 1);
            } else {
                // Pad with zeros if model has lower SH degree
                if (model_sh_rest > 0) {
                    merged_shN.slice(0, offset, offset + model_size).slice(1, 0, model_sh_rest) =
                        model_shs.slice(1, 1);
                }
                if (model_sh_rest < merged_shN.size(1)) {
                    merged_shN.slice(0, offset, offset + model_size).slice(1, model_sh_rest) = 0;
                }
            }

            // Copy scaling (get_scaling returns exp of _scaling)
            merged_scaling.slice(0, offset, offset + model_size) = torch::log(model->get_scaling());

            // Copy rotation (get_rotation returns normalized quaternions)
            merged_rotation.slice(0, offset, offset + model_size) = model->get_rotation();

            // Copy opacity (get_opacity returns sigmoid of _opacity, we need raw)
            // Since we need raw values and only have sigmoid output, we need to inverse sigmoid
            auto opacity_sigmoid = model->get_opacity();
            // Clamp to avoid log(0) or log(1)
            opacity_sigmoid = torch::clamp(opacity_sigmoid, 1e-7, 1.0 - 1e-7);
            merged_opacity.slice(0, offset, offset + model_size) =
                torch::log(opacity_sigmoid / (1.0 - opacity_sigmoid)).unsqueeze(-1);

            offset += model_size;
        }

        // Create merged model with the tensors
        float scene_scale = visible_models[0]->get_scene_scale();
        merged_model_ = std::make_unique<SplatData>(
            max_sh_degree,
            merged_means,
            merged_sh0,
            merged_shN,
            merged_scaling,
            merged_rotation,
            merged_opacity,
            scene_scale);
    }

    RenderingPipeline::RenderResult Scene::render(const RenderingPipeline::RenderRequest& request) {
        if (!hasModel() || !pipeline_) {
            return RenderingPipeline::RenderResult{.valid = false};
        }

        const SplatData* model = getModel();
        if (!model) {
            return RenderingPipeline::RenderResult{.valid = false};
        }

        return pipeline_->render(*model, request);
    }

    size_t Scene::getTotalGaussianCount() const {
        size_t total = 0;
        for (const auto& [id, entry] : models_) {
            if (entry.visible && entry.provider && entry.provider->hasModel()) {
                total += entry.provider->getModel()->size();
            }
        }
        return total;
    }

    std::string Scene::generateModelId() {
        return std::format("model_{}", next_model_id_++);
    }

    void Scene::handleModelInfoQuery() {
        events::query::ModelInfo response;

        if (hasModel()) {
            const SplatData* model = getModel();
            response.has_model = true;
            response.num_gaussians = getTotalGaussianCount();
            response.sh_degree = model ? model->get_active_sh_degree() : 0;
            response.scene_scale = model ? model->get_scene_scale() : 0.0f;
            response.source = models_.size() > 1 ? std::format("Multiple ({} models)", models_.size()) : (models_.empty() ? "None" : models_.begin()->second.provider->getModelSource());
        } else {
            response.has_model = false;
            response.num_gaussians = 0;
            response.sh_degree = 0;
            response.scene_scale = 0.0f;
            response.source = "None";
        }

        response.emit();
    }

    void Scene::publishModeChange(Mode old_mode, Mode new_mode) {
        if (old_mode == new_mode)
            return;

        // Convert internal mode to string for the event
        auto modeToString = [](Mode m) -> std::string {
            switch (m) {
            case Mode::Empty: return "Empty";
            case Mode::Viewing: return "Viewing";
            case Mode::Training: return "Training";
            default: return "Unknown";
            }
        };

        events::ui::RenderModeChanged{
            .old_mode = modeToString(old_mode),
            .new_mode = modeToString(new_mode)}
            .emit();
    }

    void Scene::publishModelAdded(const std::string& id, const ModelEntry& entry) {
        events::notify::Log{
            .level = events::notify::Log::Level::Info,
            .message = std::format("Added model: {} ({})", entry.name, id),
            .source = "Scene"}
            .emit();
    }

    void Scene::publishModelRemoved(const std::string& id) {
        events::notify::Log{
            .level = events::notify::Log::Level::Info,
            .message = std::format("Removed model: {}", id),
            .source = "Scene"}
            .emit();
    }

    void Scene::publishSelectionChanged() {
        events::state::ModelSelectionChanged{
            .selected_ids = getSelectedModelIds()}
            .emit();
    }

} // namespace gs