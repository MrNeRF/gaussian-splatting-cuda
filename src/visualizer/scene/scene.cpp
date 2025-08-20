#include "scene.hpp"
#include "core/model_providers.hpp"
#include <algorithm>
#include <print>

namespace gs {

    Scene::Scene() {
        setupEventHandlers();
    }

    void Scene::setupEventHandlers() {

        events::cmd::RemovePLY::when([this](const auto& cmd) {
            removePLY(cmd.name);
        });

        events::cmd::SetPLYVisibility::when([this](const auto& cmd) {
            setPLYVisibility(cmd.name, cmd.visible);
        });
    }

    void Scene::addPLY(const std::string& name, std::unique_ptr<SplatData> model) {
        // If we're not in viewing mode, switch to it
        if (mode_ != Mode::Viewing) {
            auto old_mode = mode_;
            mode_ = Mode::Viewing;
            // Clear any training model
            model_provider_.reset();
            publishModeChange(old_mode, mode_);
        }

        // Calculate gaussian count before moving the model
        size_t gaussian_count = static_cast<size_t>(model->size());

        // Check if name already exists
        auto it = std::find_if(scene_graph_.begin(), scene_graph_.end(),
                               [&name](const SceneNode& node) { return node.name == name; });

        if (it != scene_graph_.end()) {
            // Replace existing
            it->model = std::move(model);
            it->gaussian_count = gaussian_count;
        } else {
            // Add new node
            SceneNode node{
                .name = name,
                .model = std::move(model),
                .transform = glm::mat4(1.0f),
                .visible = true,
                .gaussian_count = gaussian_count};
            scene_graph_.push_back(std::move(node));
        }

        // Invalidate cache
        cache_valid_ = false;

        // Emit event with the correct total gaussian count
        events::state::PLYAdded{
            .name = name,
            .total_gaussians = gaussian_count}
            .emit();

        std::println("Added PLY '{}' with {} gaussians to scene graph", name, gaussian_count);

        // Force a scene change event to trigger immediate rendering
        events::state::SceneChanged{}.emit();
    }

    void Scene::removePLY(const std::string& name) {
        auto it = std::find_if(scene_graph_.begin(), scene_graph_.end(),
                               [&name](const SceneNode& node) { return node.name == name; });

        if (it != scene_graph_.end()) {
            scene_graph_.erase(it);

            // Invalidate cache
            cache_valid_ = false;

            // Emit event
            events::state::PLYRemoved{.name = name}.emit();

            // If scene graph is empty, switch to empty mode
            if (scene_graph_.empty()) {
                auto old_mode = mode_;
                mode_ = Mode::Empty;
                publishModeChange(old_mode, mode_);
            }
        }
    }

    void Scene::clearPLYs() {
        if (!scene_graph_.empty()) {
            scene_graph_.clear();
            cache_valid_ = false;
            cached_combined_model_.reset();

            auto old_mode = mode_;
            mode_ = Mode::Empty;
            publishModeChange(old_mode, mode_);
        }
    }

    void Scene::setPLYVisibility(const std::string& name, bool visible) {
        auto it = std::find_if(scene_graph_.begin(), scene_graph_.end(),
                               [&name](const SceneNode& node) { return node.name == name; });

        if (it != scene_graph_.end() && it->visible != visible) {
            it->visible = visible;
            // Invalidate cache when visibility changes
            cache_valid_ = false;

            // Emit scene changed event to trigger re-render immediately
            events::state::SceneChanged{}.emit();
        }
    }

    std::vector<const Scene::SceneNode*> Scene::getSceneNodes() const {
        std::vector<const SceneNode*> nodes;
        nodes.reserve(scene_graph_.size());
        for (const auto& node : scene_graph_) {
            nodes.push_back(&node);
        }
        return nodes;
    }

    size_t Scene::getTotalGaussianCount() const {
        size_t total = 0;
        for (const auto& node : scene_graph_) {
            if (node.visible) {
                total += node.gaussian_count;
            }
        }
        return total;
    }

    void Scene::setModelProvider(std::shared_ptr<IModelProvider> provider) {
        model_provider_ = provider;
        cache_valid_ = false;
        cached_combined_model_.reset();

        // Update mode based on provider type
        Mode old_mode = mode_;
        if (!provider) {
            mode_ = Mode::Empty;
        } else if (dynamic_cast<TrainerModelProvider*>(provider.get())) {
            mode_ = Mode::Training;
            // Clear PLY scene graph when switching to training
            scene_graph_.clear();
        } else {
            mode_ = Mode::Viewing;
            // For backwards compatibility, we can't copy SplatData since copy constructor is deleted
            // So we'll just clear the scene graph and let the user reload if needed
            scene_graph_.clear();
            std::println("Note: Cannot convert single model to scene graph (SplatData copy not allowed)");
        }

        // Publish mode change event if it changed
        if (old_mode != mode_) {
            publishModeChange(old_mode, mode_);
        }
    }

    void Scene::clearModel() {
        auto old_mode = mode_;
        model_provider_.reset();
        scene_graph_.clear();
        cache_valid_ = false;
        cached_combined_model_.reset();
        mode_ = Mode::Empty;

        if (old_mode != mode_) {
            publishModeChange(old_mode, mode_);
        }
    }

    bool Scene::hasModel() const {
        if (mode_ == Mode::Viewing) {
            // Return true if we have any models, regardless of visibility
            // The combined model will be empty if all are invisible
            return !scene_graph_.empty();
        } else if (mode_ == Mode::Training) {
            return model_provider_ && model_provider_->hasModel();
        }
        return false;
    }

    const SplatData* Scene::getModel() const {
        if (mode_ == Mode::Viewing) {
            // Return cached combined model
            rebuildCombinedModelIfNeeded();
            return cached_combined_model_.get();
        } else if (model_provider_) {
            return model_provider_->getModel();
        }
        return nullptr;
    }

    SplatData* Scene::getMutableModel() {
        if (mode_ == Mode::Viewing) {
            // For mutable access, we need to invalidate cache
            cache_valid_ = false;
            rebuildCombinedModelIfNeeded();
            return cached_combined_model_.get();
        } else if (model_provider_) {
            return model_provider_->getMutableModel();
        }
        return nullptr;
    }

    void Scene::rebuildCombinedModelIfNeeded() const {
        if (cache_valid_) {
            return;
        }

        // Collect visible models
        std::vector<const SplatData*> visible_models;
        size_t total_gaussians = 0;

        for (const auto& node : scene_graph_) {
            if (node.visible && node.model) {
                visible_models.push_back(node.model.get());
                total_gaussians += node.gaussian_count;
            }
        }

        if (visible_models.empty()) {
            // No visible models - clear cache
            cached_combined_model_.reset();
            cache_valid_ = true;
            return;
        }

        std::println("Rebuilding combined model with {} visible models, {} total gaussians",
                     visible_models.size(), total_gaussians);

        // Get device and dtype from first model
        auto device = visible_models[0]->means().device();
        auto dtype = visible_models[0]->means().dtype();

        // Pre-allocate tensors for all visible gaussians
        auto combined_means = torch::empty({static_cast<int64_t>(total_gaussians), 3}, dtype).to(device);
        auto combined_sh0 = torch::empty({static_cast<int64_t>(total_gaussians),
                                          visible_models[0]->sh0().size(1),
                                          visible_models[0]->sh0().size(2)},
                                         dtype)
                                .to(device);
        auto combined_shN = torch::empty({static_cast<int64_t>(total_gaussians),
                                          visible_models[0]->shN().size(1),
                                          visible_models[0]->shN().size(2)},
                                         dtype)
                                .to(device);
        auto combined_opacity = torch::empty({static_cast<int64_t>(total_gaussians), 1}, dtype).to(device);
        auto combined_scaling = torch::empty({static_cast<int64_t>(total_gaussians), 3}, dtype).to(device);
        auto combined_rotation = torch::empty({static_cast<int64_t>(total_gaussians), 4}, dtype).to(device);

        // Concatenate all visible models
        size_t current_idx = 0;
        int max_sh_degree = 0;
        float avg_scene_scale = 0.0f;

        for (const auto* model : visible_models) {
            int64_t model_size = model->size();

            // Copy means
            combined_means.index({torch::indexing::Slice(current_idx, current_idx + model_size)}) = model->means();

            // Copy SH coefficients
            combined_sh0.index({torch::indexing::Slice(current_idx, current_idx + model_size)}) = model->sh0();
            combined_shN.index({torch::indexing::Slice(current_idx, current_idx + model_size)}) = model->shN();

            // Copy other attributes
            combined_opacity.index({torch::indexing::Slice(current_idx, current_idx + model_size)}) = model->opacity_raw();
            combined_scaling.index({torch::indexing::Slice(current_idx, current_idx + model_size)}) = model->scaling_raw();
            combined_rotation.index({torch::indexing::Slice(current_idx, current_idx + model_size)}) = model->rotation_raw();

            // Track max SH degree and average scene scale
            max_sh_degree = std::max(max_sh_degree, model->get_active_sh_degree());
            avg_scene_scale += model->get_scene_scale();

            current_idx += model_size;
        }

        avg_scene_scale /= visible_models.size();

        // Create combined SplatData
        // Set requires_grad to match the original models
        combined_means = combined_means.set_requires_grad(true);
        combined_sh0 = combined_sh0.set_requires_grad(true);
        combined_shN = combined_shN.set_requires_grad(true);
        combined_opacity = combined_opacity.set_requires_grad(true);
        combined_scaling = combined_scaling.set_requires_grad(true);
        combined_rotation = combined_rotation.set_requires_grad(true);

        cached_combined_model_ = std::make_unique<SplatData>(
            max_sh_degree,
            combined_means,
            combined_sh0,
            combined_shN,
            combined_scaling,
            combined_rotation,
            combined_opacity,
            avg_scene_scale);

        cache_valid_ = true;
        std::println("Combined model created with {} gaussians", cached_combined_model_->size());
    }

    void Scene::setStandaloneModel(std::unique_ptr<SplatData> model) {
        // Convert to PLY node for new multi-model support
        clearModel();
        addPLY("Model", std::move(model));
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

    Scene::ModelInfo Scene::getModelInfo() const {
        ModelInfo info;

        if (hasModel()) {
            if (mode_ == Mode::Viewing) {
                info.has_model = true;
                info.num_gaussians = getTotalGaussianCount();

                // Get combined model to access its properties
                const SplatData* combined = getModel();
                if (combined) {
                    info.sh_degree = combined->get_active_sh_degree();
                    info.scene_scale = combined->get_scene_scale();
                } else {
                    info.sh_degree = 0;
                    info.scene_scale = 0.0f;
                }

                // Count visible models
                int visible_count = 0;
                for (const auto& node : scene_graph_) {
                    if (node.visible)
                        visible_count++;
                }

                info.source = std::format("PLY Scene ({} models, {} visible)",
                                          scene_graph_.size(), visible_count);
            } else if (model_provider_) {
                const SplatData* model = model_provider_->getModel();
                info.has_model = true;
                info.num_gaussians = static_cast<size_t>(model->size());
                info.sh_degree = model->get_active_sh_degree();
                info.scene_scale = model->get_scene_scale();
                info.source = model_provider_->getModelSource();
            }
        } else {
            info.has_model = false;
            info.num_gaussians = 0;
            info.sh_degree = 0;
            info.scene_scale = 0.0f;
            info.source = "None";
        }

        return info;
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

} // namespace gs