#include "scene.hpp"
#include "core/model_providers.hpp"
#include <algorithm>
#include <print>

namespace gs {

    Scene::Scene() {
        setupEventHandlers();
    }

    void Scene::setupEventHandlers() {
        // Subscribe to query events
        events::query::GetModelInfo::when([this](const auto&) {
            handleModelInfoQuery();
        });

        // Handle PLY commands
        events::cmd::AddPLY::when([this]([[maybe_unused]] const auto& cmd) {
            // This will be handled by SceneManager which has access to loading
        });

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

        // Initialize pipeline if needed
        if (!pipeline_) {
            pipeline_ = std::make_unique<RenderingPipeline>();
        }

        // Emit event with the correct total gaussian count
        events::state::PLYAdded{
            .name = name,
            .total_gaussians = gaussian_count // Use the individual model's count, not total
        }
            .emit();

        std::println("Added PLY '{}' with {} gaussians to scene graph", name, gaussian_count);
    }

    void Scene::removePLY(const std::string& name) {
        auto it = std::find_if(scene_graph_.begin(), scene_graph_.end(),
                               [&name](const SceneNode& node) { return node.name == name; });

        if (it != scene_graph_.end()) {
            scene_graph_.erase(it);

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

            auto old_mode = mode_;
            mode_ = Mode::Empty;
            publishModeChange(old_mode, mode_);
        }
    }

    void Scene::setPLYVisibility(const std::string& name, bool visible) {
        auto it = std::find_if(scene_graph_.begin(), scene_graph_.end(),
                               [&name](const SceneNode& node) { return node.name == name; });

        if (it != scene_graph_.end()) {
            it->visible = visible;
            std::println("Set PLY '{}' visibility to {}", name, visible);
            // Emit scene changed event to trigger re-render
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

        if (!pipeline_) {
            pipeline_ = std::make_unique<RenderingPipeline>();
        }

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
        mode_ = Mode::Empty;

        if (old_mode != mode_) {
            publishModeChange(old_mode, mode_);
        }
    }

    bool Scene::hasModel() const {
        if (mode_ == Mode::Viewing) {
            // Check if we have any visible models
            return std::any_of(scene_graph_.begin(), scene_graph_.end(),
                               [](const SceneNode& node) { return node.visible && node.model; });
        } else if (mode_ == Mode::Training) {
            return model_provider_ && model_provider_->hasModel();
        }
        return false;
    }

    const SplatData* Scene::getModel() const {
        if (mode_ == Mode::Viewing && !scene_graph_.empty()) {
            // Return first visible model
            for (const auto& node : scene_graph_) {
                if (node.visible && node.model) {
                    return node.model.get();
                }
            }
        } else if (model_provider_) {
            return model_provider_->getModel();
        }
        return nullptr;
    }

    SplatData* Scene::getMutableModel() {
        if (mode_ == Mode::Viewing && !scene_graph_.empty()) {
            // Return first visible model
            for (auto& node : scene_graph_) {
                if (node.visible && node.model) {
                    return node.model.get();
                }
            }
        } else if (model_provider_) {
            return model_provider_->getMutableModel();
        }
        return nullptr;
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

    RenderingPipeline::RenderResult Scene::render(const RenderingPipeline::RenderRequest& request) {
        if (!hasModel() || !pipeline_) {
            return RenderingPipeline::RenderResult(false);
        }

        // TODO: Implement proper multi-model rendering
        // For now, we need to render each visible model separately and composite them
        // This is a limitation of the current RenderingPipeline which expects a single SplatData

        // Count visible models
        int visible_count = 0;
        const SplatData* last_visible_model = nullptr;

        for (const auto& node : scene_graph_) {
            if (node.visible && node.model) {
                visible_count++;
                last_visible_model = node.model.get();
            }
        }

        if (visible_count == 0) {
            // No visible models
            return RenderingPipeline::RenderResult(false);
        }

        if (visible_count == 1) {
            // Single model - render normally
            return pipeline_->render(*last_visible_model, request);
        }

        // Multiple visible models - for now, just render the first one
        // TODO: Implement proper multi-model compositing
        const SplatData* first_visible = getModel();
        if (!first_visible) {
            return RenderingPipeline::RenderResult(false);
        }

        std::println("Warning: Rendering only first visible model out of {} visible models", visible_count);
        return pipeline_->render(*first_visible, request);
    }

    void Scene::handleModelInfoQuery() {
        events::query::ModelInfo response;

        if (hasModel()) {
            if (mode_ == Mode::Viewing) {
                response.has_model = true;
                response.num_gaussians = getTotalGaussianCount();

                // Find first visible model for sh_degree and scene_scale
                const SplatData* first_visible = getModel();
                if (first_visible) {
                    response.sh_degree = first_visible->get_active_sh_degree();
                    response.scene_scale = first_visible->get_scene_scale();
                } else {
                    response.sh_degree = 0;
                    response.scene_scale = 0.0f;
                }

                // Count visible models
                int visible_count = 0;
                for (const auto& node : scene_graph_) {
                    if (node.visible)
                        visible_count++;
                }

                response.source = std::format("PLY Scene ({} models, {} visible)",
                                              scene_graph_.size(), visible_count);
            } else if (model_provider_) {
                const SplatData* model = model_provider_->getModel();
                response.has_model = true;
                response.num_gaussians = static_cast<size_t>(model->size());
                response.sh_degree = model->get_active_sh_degree();
                response.scene_scale = model->get_scene_scale();
                response.source = model_provider_->getModelSource();
            }
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

} // namespace gs