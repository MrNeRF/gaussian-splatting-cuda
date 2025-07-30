#include "scene.hpp"
#include "core/model_providers.hpp"

namespace gs {

    Scene::Scene() {
        // Subscribe to query events
        events::query::GetModelInfo::when([this](const auto&) {
            handleModelInfoQuery();
        });
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
        } else {
            mode_ = Mode::Viewing;
        }

        // Publish mode change event if it changed
        if (old_mode != mode_) {
            publishModeChange(old_mode, mode_);
        }
    }

    void Scene::clearModel() {
        auto old_mode = mode_;
        model_provider_.reset();
        mode_ = Mode::Empty;

        if (old_mode != mode_) {
            publishModeChange(old_mode, mode_);
        }
    }

    bool Scene::hasModel() const {
        return model_provider_ && model_provider_->hasModel();
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

    void Scene::handleModelInfoQuery() {
        events::query::ModelInfo response;

        if (hasModel()) {
            const SplatData* model = getModel();
            response.has_model = true;
            response.num_gaussians = model->size();
            response.sh_degree = model->get_active_sh_degree();
            response.scene_scale = model->get_scene_scale();
            response.source = model_provider_->getModelSource();
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