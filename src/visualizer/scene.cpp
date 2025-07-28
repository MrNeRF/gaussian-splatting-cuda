#include "visualizer/scene.hpp"
#include "core/model_providers.hpp"
#include "visualizer/event_response_handler.hpp"

namespace gs {

    Scene::Scene()
        : pipeline_(std::make_unique<RenderingPipeline>()) {
    }

    void Scene::setEventBus(std::shared_ptr<EventBus> event_bus) {
        event_bus_ = event_bus;

        if (event_bus_) {
            // Subscribe to model info queries
            event_bus_->subscribe<QueryModelInfoRequest>(
                [this](const QueryModelInfoRequest& request) {
                    handleModelInfoQuery(request);
                });

            // Subscribe to scene mode queries
            event_bus_->subscribe<QuerySceneModeRequest>(
                [this](const QuerySceneModeRequest& request) {
                    handleSceneModeQuery(request);
                });
        }
    }

    void Scene::setModelProvider(std::shared_ptr<IModelProvider> provider) {
        auto old_mode = mode_;
        model_provider_ = provider;

        // Determine mode based on provider type
        if (!provider || !provider->hasModel()) {
            mode_ = Mode::Empty;
        } else if (provider->getModelSource() == "Training") {
            mode_ = Mode::Training;
        } else {
            mode_ = Mode::Viewing;
        }

        // Publish mode change event
        if (event_bus_ && old_mode != mode_) {
            publishModeChange(old_mode, mode_);
        }
    }

    void Scene::setStandaloneModel(std::unique_ptr<SplatData> model) {
        auto provider = std::make_shared<StandaloneModelProvider>(std::move(model));
        setModelProvider(provider);
    }

    void Scene::linkToTrainer(Trainer* trainer) {
        auto provider = std::make_shared<TrainerModelProvider>(trainer);
        setModelProvider(provider);
    }

    void Scene::unlinkFromTrainer() {
        clearModel();
    }

    void Scene::clearModel() {
        auto old_mode = mode_;
        model_provider_.reset();
        mode_ = Mode::Empty;

        // Publish mode change event
        if (event_bus_ && old_mode != mode_) {
            publishModeChange(old_mode, mode_);
        }
    }

    bool Scene::hasModel() const {
        return model_provider_ && model_provider_->hasModel();
    }

    RenderingPipeline::RenderResult Scene::render(const RenderingPipeline::RenderRequest& request) {
        RenderingPipeline::RenderResult result;
        result.valid = false;

        const SplatData* model = getModel();
        if (model && pipeline_) {
            result = pipeline_->render(*model, request);
        }

        return result;
    }

    void Scene::handleModelInfoQuery(const QueryModelInfoRequest& request) {
        if (!event_bus_)
            return;

        QueryModelInfoResponse response;

        if (hasModel()) {
            const SplatData* model = getModel();
            response.has_model = true;
            response.num_gaussians = model->size();
            response.sh_degree = model->get_active_sh_degree();
            response.scene_scale = model->get_scene_scale();
            response.model_source = model_provider_->getModelSource();
        } else {
            response.has_model = false;
            response.num_gaussians = 0;
            response.sh_degree = 0;
            response.scene_scale = 0.0f;
            response.model_source = "None";
        }

        event_bus_->publish(response);
    }

    void Scene::handleSceneModeQuery(const QuerySceneModeRequest& request) {
        if (!event_bus_)
            return;

        QuerySceneModeResponse response;

        // Map internal mode to response mode
        switch (mode_) {
        case Mode::Empty: response.mode = QuerySceneModeResponse::Mode::Empty; break;
        case Mode::Viewing: response.mode = QuerySceneModeResponse::Mode::Viewing; break;
        case Mode::Training: response.mode = QuerySceneModeResponse::Mode::Training; break;
        }

        // Scene doesn't store path info - that should come from viewer

        event_bus_->publish(response);
    }

    void Scene::publishModeChange(Mode old_mode, Mode new_mode) {
        if (!event_bus_)
            return;

        SceneModeChangedEvent event;

        // Map modes
        auto mapMode = [](Mode m) -> QuerySceneModeResponse::Mode {
            switch (m) {
            case Mode::Empty: return QuerySceneModeResponse::Mode::Empty;
            case Mode::Viewing: return QuerySceneModeResponse::Mode::Viewing;
            case Mode::Training: return QuerySceneModeResponse::Mode::Training;
            default: return QuerySceneModeResponse::Mode::Empty;
            }
        };

        event.old_mode = mapMode(old_mode);
        event.new_mode = mapMode(new_mode);

        event_bus_->publish(event);
    }

} // namespace gs