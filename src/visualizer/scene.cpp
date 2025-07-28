#include "visualizer/scene.hpp"
#include "core/model_providers.hpp"
#include "visualizer/event_response_handler.hpp"
#include <print>

namespace gs {

    Scene::Scene() : mode_(Mode::Empty) {
    }

    void Scene::setModelProvider(std::shared_ptr<IModelProvider> provider) {
        model_provider_ = provider;

        if (!pipeline_) {
            pipeline_ = std::make_unique<RenderingPipeline>();
        }

        // Update mode based on provider type
        if (!provider) {
            mode_ = Mode::Empty;
        } else if (dynamic_cast<TrainerModelProvider*>(provider.get())) {
            mode_ = Mode::Training;
        } else {
            mode_ = Mode::Viewing;
        }

        // Publish mode change event
        if (event_bus_) {
            publishModeChange(mode_, mode_);
        }
    }

    void Scene::clearModel() {
        auto old_mode = mode_;
        model_provider_.reset();
        mode_ = Mode::Empty;

        if (event_bus_ && old_mode != mode_) {
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

    void Scene::setEventBus(std::shared_ptr<EventBus> event_bus) {
        event_bus_ = event_bus;

        if (event_bus_) {
            // Subscribe to queries
            event_bus_->subscribe<QueryModelInfoRequest>(
                [this](const QueryModelInfoRequest& request) {
                    handleModelInfoQuery(request);
                });

            event_bus_->subscribe<QuerySceneModeRequest>(
                [this](const QuerySceneModeRequest& request) {
                    handleSceneModeQuery(request);
                });
        }
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

        switch (mode_) {
        case Mode::Empty:
            response.mode = QuerySceneModeResponse::Mode::Empty;
            break;
        case Mode::Viewing:
            response.mode = QuerySceneModeResponse::Mode::Viewing;
            break;
        case Mode::Training:
            response.mode = QuerySceneModeResponse::Mode::Training;
            break;
        }

        // We don't track the path in Scene, that's SceneManager's responsibility
        response.current_path = std::nullopt;

        event_bus_->publish(response);
    }

    void Scene::publishModeChange(Mode old_mode, Mode new_mode) {
        if (!event_bus_ || old_mode == new_mode)
            return;

        SceneModeChangedEvent event;

        // Convert internal mode to event mode
        auto convertMode = [](Mode m) -> QuerySceneModeResponse::Mode {
            switch (m) {
            case Mode::Empty: return QuerySceneModeResponse::Mode::Empty;
            case Mode::Viewing: return QuerySceneModeResponse::Mode::Viewing;
            case Mode::Training: return QuerySceneModeResponse::Mode::Training;
            default: return QuerySceneModeResponse::Mode::Empty;
            }
        };

        event.old_mode = convertMode(old_mode);
        event.new_mode = convertMode(new_mode);
        event.loaded_path = std::nullopt; // Scene doesn't track paths

        event_bus_->publish(event);
    }

} // namespace gs