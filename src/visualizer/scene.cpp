#include "visualizer/scene.hpp"
#include "core/model_providers.hpp"

namespace gs {

    Scene::Scene()
        : pipeline_(std::make_unique<RenderingPipeline>()) {
    }

    void Scene::setModelProvider(std::shared_ptr<IModelProvider> provider) {
        model_provider_ = provider;

        // Determine mode based on provider type
        if (!provider || !provider->hasModel()) {
            mode_ = Mode::Empty;
        } else if (provider->getModelSource() == "Training") {
            mode_ = Mode::Training;
        } else {
            mode_ = Mode::Viewing;
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
        model_provider_.reset();
        mode_ = Mode::Empty;
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

} // namespace gs