#include "visualizer/scene.hpp"

namespace gs {

    Scene::Scene()
        : pipeline_(std::make_unique<RenderingPipeline>()) {
    }

    void Scene::setModel(std::unique_ptr<SplatData> model) {
        // Clear any training link
        trainer_ = nullptr;

        // Set the new model
        model_ = std::move(model);
        mode_ = model_ ? Mode::Viewing : Mode::Empty;
    }

    void Scene::clearModel() {
        model_.reset();
        trainer_ = nullptr;
        mode_ = Mode::Empty;
    }

    bool Scene::hasModel() const {
        return (mode_ != Mode::Empty);
    }

    void Scene::linkToTrainer(Trainer* trainer) {
        // Clear any viewing model
        model_.reset();

        // Link to trainer
        trainer_ = trainer;
        mode_ = trainer ? Mode::Training : Mode::Empty;
    }

    void Scene::unlinkFromTrainer() {
        trainer_ = nullptr;
        if (!model_) {
            mode_ = Mode::Empty;
        }
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