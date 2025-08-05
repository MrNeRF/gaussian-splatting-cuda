#pragma once

#include "core/imodel_provider.hpp"
#include "core/trainer.hpp"
#include <memory>

namespace gs {

    /**
     * @brief Model provider for standalone PLY models
     */
    class StandaloneModelProvider : public IModelProvider {
    public:
        explicit StandaloneModelProvider(std::unique_ptr<SplatData> model)
            : model_(std::move(model)) {}

        const SplatData* getModel() const override {
            return model_.get();
        }

        SplatData* getMutableModel() override {
            return model_.get();
        }

        bool hasModel() const override {
            return model_ != nullptr;
        }

        std::string getModelSource() const override {
            return "Standalone PLY";
        }

        // Allow updating the model
        void setModel(std::unique_ptr<SplatData> model) {
            model_ = std::move(model);
        }

    private:
        std::unique_ptr<SplatData> model_;
    };

    /**
     * @brief Model provider that wraps a Trainer
     */
    class TrainerModelProvider : public IModelProvider {
    public:
        explicit TrainerModelProvider(Trainer* trainer)
            : trainer_(trainer) {}

        const SplatData* getModel() const override {
            if (!trainer_)
                return nullptr;
            // Assuming get_model() returns SplatData& not gs::SplatData&
            return &trainer_->get_strategy().get_model();
        }

        SplatData* getMutableModel() override {
            if (!trainer_)
                return nullptr;
            // This is safe because we know the trainer owns the model
            return const_cast<SplatData*>(&trainer_->get_strategy().get_model());
        }

        bool hasModel() const override {
            return trainer_ != nullptr;
        }

        std::string getModelSource() const override {
            return "Training";
        }

    private:
        Trainer* trainer_;
    };

} // namespace gs