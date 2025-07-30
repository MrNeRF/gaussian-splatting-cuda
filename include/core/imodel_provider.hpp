#pragma once

#include "core/splat_data.hpp"

namespace gs {

    /**
     * @brief Abstract interface for providing access to SplatData models
     *
     * This interface allows Scene to access models without knowing
     * whether they come from a Trainer, a loaded PLY file, or any other source.
     */
    class IModelProvider {
    public:
        virtual ~IModelProvider() = default;

        /**
         * @brief Get read-only access to the model
         * @return Pointer to the model, or nullptr if no model available
         */
        virtual const SplatData* getModel() const = 0;

        /**
         * @brief Get mutable access to the model
         * @return Pointer to the model, or nullptr if no model available
         */
        virtual SplatData* getMutableModel() = 0;

        /**
         * @brief Check if a model is currently available
         * @return true if model exists, false otherwise
         */
        virtual bool hasModel() const = 0;

        /**
         * @brief Get the type/source of the model for debugging
         * @return String describing the model source
         */
        virtual std::string getModelSource() const = 0;
    };

} // namespace gs