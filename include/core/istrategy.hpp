#pragma once

#include "core/splat_data.hpp"

namespace gs {
    namespace param {
        struct OptimizationParameters;
    }
} // namespace gs

struct RenderOutput;

class IStrategy {
public:
    virtual ~IStrategy() = default;

    virtual void initialize(const gs::param::OptimizationParameters& params) = 0;
    virtual void post_backward(int iter, RenderOutput& render_output) = 0;
    virtual void step(int iter) = 0;

    // Get the underlying Gaussian model for rendering
    virtual SplatData& get_model() = 0;
    virtual const SplatData& get_model() const = 0;
};