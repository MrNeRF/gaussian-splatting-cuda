#pragma once

namespace gs {
    namespace param {
        struct OptimizationParameters;
    }
} // namespace gs

struct RenderOutput;

class IStrategy {
    void virtual initialize(const gs::param::OptimizationParameters& params) = 0;
    void virtual post_backward(int iter, RenderOutput& render_output) = 0;
    void virtual step(int iter) = 0;
};