/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
*
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include <torch/torch.h>

namespace gs::training {
    struct RenderOutput;

    class IStrategy {
    public:
        virtual ~IStrategy() = default;

        virtual void initialize(const gs::param::OptimizationParameters& optimParams) = 0;

        virtual void post_backward(int iter, RenderOutput& render_output) = 0;

        virtual void step(int iter) = 0;

        virtual bool is_refining(int iter) const = 0;

        // Get the underlying Gaussian model for rendering
        virtual gs::SplatData& get_model() = 0;

        virtual const gs::SplatData& get_model() const = 0;

        // Remove Gaussians based on mask
        virtual void remove_gaussians(const torch::Tensor& mask) = 0;

        // Get the optimizer for memory management
        virtual torch::optim::Optimizer* get_optimizer() = 0;
    };
} // namespace gs::training