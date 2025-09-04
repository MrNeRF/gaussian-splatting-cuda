/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include <expected>
#include <memory>
#include <string>
#include <torch/torch.h>

namespace gs::training {

    /**
     * @brief Interface for sparsity optimization methods
     *
     * Provides a clean abstraction for different sparsity-inducing techniques
     * that can be applied during Gaussian Splatting training.
     */
    class ISparsityOptimizer {
    public:
        virtual ~ISparsityOptimizer() = default;

        /**
         * @brief Initialize the optimizer with initial opacities
         * @param opacities Initial opacity values from the model
         * @return Error string if initialization fails
         */
        virtual std::expected<void, std::string> initialize(const torch::Tensor& opacities) = 0;

        /**
         * @brief Compute the sparsity regularization loss
         * @param opacities Current opacity values from the model
         * @return Loss tensor or error string
         */
        virtual std::expected<torch::Tensor, std::string> compute_loss(const torch::Tensor& opacities) const = 0;

        /**
         * @brief Update internal state (called periodically during training)
         * @param opacities Current opacity values from the model
         * @return Error string if update fails
         */
        virtual std::expected<void, std::string> update_state(const torch::Tensor& opacities) = 0;

        /**
         * @brief Get mask indicating which Gaussians to prune
         * @param opacities Current opacity values from the model
         * @return Boolean mask tensor or error string
         */
        virtual std::expected<torch::Tensor, std::string> get_prune_mask(const torch::Tensor& opacities) const = 0;

        /**
         * @brief Check if we should update state at this iteration
         */
        virtual bool should_update(int iter) const = 0;

        /**
         * @brief Check if we should apply loss at this iteration
         */
        virtual bool should_apply_loss(int iter) const = 0;

        /**
         * @brief Check if we should prune at this iteration
         */
        virtual bool should_prune(int iter) const = 0;

        /**
         * @brief Get the number of Gaussians that will be pruned
         */
        virtual int get_num_to_prune(const torch::Tensor& opacities) const = 0;

        /**
         * @brief Check if the optimizer has been initialized
         */
        virtual bool is_initialized() const = 0;
    };

    /**
     * @brief ADMM-based sparsity optimizer
     *
     * Implements Alternating Direction Method of Multipliers (ADMM) for
     * inducing sparsity in Gaussian opacity values during training.
     */
    class ADMMSparsityOptimizer : public ISparsityOptimizer {
    public:
        struct Config {
            int sparsify_steps = 15000;  // Total steps for sparsification
            float init_rho = 0.0005f;    // ADMM penalty parameter
            float prune_ratio = 0.6f;    // Final pruning ratio
            int update_every = 50;       // Update ADMM state every N iterations
            int start_iteration = 30000; // When to start sparsification (after base training)
        };

        explicit ADMMSparsityOptimizer(const Config& config);

        std::expected<void, std::string> initialize(const torch::Tensor& opacities) override;
        std::expected<torch::Tensor, std::string> compute_loss(const torch::Tensor& opacities) const override;
        std::expected<void, std::string> update_state(const torch::Tensor& opacities) override;
        std::expected<torch::Tensor, std::string> get_prune_mask(const torch::Tensor& opacities) const override;

        bool should_update(int iter) const override {
            int relative_iter = iter - config_.start_iteration;
            return iter >= config_.start_iteration &&
                   relative_iter > 0 &&
                   relative_iter < config_.sparsify_steps &&
                   relative_iter % config_.update_every == 0;
        }

        bool should_apply_loss(int iter) const override {
            return iter >= config_.start_iteration &&
                   iter < (config_.start_iteration + config_.sparsify_steps);
        }

        bool should_prune(int iter) const override {
            return iter == (config_.start_iteration + config_.sparsify_steps);
        }

        int get_num_to_prune(const torch::Tensor& opacities) const override;

        bool is_initialized() const override { return initialized_; }

    private:
        /**
         * @brief Apply soft thresholding to enforce sparsity
         * @param z Input tensor
         * @return Thresholded tensor
         */
        torch::Tensor prune_z(const torch::Tensor& z) const;

        Config config_;
        torch::Tensor u_; // Dual variable (Lagrange multiplier)
        torch::Tensor z_; // Auxiliary variable for sparsity
        bool initialized_ = false;
    };

    /**
     * @brief Factory for creating sparsity optimizers
     */
    class SparsityOptimizerFactory {
    public:
        static std::unique_ptr<ISparsityOptimizer> create(
            const std::string& method,
            const ADMMSparsityOptimizer::Config& config);
    };

} // namespace gs::training