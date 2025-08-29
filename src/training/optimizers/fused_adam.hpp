/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>
#include <torch/torch.h>
#include <vector>

namespace gs::training {
    /**
     * @brief FusedAdam optimizer
     *
     * The implementation uses fused CUDA kernels for better performance.
     */
    class FusedAdam : public torch::optim::Optimizer {
    public:
        struct Options : public torch::optim::OptimizerCloneableOptions<Options> {
            Options(double lr = 1e-3) : lr_(lr) {
            }

            Options& lr(double lr) {
                lr_ = lr;
                return *this;
            }

            Options& betas(const std::tuple<double, double>& betas) {
                betas_ = betas;
                return *this;
            }

            Options& eps(double eps) {
                eps_ = eps;
                return *this;
            }

            Options& weight_decay(double weight_decay) {
                weight_decay_ = weight_decay;
                return *this;
            }

            double lr() const { return lr_; }
            const std::tuple<double, double>& betas() const { return betas_; }
            double eps() const { return eps_; }
            double weight_decay() const { return weight_decay_; }

        private:
            double lr_ = 1e-3;
            std::tuple<double, double> betas_ = std::make_tuple(0.9, 0.999);
            double eps_ = 1e-8;
            double weight_decay_ = 0;
        };

        struct AdamParamState : public torch::optim::OptimizerParamState {
            torch::Tensor exp_avg;
            torch::Tensor exp_avg_sq;
            torch::Tensor max_exp_avg_sq; // For amsgrad variant (not used currently)
            int64_t step_count = 0;

            void serialize(torch::serialize::OutputArchive& archive) const override {
                archive.write("exp_avg", exp_avg);
                archive.write("exp_avg_sq", exp_avg_sq);
                archive.write("step", step_count);
                if (max_exp_avg_sq.defined()) {
                    archive.write("max_exp_avg_sq", max_exp_avg_sq);
                }
            }
        };

        explicit FusedAdam(std::vector<torch::optim::OptimizerParamGroup> param_groups,
                           std::unique_ptr<Options> options)
            : Optimizer(std::move(param_groups),
                        std::unique_ptr<torch::optim::OptimizerOptions>(std::move(options))) {
        }

        explicit FusedAdam(std::vector<torch::Tensor> params, std::unique_ptr<Options> options)
            : Optimizer({torch::optim::OptimizerParamGroup(std::move(params))},
                        std::unique_ptr<torch::optim::OptimizerOptions>(std::move(options))) {
        }

        // Override the base class step() with proper signature
        torch::Tensor step(LossClosure closure) override;

        /**
         * @brief Perform optimization step
         */
        void step(int iteration);

        void zero_grad(bool set_to_none, int iteration);

    private:
        const Options& options() const {
            return static_cast<const Options&>(defaults());
        }
    };
} // namespace gs::training
