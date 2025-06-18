#pragma once

#include <memory>
#include <torch/torch.h>
#include <vector>

namespace gs {

    /**
     * @brief SelectiveAdam optimizer with visibility mask support
     *
     * This optimizer extends the standard Adam optimizer by incorporating selective updates
     * based on visibility masks. It's particularly useful for 3D Gaussian Splatting where
     * only visible Gaussians need to be updated.
     *
     * The implementation uses fused CUDA kernels for better performance.
     */
    class SelectiveAdam : public torch::optim::Optimizer {
    public:
        struct Options : public torch::optim::OptimizerCloneableOptions<Options> {
            Options(double lr = 1e-3) : lr_(lr) {}

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

        explicit SelectiveAdam(std::vector<torch::optim::OptimizerParamGroup> param_groups,
                               std::unique_ptr<Options> options)
            : Optimizer(std::move(param_groups),
                        std::unique_ptr<torch::optim::OptimizerOptions>(std::move(options))) {}

        explicit SelectiveAdam(std::vector<torch::Tensor> params,
                               std::unique_ptr<Options> options)
            : Optimizer({torch::optim::OptimizerParamGroup(std::move(params))},
                        std::unique_ptr<torch::optim::OptimizerOptions>(std::move(options))) {}

        // Override the base class step() with proper signature
        torch::Tensor step(LossClosure closure = nullptr) override;

        /**
         * @brief Perform optimization step with visibility mask
         * @param visibility_mask Boolean tensor indicating which elements to update
         */
        void step(const torch::Tensor& visibility_mask);

    private:
        const Options& options() const {
            return static_cast<const Options&>(defaults());
        }
    };

} // namespace gs