#include "core/trainer.hpp"
#include "core/rasterizer.hpp"
#include "kernels/fused_ssim.cuh"
#include <iostream>
#include <torch/torch.h>

namespace gs {

    Trainer::Trainer(std::shared_ptr<CameraDataset> dataset,
                     std::unique_ptr<IStrategy> strategy,
                     const param::TrainingParameters& params)
        : dataset_(std::move(dataset)),
          strategy_(std::move(strategy)),
          params_(params),
          dataset_size_(dataset_->size().value()) {

        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        // Initialize strategy
        strategy_->initialize(params.optimization);

        // Initialize bilateral grid if enabled
        initialize_bilateral_grid();

        // Initialize background
        background_ = torch::tensor({0.f, 0.f, 0.f}, torch::kFloat32).to(torch::kCUDA);

        // Initialize progress bar
        progress_ = std::make_unique<TrainingProgress>(
            params.optimization.iterations,
            /*bar_width=*/100);
    }

    void Trainer::initialize_bilateral_grid() {
        if (!params_.optimization.use_bilateral_grid) {
            return;
        }

        bilateral_grid_ = std::make_unique<gs::BilateralGrid>(
            dataset_size_,
            params_.optimization.bilateral_grid_X,
            params_.optimization.bilateral_grid_Y,
            params_.optimization.bilateral_grid_W);

        bilateral_grid_optimizer_ = std::make_unique<torch::optim::Adam>(
            std::vector<torch::Tensor>{bilateral_grid_->parameters()},
            torch::optim::AdamOptions(params_.optimization.bilateral_grid_lr)
                .eps(1e-15));
    }

    auto Trainer::make_dataloader(int workers) const {
        return create_dataloader_from_dataset(dataset_, workers);
    }

    torch::Tensor Trainer::ensure_4d(const torch::Tensor& image) const {
        return image.dim() == 3 ? image.unsqueeze(0) : image;
    }

    torch::Tensor Trainer::compute_losses(
        const torch::Tensor& rendered,
        const torch::Tensor& ground_truth) {

        // Ensure both tensors are 4D for consistent processing
        auto rendered_4d = ensure_4d(rendered);
        auto gt_4d = ensure_4d(ground_truth);

        // Verify dimensions match
        if (rendered_4d.sizes() != gt_4d.sizes()) {
            std::ostringstream oss;
            oss << "Image size mismatch – rendered " << rendered_4d.sizes()
                << " vs. ground truth " << gt_4d.sizes();
            throw std::runtime_error(oss.str());
        }

        // Compute L1 loss
        auto l1_loss = torch::l1_loss(rendered_4d, gt_4d);

        // Compute SSIM loss
        auto ssim_val = fused_ssim(rendered_4d, gt_4d, "same", /*train=*/true);
        auto ssim_loss = 1.0f - ssim_val;

        // Combine losses
        return (1.0f - params_.optimization.lambda_dssim) * l1_loss +
               params_.optimization.lambda_dssim * ssim_loss;
    }

    torch::Tensor Trainer::compute_regularization_losses() {
        torch::Tensor reg_loss = torch::zeros({1}, torch::kCUDA);

        // Opacity regularization
        if (params_.optimization.opacity_reg > 0.0f) {
            auto opacity_l1 = torch::abs(strategy_->get_model().get_opacity()).mean();
            reg_loss += params_.optimization.opacity_reg * opacity_l1;
        }

        // Scale regularization
        if (params_.optimization.scale_reg > 0.0f) {
            auto scale_l1 = torch::abs(strategy_->get_model().get_scaling()).mean();
            reg_loss += params_.optimization.scale_reg * scale_l1;
        }

        // Total variation loss for bilateral grid
        if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
            reg_loss += params_.optimization.tv_loss_weight * bilateral_grid_->tv_loss();
        }

        return reg_loss;
    }

    void Trainer::step_optimizers(int iter) {
        // Step strategy optimizer
        strategy_->step(iter);

        // Step bilateral grid optimizer if enabled
        if (bilateral_grid_optimizer_) {
            bilateral_grid_optimizer_->step();
            bilateral_grid_optimizer_->zero_grad(true);
        }
    }

    void Trainer::save_checkpoint(int iter) {
        if (iter % 7000 == 0) {
            strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/false);
        }
    }

    void Trainer::train() {
        const int max_iterations = params_.optimization.iterations;
        const int epochs_needed = (max_iterations + dataset_size_ - 1) / dataset_size_;

        int iter = 1;
        auto train_dataloader = make_dataloader();

        for (int epoch = 0; epoch < epochs_needed && iter <= max_iterations; ++epoch) {
            for (auto& batch : *train_dataloader) {
                if (iter > max_iterations) {
                    break;
                }

                // Extract batch data
                auto camera_with_image = batch[0].data;
                Camera* cam = camera_with_image.camera;
                torch::Tensor gt_image = std::move(camera_with_image.image);

                // Render the scene
                auto render_output = gs::rasterize(
                    *cam,
                    strategy_->get_model(),
                    background_,
                    /*scaling_modifier=*/1,
                    /*packed=*/false);

                // Apply bilateral grid if enabled
                if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                    render_output.image = bilateral_grid_->apply(
                        render_output.image,
                        cam->uid());
                }

                // Compute losses
                auto reconstruction_loss = compute_losses(render_output.image, gt_image);
                auto regularization_loss = compute_regularization_losses();
                auto total_loss = reconstruction_loss + regularization_loss;

                // Backward pass
                total_loss.backward();

                // Update model
                {
                    torch::NoGradGuard no_grad;

                    // Save checkpoints
                    save_checkpoint(iter);

                    // Strategy-specific post-backward operations
                    strategy_->post_backward(iter, render_output);

                    // Step all optimizers
                    step_optimizers(iter);
                }

                // Update progress
                const bool is_densifying = (iter > params_.optimization.start_densify &&
                                            iter < params_.optimization.stop_densify &&
                                            iter % params_.optimization.growth_interval == 0);

                progress_->update(
                    iter,
                    total_loss.item<float>(),
                    static_cast<int>(strategy_->get_model().size()),
                    is_densifying);

                ++iter;
            }

            // Recreate dataloader for next epoch
            train_dataloader = make_dataloader();
        }

        // Final save
        strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/true);
        progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));
    }

} // namespace gs