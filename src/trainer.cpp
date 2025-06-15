#include "core/trainer.hpp"
#include "core/rasterizer.hpp"
#include "kernels/fused_ssim.cuh"
#include <chrono>
#include <iostream>
#include <numeric>
#include <torch/torch.h>

namespace gs {


    Trainer::Trainer(std::shared_ptr<CameraDataset> dataset,
                     std::unique_ptr<IStrategy> strategy,
                     const param::TrainingParameters& params)
        : strategy_(std::move(strategy)),
          params_(params) {

        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        // Handle dataset split based on evaluation flag
        if (params.optimization.enable_eval) {
            // Create train/val split
            train_dataset_ = std::make_shared<CameraDataset>(
                dataset->get_cameras(), params.dataset, CameraDataset::Split::TRAIN);
            val_dataset_ = std::make_shared<CameraDataset>(
                dataset->get_cameras(), params.dataset, CameraDataset::Split::VAL);

            std::cout << "Created train/val split: "
                      << train_dataset_->size().value() << " train, "
                      << val_dataset_->size().value() << " val images" << std::endl;
        } else {
            // Use all images for training
            train_dataset_ = dataset;
            val_dataset_ = nullptr;

            std::cout << "Using all " << train_dataset_->size().value()
                      << " images for training (no evaluation)" << std::endl;
        }

        train_dataset_size_ = train_dataset_->size().value();
        val_dataset_size_ = val_dataset_ ? val_dataset_->size().value() : 0;

        strategy_->initialize(params.optimization);

        background_ = torch::tensor({0.f, 0.f, 0.f}, torch::TensorOptions().dtype(torch::kFloat32));
        background_ = background_.to(torch::kCUDA);

        progress_ = std::make_unique<TrainingProgress>(
            params.optimization.iterations,
            /*bar_width=*/100);

        // Initialize the evaluator - it handles all metrics internally
        evaluator_ = std::make_unique<metrics::MetricsEvaluator>(params);

        // Print render mode configuration
        std::cout << "Render mode: " << params.optimization.render_mode << std::endl;
    }

    auto Trainer::make_train_dataloader(int workers) const {
        return create_dataloader_from_dataset(train_dataset_, workers);
    }

    auto Trainer::make_val_dataloader(int workers) const {
        return create_dataloader_from_dataset(val_dataset_, workers);
    }

    void Trainer::train() {
        int iter = 1;
        int epochs_needed = (params_.optimization.iterations + train_dataset_size_ - 1) / train_dataset_size_;

        auto train_dataloader = make_train_dataloader();

        // Convert string render mode to enum once
        RenderMode render_mode = stringToRenderMode(params_.optimization.render_mode);
        const bool has_rgb = renderModeHasRGB(render_mode);
        const bool has_depth = renderModeHasDepth(render_mode);

        for (int epoch = 0; epoch < epochs_needed && iter <= params_.optimization.iterations; ++epoch) {
            for (auto& batch : *train_dataloader) {
                auto camera_with_image = batch[0].data;
                Camera* cam = camera_with_image.camera;
                torch::Tensor gt_image = std::move(camera_with_image.image);

                // Use the render mode from parameters
                auto r_output = gs::rasterize(
                    *cam,
                    strategy_->get_model(),
                    background_,
                    1.0f,
                    false,
                    false,
                    render_mode // Use the configured render mode
                );

                torch::Tensor loss;

                // Only process RGB if render mode includes it
                if (has_rgb) {
                    if (r_output.image.dim() == 3)
                        r_output.image = r_output.image.unsqueeze(0);

                    if (gt_image.dim() == 3)
                        gt_image = gt_image.unsqueeze(0);

                    if (r_output.image.sizes() != gt_image.sizes()) {
                        std::cerr << "ERROR: size mismatch – rendered " << r_output.image.sizes()
                                  << " vs. ground truth " << gt_image.sizes() << '\n';
                        throw std::runtime_error("Image size mismatch");
                    }

                    // Base loss computation
                    auto l1l = torch::l1_loss(r_output.image.squeeze(0), gt_image.squeeze(0));
                    auto ssim_loss = fused_ssim(r_output.image, gt_image, "same", /*train=*/true);
                    loss = (1.f - params_.optimization.lambda_dssim) * l1l +
                           params_.optimization.lambda_dssim * (1.f - ssim_loss);

                    // Add opacity regularization
                    if (params_.optimization.opacity_reg > 0.0f) {
                        auto opacity_l1 = torch::abs(strategy_->get_model().get_opacity()).mean();
                        loss += params_.optimization.opacity_reg * opacity_l1;
                    }

                    // Add scale regularization
                    if (params_.optimization.scale_reg > 0.0f) {
                        auto scale_l1 = torch::abs(strategy_->get_model().get_scaling()).mean();
                        loss += params_.optimization.scale_reg * scale_l1;
                    }
                } else {
                    // For depth-only modes, create a dummy loss or implement depth supervision
                    std::cerr << "Warning: Training with depth-only mode without RGB loss. ";
                    std::cerr << "Consider implementing depth supervision or switching to RGB_D/RGB_ED mode.\n";

                    // Create a small dummy loss to allow gradient computation
                    loss = torch::zeros({1}, torch::kFloat32).to(torch::kCUDA);
                    loss.requires_grad_(true);
                }

                loss.backward();

                {
                    torch::NoGradGuard no_grad;

                    // Clean evaluation - let the evaluator handle everything
                    if (evaluator_->is_enabled() && evaluator_->should_evaluate(iter)) {
                        evaluator_->print_evaluation_header(iter);
                        auto metrics = evaluator_->evaluate(iter,
                                                            strategy_->get_model(),
                                                            val_dataset_,
                                                            background_);
                        std::cout << metrics.to_string() << std::endl;
                    }

                    // Save model at specified steps
                    for (size_t save_step : params_.optimization.save_steps) {
                        if (iter == static_cast<int>(save_step) && iter != params_.optimization.iterations) {
                            strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/false);
                        }
                    }

                    strategy_->post_backward(iter, r_output);
                    strategy_->step(iter);
                }

                const bool is_densifying = (iter < params_.optimization.stop_densify &&
                                            iter > params_.optimization.start_densify &&
                                            iter % params_.optimization.growth_interval == 0);

                progress_->update(iter, loss.item<float>(), static_cast<int>(strategy_->get_model().size()), is_densifying);

                if (iter == params_.optimization.iterations) {
                    break;
                }

                ++iter;
            }

            train_dataloader = make_train_dataloader();
        }

        // Final evaluation and save
        progress_->complete();
        if (evaluator_->is_enabled()) {
            evaluator_->print_final_evaluation_header();
            auto final_metrics = evaluator_->evaluate(params_.optimization.iterations,
                                                      strategy_->get_model(),
                                                      val_dataset_,
                                                      background_);
            std::cout << final_metrics.to_string() << std::endl;
            evaluator_->save_report();
        } else {
        }

        strategy_->get_model().save_ply(params_.dataset.output_path, params_.optimization.iterations, /*join=*/true);
        progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));
    }

} // namespace gs