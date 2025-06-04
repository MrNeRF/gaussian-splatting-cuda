#include "core/trainer.hpp"
#include "core/image_io.hpp"
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

        // Only initialize evaluation components if needed
        if (params.optimization.enable_eval) {
            psnr_metric_ = std::make_unique<metrics::PSNR>(1.0f);
            ssim_metric_ = std::make_unique<metrics::SSIM>(11, 3);

            std::filesystem::path lpips_path = params.dataset.output_path.parent_path() / "weights" / "lpips_vgg.pt";
            if (!std::filesystem::exists(lpips_path)) {
                lpips_path = "weights/lpips_vgg.pt";
            }
            lpips_metric_ = std::make_unique<metrics::LPIPS>(lpips_path.string());
            metrics_reporter_ = std::make_unique<metrics::MetricsReporter>(params.dataset.output_path);
        }
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

        for (int epoch = 0; epoch < epochs_needed && iter <= params_.optimization.iterations; ++epoch) {
            for (auto& batch : *train_dataloader) {
                if (iter > params_.optimization.iterations) {
                    break;
                }

                auto camera_with_image = batch[0].data;
                Camera* cam = camera_with_image.camera;
                torch::Tensor gt_image = std::move(camera_with_image.image);

                auto r_output = gs::rasterize(*cam, strategy_->get_model(), background_, 1, false);

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
                auto loss = (1.f - params_.optimization.lambda_dssim) * l1l + params_.optimization.lambda_dssim * (1.f - ssim_loss);

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

                loss.backward();

                {
                    torch::NoGradGuard no_grad;

                    // Simple evaluation - just print without progress bar interaction
                    if (params_.optimization.enable_eval) {
                        for (size_t eval_step : params_.optimization.eval_steps) {
                            if (iter == static_cast<int>(eval_step)) {
                                // Don't touch the progress bar, just print on new lines
                                std::cout << std::endl; // Move to new line
                                std::cout << "[Evaluation at step " << iter << "]" << std::endl;
                                auto metrics = evaluate(iter);
                                std::cout << metrics.to_string() << std::endl;
                                // Progress bar will continue on next update
                            }
                        }
                    }

                    // Save model at specified steps
                    for (size_t save_step : params_.optimization.save_steps) {
                        if (iter == static_cast<int>(save_step)) {
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
                ++iter;
            }

            train_dataloader = make_train_dataloader();
        }

        // Final evaluation (only if enabled)
        if (params_.optimization.enable_eval) {
            progress_->complete(); // Complete progress bar before final output
            std::cout << "\n[Final Evaluation]" << std::endl;
            auto final_metrics = evaluate(iter);
            std::cout << final_metrics.to_string() << std::endl;
            metrics_reporter_->save_report();
        } else {
            progress_->complete(); // Still need to complete progress bar
        }

        strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/true);
        progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));
    }

    metrics::EvalMetrics Trainer::evaluate(int iteration) {
        metrics::EvalMetrics result;
        result.num_gaussians = static_cast<int>(strategy_->get_model().size());
        result.iteration = iteration;

        auto val_dataloader = make_val_dataloader();

        std::vector<float> psnr_values, ssim_values, lpips_values;
        auto start_time = std::chrono::steady_clock::now();

        for (auto& batch : *val_dataloader) {
            auto camera_with_image = batch[0].data;
            Camera* cam = camera_with_image.camera;
            torch::Tensor gt_image = std::move(camera_with_image.image);

            auto r_output = gs::rasterize(*cam, strategy_->get_model(), background_, 1, false);

            // Ensure correct dimensions
            if (r_output.image.dim() == 3)
                r_output.image = r_output.image.unsqueeze(0);
            if (gt_image.dim() == 3)
                gt_image = gt_image.unsqueeze(0);

            // Clamp rendered image to [0, 1]
            r_output.image = torch::clamp(r_output.image, 0.0, 1.0);

            // Compute metrics
            float psnr = psnr_metric_->compute(r_output.image, gt_image);
            float ssim = ssim_metric_->compute(r_output.image, gt_image);
            float lpips = lpips_metric_->compute(r_output.image, gt_image);

            psnr_values.push_back(psnr);
            ssim_values.push_back(ssim);
            lpips_values.push_back(lpips);
        }

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<float>(end_time - start_time).count();

        // Compute averages
        result.psnr = std::accumulate(psnr_values.begin(), psnr_values.end(), 0.0f) / psnr_values.size();
        result.ssim = std::accumulate(ssim_values.begin(), ssim_values.end(), 0.0f) / ssim_values.size();
        result.lpips = std::accumulate(lpips_values.begin(), lpips_values.end(), 0.0f) / lpips_values.size();
        result.elapsed_time = elapsed / val_dataset_size_; // Time per image

        // Add metrics to reporter
        metrics_reporter_->add_metrics(result);

        return result;
    }

} // namespace gs