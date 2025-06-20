#include "core/trainer.hpp"
#include "core/rasterizer.hpp"
#include "kernels/fused_ssim.cuh"
#include "visualizer/detail.hpp"
#include <chrono>
#include <iostream>
#include <numeric>
#include <torch/torch.h>

namespace gs {

    static inline torch::Tensor ensure_4d(const torch::Tensor& image) {
        return image.dim() == 3 ? image.unsqueeze(0) : image;
    }

    void Trainer::initialize_bilateral_grid() {
        if (!params_.optimization.use_bilateral_grid) {
            return;
        }

        bilateral_grid_ = std::make_unique<gs::BilateralGrid>(
            train_dataset_size_,
            params_.optimization.bilateral_grid_X,
            params_.optimization.bilateral_grid_Y,
            params_.optimization.bilateral_grid_W);

        bilateral_grid_optimizer_ = std::make_unique<torch::optim::Adam>(
            std::vector<torch::Tensor>{bilateral_grid_->parameters()},
            torch::optim::AdamOptions(params_.optimization.bilateral_grid_lr)
                .eps(1e-15));
    }

    torch::Tensor Trainer::compute_loss(const RenderOutput& render_output,
                                        const torch::Tensor& gt_image,
                                        const SplatData& splatData,
                                        const param::OptimizationParameters& opt_params) {
        // Ensure images have same dimensions
        torch::Tensor rendered = render_output.image;
        torch::Tensor gt = gt_image;

        // Ensure both tensors are 4D (batch, height, width, channels)
        rendered = rendered.dim() == 3 ? rendered.unsqueeze(0) : rendered;
        gt = gt.dim() == 3 ? gt.unsqueeze(0) : gt;

        TORCH_CHECK(rendered.sizes() == gt.sizes(), "ERROR: size mismatch – rendered ", rendered.sizes(), " vs. ground truth ", gt.sizes());

        // Base loss: L1 + SSIM
        auto l1_loss = torch::l1_loss(rendered, gt);
        auto ssim_loss = 1.f - fused_ssim(rendered, gt, "valid", /*train=*/true);
        torch::Tensor loss = (1.f - opt_params.lambda_dssim) * l1_loss +
                             opt_params.lambda_dssim * ssim_loss;

        // Regularization terms
        if (opt_params.opacity_reg > 0.0f) {
            auto opacity_l1 = torch::abs(splatData.get_opacity()).mean();
            loss += opt_params.opacity_reg * opacity_l1;
        }

        if (opt_params.scale_reg > 0.0f) {
            auto scale_l1 = torch::abs(splatData.get_scaling()).mean();
            loss += opt_params.scale_reg * scale_l1;
        }
        // Total variation loss for bilateral grid
        if (params_.optimization.use_bilateral_grid) {
            loss += params_.optimization.tv_loss_weight * bilateral_grid_->tv_loss();
        }

        return loss;
    }

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

        strategy_->initialize(params.optimization);

        // Initialize bilateral grid if enabled
        initialize_bilateral_grid();

        background_ = torch::tensor({0.f, 0.f, 0.f}, torch::TensorOptions().dtype(torch::kFloat32));
        background_ = background_.to(torch::kCUDA);

        progress_ = std::make_unique<TrainingProgress>(
            params.optimization.iterations,
            /*bar_width=*/100);

        // Initialize the evaluator - it handles all metrics internally
        evaluator_ = std::make_unique<metrics::MetricsEvaluator>(params);

        // Print render mode configuration
        std::cout << "Render mode: " << params.optimization.render_mode << std::endl;

        std::cout << "Visualization: " << (params.optimization.enable_viz ? "enabled" : "disabled") << std::endl;
    }

    Trainer::~Trainer() {
        // Ensure training is stopped
        stop_requested_ = true;
    }

    GSViewer* Trainer::create_and_get_viewer() {
        if (!params_.optimization.enable_viz) {
            return nullptr;
        }

        if (!viewer_) {
            viewer_ = std::make_unique<GSViewer>("GS-CUDA", 1280, 720);
            viewer_->setTrainer(this);
        }

        return viewer_.get();
    }

    void Trainer::handle_control_requests(int iter) {
        // Handle pause/resume
        if (pause_requested_.load() && !is_paused_.load()) {
            is_paused_ = true;
            progress_->pause();
            std::cout << "\nTraining paused at iteration " << iter << std::endl;
            std::cout << "Click 'Resume Training' to continue." << std::endl;
        } else if (!pause_requested_.load() && is_paused_.load()) {
            is_paused_ = false;
            progress_->resume(iter, current_loss_, static_cast<int>(strategy_->get_model().size()));
            std::cout << "\nTraining resumed at iteration " << iter << std::endl;
        }

        // Handle save request
        if (save_requested_.load()) {
            save_requested_ = false;
            std::cout << "\nSaving checkpoint at iteration " << iter << "..." << std::endl;
            strategy_->get_model().save_ply(params_.dataset.output_path / "checkpoints", iter, /*join=*/true);
            std::cout << "Checkpoint saved to " << (params_.dataset.output_path / "checkpoints").string() << std::endl;
        }

        // Handle stop request - this permanently stops training
        if (stop_requested_.load()) {
            std::cout << "\nStopping training permanently at iteration " << iter << "..." << std::endl;
            std::cout << "Saving final model..." << std::endl;
            strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/true);
            is_running_ = false;
        }
    }

    bool Trainer::train_step(int iter, Camera* cam, torch::Tensor gt_image, RenderMode render_mode) {
        current_iteration_ = iter;

        // Check control requests at the beginning
        handle_control_requests(iter);

        // If stop requested, return false to end training
        if (stop_requested_) {
            return false;
        }

        // If paused, wait
        while (is_paused_ && !stop_requested_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            handle_control_requests(iter);
        }

        // Check stop again after potential pause
        if (stop_requested_) {
            return false;
        }

        // Use the render mode from parameters
        auto render_fn = [this, &cam, render_mode]() {
            return gs::rasterize(
                *cam,
                strategy_->get_model(),
                background_,
                1.0f,
                false,
                false,
                render_mode);
        };

        RenderOutput r_output;

        if (viewer_) {
            std::lock_guard<std::mutex> lock(viewer_->splat_mtx_);
            r_output = render_fn();
        } else {
            r_output = render_fn();
        }

        // Apply bilateral grid if enabled
        if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
            r_output.image = bilateral_grid_->apply(r_output.image, cam->uid());
        }
        // Compute loss using the factored-out function
        torch::Tensor loss = compute_loss(r_output,
                                          gt_image,
                                          strategy_->get_model(),
                                          params_.optimization);

        current_loss_ = loss.item<float>();

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
                    const bool join_threads = (iter == params_.optimization.save_steps.back());
                    strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/join_threads);
                }
            }

            auto do_strategy = [&]() {
                strategy_->post_backward(iter, r_output);
                strategy_->step(iter);
            };

            if (viewer_) {
                std::lock_guard<std::mutex> lock(viewer_->splat_mtx_);
                do_strategy();
            } else {
                do_strategy();
            }

            if (params_.optimization.use_bilateral_grid) {
                bilateral_grid_optimizer_->step();
                bilateral_grid_optimizer_->zero_grad(true);
            }
        }

        progress_->update(iter, loss.item<float>(),
                          static_cast<int>(strategy_->get_model().size()),
                          strategy_->is_refining(iter));

        if (viewer_) {
            if (viewer_->info_) {
                auto& info = viewer_->info_;
                std::lock_guard<std::mutex> lock(viewer_->info_->mtx);
                info->updateProgress(iter, params_.optimization.iterations);
                info->updateNumSplats(static_cast<size_t>(strategy_->get_model().size()));
                info->updateLoss(loss.item<float>());
            }

            if (viewer_->notifier_) {
                auto& notifier = viewer_->notifier_;
                std::unique_lock<std::mutex> lock(notifier->mtx);
                notifier->cv.wait(lock, [&notifier] { return notifier->ready; });
            }
        }

        // Return true if we should continue training
        return iter < params_.optimization.iterations && !stop_requested_;
    }

    void Trainer::train() {
        is_running_ = false; // Don't start running until notified
        training_complete_ = false;

        // Wait for the start signal from GUI if visualization is enabled
        if (viewer_ && viewer_->notifier_) {
            auto& notifier = viewer_->notifier_;
            std::unique_lock<std::mutex> lock(notifier->mtx);
            notifier->cv.wait(lock, [&notifier] { return notifier->ready; });
        }

        is_running_ = true; // Now we can start

        int iter = 1;
        const int epochs_needed = (params_.optimization.iterations + train_dataset_size_ - 1) / train_dataset_size_;

        const int num_workers = 4;

        const RenderMode render_mode = stringToRenderMode(params_.optimization.render_mode);

        bool should_continue = true;

        for (int epoch = 0; epoch < epochs_needed && should_continue; ++epoch) {
            auto train_dataloader = create_dataloader_from_dataset(train_dataset_, num_workers);

            for (auto& batch : *train_dataloader) {
                auto camera_with_image = batch[0].data;
                Camera* cam = camera_with_image.camera;
                torch::Tensor gt_image = std::move(camera_with_image.image);

                should_continue = train_step(iter, cam, gt_image, render_mode);

                if (!should_continue) {
                    break;
                }

                ++iter;
            }
        }

        // Final save if not already saved by stop request
        if (!stop_requested_) {
            strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/true);
        }

        progress_->complete();
        evaluator_->save_report();
        progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));

        is_running_ = false;
        training_complete_ = true;
    }

} // namespace gs