#include "core/trainer.hpp"
#include "core/rasterizer.hpp"
#include "kernels/fused_ssim.cuh"
#include "visualizer/detail.hpp"
#include <chrono>
#include <expected>
#include <numeric>
#include <print>

namespace gs {

    static inline torch::Tensor ensure_4d(const torch::Tensor& image) {
        return image.dim() == 3 ? image.unsqueeze(0) : image;
    }

    std::expected<void, std::string> Trainer::initialize_bilateral_grid() {
        if (!params_.optimization.use_bilateral_grid) {
            return {};
        }

        try {
            bilateral_grid_ = std::make_unique<gs::BilateralGrid>(
                train_dataset_size_,
                params_.optimization.bilateral_grid_X,
                params_.optimization.bilateral_grid_Y,
                params_.optimization.bilateral_grid_W);

            bilateral_grid_optimizer_ = std::make_unique<torch::optim::Adam>(
                std::vector<torch::Tensor>{bilateral_grid_->parameters()},
                torch::optim::AdamOptions(params_.optimization.bilateral_grid_lr)
                    .eps(1e-15));

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize bilateral grid: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_photometric_loss(
        const RenderOutput& render_output,
        const torch::Tensor& gt_image,
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {

        try {
            // Ensure images have same dimensions
            torch::Tensor rendered = render_output.image;
            torch::Tensor gt = gt_image;

            // Ensure both tensors are 4D (batch, height, width, channels)
            rendered = rendered.dim() == 3 ? rendered.unsqueeze(0) : rendered;
            gt = gt.dim() == 3 ? gt.unsqueeze(0) : gt;

            TORCH_CHECK(rendered.sizes() == gt.sizes(),
                        "ERROR: size mismatch – rendered ", rendered.sizes(),
                        " vs. ground truth ", gt.sizes());

            // Base loss: L1 + SSIM
            auto l1_loss = torch::l1_loss(rendered, gt);
            auto ssim_loss = 1.f - fused_ssim(rendered, gt, "valid", /*train=*/true);
            torch::Tensor loss = (1.f - opt_params.lambda_dssim) * l1_loss +
                                 opt_params.lambda_dssim * ssim_loss;
            return loss;
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing photometric loss: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_scale_reg_loss(
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {

        try {
            if (opt_params.scale_reg > 0.0f) {
                auto scale_l1 = splatData.get_scaling().mean();
                return opt_params.scale_reg * scale_l1;
            }
            return torch::zeros({1}, torch::kFloat32).requires_grad_();
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing scale regularization loss: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_opacity_reg_loss(
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {

        try {
            if (opt_params.opacity_reg > 0.0f) {
                auto opacity_l1 = splatData.get_opacity().mean();
                return opt_params.opacity_reg * opacity_l1;
            }
            return torch::zeros({1}, torch::kFloat32).requires_grad_();
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing opacity regularization loss: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_bilateral_grid_tv_loss(
        const std::unique_ptr<gs::BilateralGrid>& bilateral_grid,
        const param::OptimizationParameters& opt_params) {

        try {
            if (opt_params.use_bilateral_grid) {
                return opt_params.tv_loss_weight * bilateral_grid->tv_loss();
            }
            return torch::zeros({1}, torch::kFloat32).requires_grad_();
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing bilateral grid TV loss: {}", e.what()));
        }
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

            std::println("Created train/val split: {} train, {} val images",
                         train_dataset_->size().value(),
                         val_dataset_->size().value());
        } else {
            // Use all images for training
            train_dataset_ = dataset;
            val_dataset_ = nullptr;

            std::println("Using all {} images for training (no evaluation)",
                         train_dataset_->size().value());
        }

        if (params_.optimization.preload_to_ram) {
            std::cout << "Preload to RAM enabled. Caching datasets..." << std::endl;
            if (train_dataset_) {
                train_dataset_->preload_data();
            }
            if (val_dataset_) {
                val_dataset_->preload_data();
            }
        } else {
            std::cout << "Loading dataset from disk on-the-fly." << std::endl;
        }

        train_dataset_size_ = train_dataset_->size().value();

        strategy_->initialize(params.optimization);

        // Initialize bilateral grid if enabled
        if (auto result = initialize_bilateral_grid(); !result) {
            throw std::runtime_error(result.error());
        }

        background_ = torch::tensor({0.f, 0.f, 0.f}, torch::TensorOptions().dtype(torch::kFloat32));
        background_ = background_.to(torch::kCUDA);

        // Only create progress bar if no viewer
        if (!viewer_) {
            progress_ = std::make_unique<TrainingProgress>(
                params.optimization.iterations,
                /*bar_width=*/100);
        }

        // Initialize the evaluator - it handles all metrics internally
        evaluator_ = std::make_unique<metrics::MetricsEvaluator>(params);

        // Print render mode configuration
        std::println("Render mode: {}", params.optimization.render_mode);
        std::println("Visualization: {}", params.optimization.headless ? "disabled" : "enabled");
    }

    Trainer::~Trainer() {
        // Ensure training is stopped
        stop_requested_ = true;
    }

    void Trainer::handle_control_requests(int iter, std::stop_token stop_token) {
        // Check stop token first
        if (stop_token.stop_requested()) {
            stop_requested_ = true;
            return;
        }

        // Handle pause/resume
        if (pause_requested_.load() && !is_paused_.load()) {
            is_paused_ = true;
            if (!viewer_ && progress_) {
                progress_->pause();
            }
            std::println("\nTraining paused at iteration {}", iter);
            std::println("Click 'Resume Training' to continue.");
        } else if (!pause_requested_.load() && is_paused_.load()) {
            is_paused_ = false;
            if (!viewer_ && progress_) {
                progress_->resume(iter, current_loss_.load(), static_cast<int>(strategy_->get_model().size()));
            }
            std::println("\nTraining resumed at iteration {}", iter);
        }

        // Handle save request
        if (save_requested_.exchange(false)) {
            std::println("\nSaving checkpoint at iteration {}...", iter);
            auto checkpoint_path = params_.dataset.output_path / "checkpoints";
            strategy_->get_model().save_ply(checkpoint_path, iter, /*join=*/true);
            std::println("Checkpoint saved to {}", checkpoint_path.string());

            // Publish checkpoint saved event
            if (event_bus_) {
                publishCheckpointSaved(iter, checkpoint_path);
            }
        }

        // Handle stop request - this permanently stops training
        if (stop_requested_.load()) {
            std::println("\nStopping training permanently at iteration {}...", iter);
            std::println("Saving final model...");
            strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/true);
            is_running_ = false;
        }
    }

    void Trainer::publishTrainingProgress(int iteration, float loss, int num_gaussians, bool is_refining) {
        if (event_bus_) {
            event_bus_->publish(TrainingProgressEvent{
                iteration, loss, num_gaussians, is_refining});
        }
    }

    void Trainer::publishCheckpointSaved(int iteration, const std::filesystem::path& path) {
        if (event_bus_) {
            event_bus_->publish(CheckpointSavedEvent{
                iteration, path});

            event_bus_->publish(LogMessageEvent{
                LogMessageEvent::Level::Info,
                std::format("Checkpoint saved at iteration {}", iteration),
                "Trainer"});
        }
    }

    void Trainer::publishModelUpdated(int iteration, size_t num_gaussians) {
        if (event_bus_) {
            event_bus_->publish(ModelUpdatedEvent{
                iteration, num_gaussians});
        }
    }

    std::expected<Trainer::StepResult, std::string> Trainer::train_step(
        int iter,
        Camera* cam,
        torch::Tensor gt_image,
        RenderMode render_mode,
        std::stop_token stop_token) {

        try {
            if (cam->radial_distortion().numel() != 0 ||
                cam->tangential_distortion().numel() != 0) {
                return std::unexpected("Training on cameras with distortion is not supported yet.");
            }
            if (cam->camera_model_type() != gsplat::CameraModelType::PINHOLE) {
                return std::unexpected("Training on cameras with non-pinhole model is not supported yet.");
            }

            current_iteration_ = iter;

            // Check control requests at the beginning
            handle_control_requests(iter, stop_token);

            // If stop requested, return Stop
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            // If paused, wait
            while (is_paused_.load() && !stop_requested_.load() && !stop_token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                handle_control_requests(iter, stop_token);
            }

            // Check stop again after potential pause
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            // Use the render mode from parameters
            auto render_fn = [this, &cam, render_mode]() {
                return gs::rasterize(
                    *cam,
                    strategy_->get_model(),
                    background_,
                    1.0f,
                    false,
                    params_.optimization.antialiasing,
                    render_mode);
            };

            RenderOutput r_output = render_fn();

            // Apply bilateral grid if enabled
            if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                r_output.image = bilateral_grid_->apply(r_output.image, cam->uid());
            }

            // Compute losses using the factored-out functions
            auto loss_result = compute_photometric_loss(r_output,
                                                        gt_image,
                                                        strategy_->get_model(),
                                                        params_.optimization);
            if (!loss_result) {
                return std::unexpected(loss_result.error());
            }

            torch::Tensor loss = *loss_result;
            loss.backward();
            float loss_value = loss.item<float>();

            // Scale regularization loss
            auto scale_loss_result = compute_scale_reg_loss(strategy_->get_model(), params_.optimization);
            if (!scale_loss_result) {
                return std::unexpected(scale_loss_result.error());
            }
            loss = *scale_loss_result;
            loss.backward();
            loss_value += loss.item<float>();

            // Opacity regularization loss
            auto opacity_loss_result = compute_opacity_reg_loss(strategy_->get_model(), params_.optimization);
            if (!opacity_loss_result) {
                return std::unexpected(opacity_loss_result.error());
            }
            loss = *opacity_loss_result;
            loss.backward();
            loss_value += loss.item<float>();

            // Bilateral grid TV loss
            auto tv_loss_result = compute_bilateral_grid_tv_loss(bilateral_grid_, params_.optimization);
            if (!tv_loss_result) {
                return std::unexpected(tv_loss_result.error());
            }
            loss = *tv_loss_result;
            loss.backward();
            loss_value += loss.item<float>();

            current_loss_ = loss_value;

            // Publish training progress event (throttled to reduce GUI updates)
            if (event_bus_ && (iter % 10 == 0 || iter == 1)) { // Only update every 10 iterations
                publishTrainingProgress(
                    iter,
                    loss_value,
                    static_cast<int>(strategy_->get_model().size()),
                    strategy_->is_refining(iter));
            }

            {
                torch::NoGradGuard no_grad;

                // Lock for writing during parameter updates
                {
                    std::unique_lock<std::shared_mutex> lock(render_mutex_);

                    // Execute strategy post-backward and step
                    strategy_->post_backward(iter, r_output);
                    strategy_->step(iter);

                    if (params_.optimization.use_bilateral_grid) {
                        bilateral_grid_optimizer_->step();
                        bilateral_grid_optimizer_->zero_grad(true);
                    }

                    // Publish model updated event
                    if (event_bus_) {
                        publishModelUpdated(iter, strategy_->get_model().size());
                    }
                }

                // Clean evaluation - let the evaluator handle everything
                if (evaluator_->is_enabled() && evaluator_->should_evaluate(iter)) {
                    evaluator_->print_evaluation_header(iter);
                    auto metrics = evaluator_->evaluate(iter,
                                                        strategy_->get_model(),
                                                        val_dataset_,
                                                        background_);
                    std::println("{}", metrics.to_string());
                }

                // Save model at specified steps
                if (!params_.optimization.skip_intermediate_saving) {
                    for (size_t save_step : params_.optimization.save_steps) {
                        if (iter == static_cast<int>(save_step) && iter != params_.optimization.iterations) {
                            const bool join_threads = (iter == params_.optimization.save_steps.back());
                            auto save_path = params_.dataset.output_path;
                            strategy_->get_model().save_ply(save_path, iter, /*join=*/join_threads);

                            // Publish checkpoint saved event
                            if (event_bus_) {
                                publishCheckpointSaved(iter, save_path);
                            }
                        }
                    }
                }
            }

            if (!viewer_ && progress_) {
                progress_->update(iter, current_loss_.load(),
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            if (viewer_) {
                if (viewer_->info_) {
                    auto& info = viewer_->info_;
                    info->updateProgress(iter, params_.optimization.iterations);
                    info->updateNumSplats(static_cast<size_t>(strategy_->get_model().size()));
                    info->updateLoss(current_loss_.load());
                }
            }

            // Return Continue if we should continue training
            if (iter < params_.optimization.iterations && !stop_requested_.load() && !stop_token.stop_requested()) {
                return StepResult::Continue;
            } else {
                return StepResult::Stop;
            }

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Training step failed: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::train(std::stop_token stop_token) {
        is_running_ = false;
        training_complete_ = false;

        if (viewer_ && viewer_->notifier_) {
            // Simple spin wait with atomic
            while (!viewer_->notifier_->ready.load() && !stop_token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }

        is_running_ = true; // Now we can start

        try {
            int iter = 1;
            const int epochs_needed = (params_.optimization.iterations + train_dataset_size_ - 1) / train_dataset_size_;
            const int num_workers = 4;
            const RenderMode render_mode = stringToRenderMode(params_.optimization.render_mode);

            if (!viewer_ && progress_) {
                progress_->update(iter, current_loss_.load(),
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            for (int epoch = 0; epoch < epochs_needed; ++epoch) {
                if (stop_token.stop_requested() || stop_requested_.load()) {
                    break;
                }

                auto train_dataloader = create_dataloader_from_dataset(train_dataset_, num_workers);

                for (auto& batch : *train_dataloader) {
                    if (stop_token.stop_requested() || stop_requested_.load()) {
                        break;
                    }

                    auto camera_with_image = batch[0].data;
                    Camera* cam = camera_with_image.camera;
                    torch::Tensor gt_image = std::move(camera_with_image.image);

                    auto step_result = train_step(iter, cam, gt_image, render_mode, stop_token);
                    if (!step_result) {
                        return std::unexpected(step_result.error());
                    }

                    if (*step_result == StepResult::Stop) {
                        goto training_complete; // Break out of nested loops
                    }

                    ++iter;
                }
            }

training_complete:
            // Final save if not already saved by stop request
            if (!stop_requested_.load() && !stop_token.stop_requested()) {
                auto final_path = params_.dataset.output_path;
                strategy_->get_model().save_ply(final_path, iter, /*join=*/true);

                // Publish final checkpoint saved event
                if (event_bus_) {
                    publishCheckpointSaved(iter, final_path);

                    event_bus_->publish(LogMessageEvent{
                        LogMessageEvent::Level::Info,
                        std::format("Training completed. Final model saved at iteration {}", iter),
                        "Trainer"});
                }
            }

            if (!viewer_ && progress_) {
                progress_->complete();
            }
            evaluator_->save_report();
            if (!viewer_ && progress_) {
                progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));
            }

            is_running_ = false;
            training_complete_ = true;

            return {};

        } catch (const std::exception& e) {
            is_running_ = false;
            return std::unexpected(std::format("Training failed: {}", e.what()));
        }
    }

} // namespace gs