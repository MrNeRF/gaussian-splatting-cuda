/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "trainer.hpp"
#include "components/bilateral_grid.hpp"
#include "components/poseopt.hpp"
#include "components/sparsity_optimizer.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "kernels/fused_ssim.cuh"
#include "rasterization/fast_rasterizer.hpp"
#include "rasterization/rasterizer.hpp"
#include <ATen/cuda/CUDAEvent.h>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <expected>
#include <memory>

namespace gs::training {

    void Trainer::cleanup() {
        LOG_DEBUG("Cleaning up trainer for re-initialization");

        // Stop any ongoing operations
        stop_requested_ = true;

        // Wait for callback to finish if busy
        if (callback_busy_.load()) {
            callback_stream_.synchronize();
            callback_busy_.store(false);
        }

        // Reset all components
        progress_.reset();
        bilateral_grid_.reset();
        bilateral_grid_optimizer_.reset();
        poseopt_module_.reset();
        poseopt_optimizer_.reset();
        sparsity_optimizer_.reset();
        evaluator_.reset();

        // Clear datasets (will be recreated)
        train_dataset_.reset();
        val_dataset_.reset();

        // Clear camera cache
        m_cam_id_to_cam.clear();

        // Reset flags
        pause_requested_ = false;
        save_requested_ = false;
        stop_requested_ = false;
        is_paused_ = false;
        is_running_ = false;
        training_complete_ = false;
        ready_to_start_ = false;
        current_iteration_ = 0;
        current_loss_ = 0.0f;

        LOG_DEBUG("Trainer cleanup complete");
    }

    std::expected<void, std::string> Trainer::initialize_bilateral_grid() {
        if (!params_.optimization.use_bilateral_grid) {
            return {};
        }

        try {
            bilateral_grid_ = std::make_unique<BilateralGrid>(
                train_dataset_size_,
                params_.optimization.bilateral_grid_X,
                params_.optimization.bilateral_grid_Y,
                params_.optimization.bilateral_grid_W);

            bilateral_grid_optimizer_ = std::make_unique<torch::optim::Adam>(
                std::vector<torch::Tensor>{bilateral_grid_->parameters()},
                torch::optim::AdamOptions(params_.optimization.bilateral_grid_lr)
                    .eps(1e-15));

            LOG_DEBUG("Bilateral grid initialized with size {}x{}x{}",
                      params_.optimization.bilateral_grid_X,
                      params_.optimization.bilateral_grid_Y,
                      params_.optimization.bilateral_grid_W);
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
        const std::unique_ptr<BilateralGrid>& bilateral_grid,
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

    std::expected<torch::Tensor, std::string> Trainer::compute_sparsity_loss(
        int iter,
        const SplatData& splatData) {
        try {
            if (sparsity_optimizer_ && sparsity_optimizer_->should_apply_loss(iter)) {
                auto loss_result = sparsity_optimizer_->compute_loss(splatData.opacity_raw());
                if (!loss_result) {
                    return std::unexpected(loss_result.error());
                }
                return *loss_result;
            }
            return torch::zeros({1}, torch::kFloat32).to(torch::kCUDA).requires_grad_();
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing sparsity loss: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::handle_sparsity_update(
        int iter,
        SplatData& splatData) {
        try {
            if (sparsity_optimizer_ && sparsity_optimizer_->should_update(iter)) {
                LOG_TRACE("Updating sparsity state at iteration {}", iter);
                auto result = sparsity_optimizer_->update_state(splatData.opacity_raw());
                if (!result) {
                    return std::unexpected(result.error());
                }
            }
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error updating sparsity state: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::apply_sparsity_pruning(
        int iter,
        SplatData& splatData) {
        try {
            if (sparsity_optimizer_ && sparsity_optimizer_->should_prune(iter)) {
                LOG_INFO("Applying sparsity-based pruning at iteration {}", iter);

                auto mask_result = sparsity_optimizer_->get_prune_mask(splatData.opacity_raw());
                if (!mask_result) {
                    return std::unexpected(mask_result.error());
                }

                auto prune_mask = *mask_result;
                int n_prune = prune_mask.sum().item<int>();
                int n_before = splatData.size();

                // Use strategy's remove functionality
                strategy_->remove_gaussians(prune_mask);

                int n_after = splatData.size();
                std::println("Sparsity pruning complete: {} -> {} Gaussians (removed {})",
                            n_before, n_after, n_prune);

                // Clear sparsity optimizer after pruning
                sparsity_optimizer_.reset();
                LOG_DEBUG("Sparsity optimizer cleared after pruning");
            }
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error applying sparsity pruning: {}", e.what()));
        }
    }

    Trainer::Trainer(std::shared_ptr<CameraDataset> dataset,
                     std::unique_ptr<IStrategy> strategy)
        : base_dataset_(std::move(dataset)),
          strategy_(std::move(strategy)) {
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }
        LOG_DEBUG("Trainer constructed with {} cameras", base_dataset_->get_cameras().size());
    }

    void Trainer::load_cameras_info() {
        m_cam_id_to_cam.clear();
        // Setup camera cache
        for (const auto& cam : base_dataset_->get_cameras()) {
            m_cam_id_to_cam[cam->uid()] = cam;
        }
    }

    std::expected<void, std::string> Trainer::initialize(const param::TrainingParameters& params) {
        // Thread-safe initialization using mutex
        std::lock_guard<std::mutex> lock(init_mutex_);

        // Check again after acquiring lock (double-checked locking pattern)
        if (initialized_.load()) {
            LOG_INFO("Re-initializing trainer with new parameters");
            // Clean up existing state for re-initialization
            cleanup();
        }

        LOG_INFO("Initializing trainer with {} iterations", params.optimization.iterations);

        try {
            params_ = params;

            // Handle dataset split based on evaluation flag
            if (params.optimization.enable_eval) {
                // Create train/val split
                train_dataset_ = std::make_shared<CameraDataset>(
                    base_dataset_->get_cameras(), params.dataset, CameraDataset::Split::TRAIN);
                val_dataset_ = std::make_shared<CameraDataset>(
                    base_dataset_->get_cameras(), params.dataset, CameraDataset::Split::VAL);

                LOG_INFO("Created train/val split: {} train, {} val images",
                         train_dataset_->size().value(),
                         val_dataset_->size().value());
            } else {
                // Use all images for training
                train_dataset_ = base_dataset_;
                val_dataset_ = nullptr;

                LOG_INFO("Using all {} images for training (no evaluation)",
                         train_dataset_->size().value());
            }

            train_dataset_size_ = train_dataset_->size().value();

            m_cam_id_to_cam.clear();
            // Setup camera cache
            for (const auto& cam : base_dataset_->get_cameras()) {
                m_cam_id_to_cam[cam->uid()] = cam;
            }
            LOG_DEBUG("Camera cache initialized with {} cameras", m_cam_id_to_cam.size());

            // Re-initialize strategy with new parameters
            strategy_->initialize(params.optimization);
            LOG_DEBUG("Strategy initialized");

            // Initialize bilateral grid if enabled
            if (auto result = initialize_bilateral_grid(); !result) {
                return std::unexpected(result.error());
            }

            // Initialize sparsity optimizer if enabled
            if (params.optimization.enable_sparsity) {
                ADMMSparsityOptimizer::Config sparsity_config{
                    .sparsify_steps = params.optimization.sparsify_steps,
                    .init_rho = params.optimization.init_rho,
                    .prune_ratio = params.optimization.prune_ratio,
                    .update_every = 50
                };

                sparsity_optimizer_ = SparsityOptimizerFactory::create("admm", sparsity_config);

                if (sparsity_optimizer_) {
                    // Initialize with current model opacities
                    auto init_result = sparsity_optimizer_->initialize(strategy_->get_model().opacity_raw());
                    if (!init_result) {
                        LOG_ERROR("Failed to initialize sparsity optimizer: {}", init_result.error());
                        return std::unexpected(init_result.error());
                    }
                    std::println("Sparsity optimization enabled: ADMM with prune_ratio={}",
                                 params.optimization.prune_ratio);
                }
            }

            background_ = torch::tensor({0.f, 0.f, 0.f},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

            if (params.optimization.pose_optimization != "none") {
                if (params.optimization.enable_eval) {
                    return std::unexpected("Evaluating with pose optimization is not supported yet. "
                                           "Please disable pose optimization or evaluation.");
                }
                if (params.optimization.gut) {
                    return std::unexpected("The 3DGUT rasterizer doesn't have camera gradients yet. "
                                           "Please disable pose optimization or disable gut.");
                }
                if (params.optimization.pose_optimization == "direct") {
                    poseopt_module_ = std::make_unique<DirectPoseOptimizationModule>(train_dataset_->get_cameras().size());
                    LOG_DEBUG("Direct pose optimization module created");
                } else if (params.optimization.pose_optimization == "mlp") {
                    poseopt_module_ = std::make_unique<MLPPoseOptimizationModule>(train_dataset_->get_cameras().size());
                    LOG_DEBUG("MLP pose optimization module created");
                } else {
                    return std::unexpected("Invalid pose optimization type: " + params.optimization.pose_optimization);
                }
                poseopt_optimizer_ = std::make_unique<torch::optim::Adam>(
                    std::vector<torch::Tensor>{poseopt_module_->parameters()},
                    torch::optim::AdamOptions(1e-5));
            } else {
                poseopt_module_ = std::make_unique<PoseOptimizationModule>();
            }

            // Create progress bar based on headless flag
            if (params.optimization.headless) {
                progress_ = std::make_unique<TrainingProgress>(
                    params.optimization.iterations,
                    /*update_frequency=*/100);
                LOG_DEBUG("Progress bar initialized for headless mode");
            }

            // Initialize the evaluator - it handles all metrics internally
            evaluator_ = std::make_unique<MetricsEvaluator>(params_);
            LOG_DEBUG("Metrics evaluator initialized");

            // Print configuration
            LOG_INFO("Render mode: {}", params.optimization.render_mode);
            LOG_INFO("Visualization: {}", params.optimization.headless ? "disabled" : "enabled");
            LOG_INFO("Strategy: {}", params.optimization.strategy);

            initialized_ = true;
            LOG_INFO("Trainer initialization complete");
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize trainer: {}", e.what()));
        }
    }

    Trainer::~Trainer() {
        // Ensure training is stopped
        stop_requested_ = true;

        // Wait for callback to finish if busy
        if (callback_busy_.load()) {
            callback_stream_.synchronize();
        }
        LOG_DEBUG("Trainer destroyed");
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
            if (progress_) {
                progress_->pause();
            }
            LOG_INFO("Training paused at iteration {}", iter);
            LOG_DEBUG("Click 'Resume Training' to continue.");
        } else if (!pause_requested_.load() && is_paused_.load()) {
            is_paused_ = false;
            if (progress_) {
                progress_->resume(iter, current_loss_.load(), static_cast<int>(strategy_->get_model().size()));
            }
            LOG_INFO("Training resumed at iteration {}", iter);
        }

        // Handle save request
        if (save_requested_.exchange(false)) {
            LOG_INFO("Saving checkpoint at iteration {}...", iter);
            auto checkpoint_path = params_.dataset.output_path / "checkpoints";
            save_ply(checkpoint_path, iter, /*join=*/true);

            LOG_INFO("Checkpoint saved to {}", checkpoint_path.string());

            // Emit checkpoint saved event
            events::state::CheckpointSaved{
                .iteration = iter,
                .path = checkpoint_path}
                .emit();
        }

        // Handle stop request - this permanently stops training
        if (stop_requested_.load()) {
            LOG_INFO("Stopping training permanently at iteration {}...", iter);
            LOG_DEBUG("Saving final model...");
            save_ply(params_.dataset.output_path, iter, /*join=*/true);
            is_running_ = false;
        }
    }

    std::expected<Trainer::StepResult, std::string> Trainer::train_step(
        int iter,
        Camera* cam,
        torch::Tensor gt_image,
        RenderMode render_mode,
        std::stop_token stop_token) {
        try {
            if (params_.optimization.gut) {
                if (cam->camera_model_type() == gsplat::CameraModelType::ORTHO) {
                    return std::unexpected("Training on cameras with ortho model is not supported yet.");
                }
            } else {
                // Flag is workaround for non-RC datasets with distortion. By default it is off.
                if (!params_.optimization.rc) {
                    if (cam->radial_distortion().numel() != 0 ||
                        cam->tangential_distortion().numel() != 0) {
                        return std::unexpected("You must use --gut option to train on cameras with distortion.");
                    }
                    if (cam->camera_model_type() != gsplat::CameraModelType::PINHOLE) {
                        return std::unexpected("You must use --gut option to train on cameras with non-pinhole model.");
                    }
                }
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

            auto adjusted_cam_pos = poseopt_module_->forward(cam->world_view_transform(), torch::tensor({cam->uid()}));
            auto adjusted_cam = Camera(*cam, adjusted_cam_pos);

            RenderOutput r_output;
            // Use the render mode from parameters
            if (!params_.optimization.gut) {
                r_output = fast_rasterize(adjusted_cam, strategy_->get_model(), background_);
            } else {
                r_output = rasterize(adjusted_cam, strategy_->get_model(), background_, 1.0f, false, false, render_mode,
                                     nullptr);
            }

            // Apply bilateral grid if enabled
            if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                r_output.image = bilateral_grid_->apply(r_output.image, cam->uid());
            }

            // Compute losses
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

            // Add sparsity loss
            auto sparsity_loss_result = compute_sparsity_loss(iter, strategy_->get_model());
            if (!sparsity_loss_result) {
                return std::unexpected(sparsity_loss_result.error());
            }
            loss = *sparsity_loss_result;
            loss.backward();
            loss_value += loss.item<float>();

            // Store the loss value immediately
            current_loss_ = loss_value;

            // Update progress synchronously if needed
            if (progress_) {
                progress_->update(iter, loss_value,
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // Emit training progress event (throttled to reduce GUI updates)
            if (iter % 10 == 0 || iter == 1) {
                // Only update every 10 iterations
                events::state::TrainingProgress{
                    .iteration = iter,
                    .loss = loss_value,
                    .num_gaussians = static_cast<int>(strategy_->get_model().size()),
                    .is_refining = strategy_->is_refining(iter)}
                    .emit();
            }
            {
                torch::NoGradGuard no_grad;

                DeferredEvents deferred;
                {
                    std::unique_lock<std::shared_mutex> lock(render_mutex_);

                    // Execute strategy post-backward and step
                    if (!params_.optimization.enable_sparsity) {
                        strategy_->post_backward(iter, r_output);
                    }
                    strategy_->step(iter);

                    if (params_.optimization.use_bilateral_grid) {
                        bilateral_grid_optimizer_->step();
                        bilateral_grid_optimizer_->zero_grad(true);
                    }
                    if (params_.optimization.pose_optimization != "none") {
                        poseopt_optimizer_->step();
                        poseopt_optimizer_->zero_grad(true);
                    }

                    // Queue event for emission after lock release
                    deferred.add(events::state::ModelUpdated{
                        .iteration = iter,
                        .num_gaussians = static_cast<size_t>(strategy_->get_model().size())});
                } // Lock released here

                // Events automatically emitted here when deferred destructs

                // Handle sparsity updates
                if (auto result = handle_sparsity_update(iter, strategy_->get_model()); !result) {
                    LOG_ERROR("Sparsity update failed: {}", result.error());
                }

                // Apply sparsity pruning if needed
                if (auto result = apply_sparsity_pruning(iter, strategy_->get_model()); !result) {
                    LOG_ERROR("Sparsity pruning failed: {}", result.error());
                }

                // Clean evaluation - let the evaluator handle everything
                if (evaluator_->is_enabled() && evaluator_->should_evaluate(iter)) {
                    evaluator_->print_evaluation_header(iter);
                    auto metrics = evaluator_->evaluate(iter,
                                                        strategy_->get_model(),
                                                        val_dataset_,
                                                        background_);
                    LOG_INFO("{}", metrics.to_string());
                }

                // Save model at specified steps
                if (!params_.optimization.skip_intermediate_saving) {
                    for (size_t save_step : params_.optimization.save_steps) {
                        if (iter == static_cast<int>(save_step) && iter != params_.optimization.iterations) {
                            const bool join_threads = (iter == params_.optimization.save_steps.back());
                            auto save_path = params_.dataset.output_path;
                            save_ply(save_path, iter, /*join=*/join_threads);
                            // Emit checkpoint saved event
                            events::state::CheckpointSaved{
                                .iteration = iter,
                                .path = save_path}
                                .emit();
                        }
                    }
                }

                if (!params_.dataset.timelapse_images.empty() && iter % params_.dataset.timelapse_every == 0) {
                    for (const auto& img_name : params_.dataset.timelapse_images) {
                        auto train_cam = train_dataset_->get_camera_by_filename(img_name);
                        auto val_cam = val_dataset_ ? val_dataset_->get_camera_by_filename(img_name) : std::nullopt;
                        if (train_cam.has_value() || val_cam.has_value()) {
                            Camera* cam_to_use = train_cam.has_value() ? train_cam.value() : val_cam.value();

                            // Image size isn't correct until the image has been loaded once
                            // If we use the camera before it's loaded, it will render images at the non-scaled size
                            if (cam_to_use->camera_height() == cam_to_use->image_height() && params_.dataset.resize_factor != 1) {
                                cam_to_use->load_image_size(params_.dataset.resize_factor);
                            }

                            RenderOutput rendered_timelapse_output = fast_rasterize(
                                *cam_to_use, strategy_->get_model(), background_);

                            // Get folder name to save in by stripping file extension
                            std::string folder_name = img_name;
                            auto last_dot = folder_name.find_last_of('.');
                            if (last_dot != std::string::npos) {
                                folder_name = folder_name.substr(0, last_dot);
                            }

                            auto output_path = params_.dataset.output_path / "timelapse" / folder_name;
                            std::filesystem::create_directories(output_path);

                            image_io::save_image_async(output_path / std::format("{:06d}.jpg", iter),
                                                       rendered_timelapse_output.image);
                        } else {
                            LOG_WARN("Timelapse image '{}' not found in dataset.", img_name);
                        }
                    }
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
        // Check if initialized
        if (!initialized_.load()) {
            return std::unexpected("Trainer not initialized. Call initialize() before train()");
        }

        is_running_ = false;
        training_complete_ = false;
        ready_to_start_ = false; // Reset the flag

        // Event-based ready signaling
        if (!params_.optimization.headless) {
            // Subscribe to start signal (no need to store handle)
            events::internal::TrainingReadyToStart::when([this](const auto&) {
                ready_to_start_ = true;
            });

            // Signal we're ready
            events::internal::TrainerReady{}.emit();

            // Wait for start signal
            LOG_DEBUG("Waiting for start signal from GUI...");
            while (!ready_to_start_.load() && !stop_token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }

        is_running_ = true; // Now we can start
        LOG_INFO("Starting training loop");

        try {
            int iter = 1;
            const int num_workers = 16;
            const RenderMode render_mode = stringToRenderMode(params_.optimization.render_mode);

            if (progress_) {
                progress_->update(iter, current_loss_.load(),
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // Use infinite dataloader to avoid epoch restarts
            auto train_dataloader = create_infinite_dataloader_from_dataset(train_dataset_, num_workers);
            auto loader = train_dataloader->begin();

            LOG_DEBUG("Starting training iterations");
            // Single loop without epochs
            while (iter <= params_.optimization.iterations) {
                if (stop_token.stop_requested() || stop_requested_.load()) {
                    break;
                }

                // Wait for previous callback if still running
                if (callback_busy_.load()) {
                    callback_stream_.synchronize();
                }

                auto& batch = *loader;
                auto camera_with_image = batch[0].data;
                Camera* cam = camera_with_image.camera;
                torch::Tensor gt_image = std::move(camera_with_image.image).to(torch::kCUDA, /*non_blocking=*/true);

                auto step_result = train_step(iter, cam, gt_image, render_mode, stop_token);
                if (!step_result) {
                    return std::unexpected(step_result.error());
                }

                if (*step_result == StepResult::Stop) {
                    break;
                }

                // Launch callback for async progress update (except first iteration)
                if (iter > 1 && callback_) {
                    callback_busy_ = true;
                    auto err = cudaLaunchHostFunc(
                        callback_stream_.stream(),
                        [](void* self) {
                            auto* trainer = static_cast<Trainer*>(self);
                            if (trainer->callback_) {
                                trainer->callback_();
                            }
                            trainer->callback_busy_ = false;
                        },
                        this);
                    if (err != cudaSuccess) {
                        LOG_WARN("Failed to launch callback: {}", cudaGetErrorString(err));
                        callback_busy_ = false;
                    }
                }

                ++iter;
                ++loader;
            }

            // Ensure callback is finished before final save
            if (callback_busy_.load()) {
                callback_stream_.synchronize();
            }

            // Final save if not already saved by stop request
            if (!stop_requested_.load() && !stop_token.stop_requested()) {
                auto final_path = params_.dataset.output_path;
                save_ply(final_path, iter - 1, /*join=*/true);
                // Emit final checkpoint saved event
                events::state::CheckpointSaved{
                    .iteration = iter - 1,
                    .path = final_path}
                    .emit();
            }

            if (progress_) {
                progress_->complete();
            }
            evaluator_->save_report();
            if (progress_) {
                progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));
            }

            is_running_ = false;
            training_complete_ = true;

            LOG_INFO("Training completed successfully");
            return {};
        } catch (const std::exception& e) {
            is_running_ = false;
            return std::unexpected(std::format("Training failed: {}", e.what()));
        }
    }

    std::shared_ptr<const Camera> Trainer::getCamById(int camId) const {
        const auto it = m_cam_id_to_cam.find(camId);
        if (it == m_cam_id_to_cam.end()) {
            LOG_ERROR("getCamById - could not find cam with cam id {}", camId);
            return nullptr;
        }
        return it->second;
    }

    std::vector<std::shared_ptr<const Camera>> Trainer::getCamList() const {
        std::vector<std::shared_ptr<const Camera>> cams;
        cams.reserve(m_cam_id_to_cam.size());
        for (auto& [key, value] : m_cam_id_to_cam) {
            cams.push_back(value);
        }

        return cams;
    }

    void Trainer::save_ply(const std::filesystem::path& save_path, int iter_num, bool join_threads) {
        strategy_->get_model().save_ply(save_path, iter_num + 1, /*join=*/join_threads);
        if (lf_project_) {
            const std::string ply_name = "splat_" + std::to_string(iter_num + 1);
            const std::filesystem::path ply_path = save_path / (ply_name + ".ply");
            lf_project_->addPly(gs::management::PlyData(false, ply_path, iter_num, ply_name));
        }
        LOG_DEBUG("PLY saved: {}", save_path.string());
    }
} // namespace gs::training