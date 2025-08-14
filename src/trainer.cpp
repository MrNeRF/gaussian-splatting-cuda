#include "core/trainer.hpp"
#include "core/fast_rasterizer.hpp"
#include "core/rasterizer.hpp"
#include "kernels/fused_ssim.cuh"
#include <ATen/cuda/CUDAEvent.h>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <expected>
#include <memory>
#include <numeric>
#include <print>
#include "core/rasterizer_autograd.hpp"

namespace gs {

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
    // Place this new static function inside trainer.cpp

    /**
     * @brief Applies a penalty based on the final rendered alpha channel of the image.
     *
     * This method penalizes pixels inside the mask that are semi-transparent and
     * pixels outside the mask that have any opacity. It's a more direct approach
     * than the splat-based penalty.
     *
     * @param base_loss The pre-computed photometric loss.
     * @param rendered_alpha The final accumulated alpha channel from the rasterizer [H, W].
     * @param weights The float weight map, where values > 0.5 define the ROI.
     * @param w The global weight multiplier for this penalty.
     * @param kIn The weight for penalizing transparency INSIDE the mask.
     * @param kOut The weight for penalizing opacity OUTSIDE the mask.
     * @return torch::Tensor The loss with the pixel-based opacity penalty added.
     */
    static torch::Tensor pixelBasedOpacityPenalty(const torch::Tensor& base_loss,
                                                  const torch::Tensor& rendered_alpha,
                                                  const torch::Tensor& weights,
                                                  const float w) {
        
        const float kOut = 0.333f;
        const float kIn =  0.667f;

        // Convert the float weight map to a clean boolean mask
        auto bool_mask = (weights > 0.5f);

        // 1. Penalty for being transparent INSIDE the mask
        // (1.0 - alpha) is high where pixels are transparent.
        // We multiply by the boolean mask to only consider pixels inside the ROI.
        auto inside_penalty_map = (1.0f - rendered_alpha) * bool_mask;

        // 2. Penalty for being opaque OUTSIDE the mask
        // `alpha` is high where pixels are opaque.
        // We multiply by the inverted mask to only consider pixels outside the ROI.
        auto outside_penalty_map = rendered_alpha * (~bool_mask);

        // 3. Combine and normalize the penalties
        // We sum the penalty over all pixels and normalize by the total number of pixels.
        float num_pixels = static_cast<float>(rendered_alpha.numel());
        auto total_penalty = w * (kIn * inside_penalty_map.sum() + kOut * outside_penalty_map.sum()) / num_pixels;

        // 4. Add the penalty to the base loss
        // For pixel-based losses, adding is often more stable than multiplying.
        //return base_loss + 0.01 * total_penalty;
        
        return (base_loss * (1.0f + total_penalty)) + (1e-2f * total_penalty);
    }

    /**
     * @brief Applies a penalty to guide splat opacity based on an attention mask.
     * @param base_loss The pre-computed photometric loss.
     * @param out The RenderOutput from the rasterizer.
     * @param weights The float weight map, where values > 0.5 define the ROI.
     * @param opacities_alpha The opacity alpha values (0-1) from get_opacity().
     * @param w The global weight multiplier for this penalty.
     */
    static torch::Tensor outsideMaskOpacityPenalty(const torch::Tensor& base_loss,
                                                   const RenderOutput& out,
                                                   const torch::Tensor& weights,
                                                   const torch::Tensor& opacities_alpha,
                                                   float w) {
        if (w == 0.0f) {
            return base_loss;
        }
        if (!weights.defined() || !out.means2d.defined() || !out.radii.defined()) {
            return base_loss;
        }

        torch::Tensor visible_mask = (out.radii > 0.0f);
        if (!visible_mask.any().item<bool>()) {
            return base_loss;
        }
        auto visible_indices = visible_mask.nonzero().squeeze(-1);
        auto xy = out.means2d.index({visible_indices});

        auto alpha = opacities_alpha.index({visible_indices});

        const int W = out.width, H = out.height;
        auto x = torch::round(xy.select(1, 0)).to(torch::kLong).clamp(0, W - 1);
        auto y = torch::round(xy.select(1, 1)).to(torch::kLong).clamp(0, H - 1);
        auto linear_indices = y * W + x;

        auto bool_mask = (weights > 0.5f);
        auto is_in_mask = bool_mask.to(torch::kFloat32).flatten();
        auto is_in = is_in_mask.index({linear_indices});
        auto is_out = 1.0f - is_in;

        const float kOut = 2.0f;
        const float kIn = 0.02f;
        auto outside_penalty = alpha * is_out;
        auto inside_penalty = (1.0f - alpha) * is_in;

        auto combined_penalty = w * (kOut * outside_penalty + kIn * inside_penalty);
        auto mean_penalty = combined_penalty.mean();
        
        return (base_loss * (1.0f + mean_penalty)) + (1e-5f*mean_penalty);
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_photometric_loss(const RenderOutput& render_output,
                                                    const torch::Tensor& gt_image,
                                                    const torch::Tensor& weights,
                                                    const float outOfMaskAlphaPenalty,
                                                    const SplatData& splatData,
                                                    const param::OptimizationParameters& opt_params) {

        if (!weights.defined() || weights.numel() == 0) {
            // fallback to the previous mode
            return compute_photometric_loss(render_output, gt_image, splatData, opt_params);
        }

        try {
            // Ensure images have same dimensions
            torch::Tensor rendered = render_output.image;
            torch::Tensor gt = gt_image;

            // Ensure both tensors are 4D (batch, height, width, channels)
            rendered = rendered.dim() == 3 ? rendered.unsqueeze(0) : rendered;
            gt = gt.dim() == 3 ? gt.unsqueeze(0) : gt;

            const int Height = rendered.size(2);
            const int Width = rendered.size(3);
            TORCH_CHECK(rendered.sizes() == gt.sizes(), "ERROR: size mismatch - rendered ", rendered.sizes(), " vs. ground truth ", gt.sizes());
            TORCH_CHECK((Height == weights.size(1) && Width == weights.size(2)),
                        "ERROR: size mismatch - rendered ", rendered.sizes(), " vs. mask ", weights.sizes());

            torch::Tensor W = weights;

            // Pixel-wise L1 map and weighted L1 loss
            auto l1_map = torch::abs(rendered - gt).mean(/*dim=*/1); // Resultado: [B, H, W]
            auto wSum = W.sum().clamp_min(1e-6f);
            auto l1_loss = (l1_map * W).sum() / wSum;

            // 2) pixel-wise SSIM map and weighted SSIM loss
            
            // Compute the SSIM map. Using "valid" padding results in a smaller map.
            auto ssim_map = fused_ssim_map(rendered, gt, "valid", /*train=*/true);

            // Manually crop the weight map `W` to match the `ssim_map` dimensions.
            namespace I = torch::indexing;

            // Get original and target dimensions.
            const int orig_h = W.size(-2);
            const int orig_w = W.size(-1);
            const int target_h = ssim_map.size(-2);
            const int target_w = ssim_map.size(-1);

            // Calculate offsets for a centered crop.
            const int crop_h_start = (orig_h - target_h) / 2;
            const int crop_w_start = (orig_w - target_w) / 2;

            // Apply the crop using tensor slicing.
            torch::Tensor W_cropped = W.index({I::Slice(),
                                                I::Slice(crop_h_start, crop_h_start + target_h),
                                                I::Slice(crop_w_start, crop_w_start + target_w)});
            // Compute the weighted SSIM loss.
            auto ssim_loss_map = 1.0f - ssim_map;

            // Normalize by the sum of the cropped weights for correct loss scaling.
            auto W_cropped_sum = W_cropped.sum().clamp_min(1e-6f);
            auto ssim_loss = (ssim_loss_map * W_cropped).sum() / W_cropped_sum;

            // 3) combined loss
            auto loss = (1.0f - opt_params.lambda_dssim) * l1_loss + opt_params.lambda_dssim * ssim_loss;
            
        
            if (outOfMaskAlphaPenalty > 0) {
                if (!render_output.image.defined() || render_output.image.numel() == 0) {
                    printf("Image failed!\n");
                } else if (!render_output.alpha.defined() || render_output.alpha.numel() == 0) {
                    //printf("Alpha failed!\n");
                }
                else {
                    /* auto opacity = splatData.get_opacity();
                    loss = outsideMaskOpacityPenalty(loss,
                                                    render_output,
                                                    weights,
                                                    opacity,
                                                    outOfMaskAlphaPenalty);*/

                    loss = pixelBasedOpacityPenalty(loss,
                                                    render_output.alpha.squeeze(0), // Squeeze to [H, W] if needed
                                                    weights,
                                                    outOfMaskAlphaPenalty);
                }
            }
        

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

        train_dataset_size_ = train_dataset_->size().value();

        strategy_->initialize(params.optimization);

        // Initialize bilateral grid if enabled
        if (auto result = initialize_bilateral_grid(); !result) {
            throw std::runtime_error(result.error());
        }

        background_ = torch::tensor({0.f, 0.f, 0.f}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        // Create progress bar based on headless flag
        if (params.optimization.headless) {
            progress_ = std::make_unique<TrainingProgress>(
                params.optimization.iterations,
                /*update_frequency=*/100);
        }

        // Initialize the evaluator - it handles all metrics internally
        evaluator_ = std::make_unique<metrics::MetricsEvaluator>(params);

        // setup camera cache
        for (const auto& cam : dataset->get_cameras()) {
            m_cam_id_to_cam[cam->uid()] = cam;
        }

        // Print render mode configuration
        std::println("Render mode: {}", params.optimization.render_mode);
        std::println("Visualization: {}", params.optimization.headless ? "disabled" : "enabled");
        std::println("Strategy: {}", params.optimization.strategy);
    }

    Trainer::~Trainer() {
        // Ensure training is stopped
        stop_requested_ = true;

        // Wait for callback to finish if busy
        if (callback_busy_.load()) {
            callback_stream_.synchronize();
        }
        // unsubscribe - because when the event emits while class destroyed we get crash
        gs::event::bus().remove<gs::events::internal::TrainingReadyToStart>(train_started_handle_);
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
            std::println("\nTraining paused at iteration {}", iter);
            std::println("Click 'Resume Training' to continue.");
        } else if (!pause_requested_.load() && is_paused_.load()) {
            is_paused_ = false;
            if (progress_) {
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

            // Emit checkpoint saved event
            events::state::CheckpointSaved{
                .iteration = iter,
                .path = checkpoint_path}
                .emit();
        }

        // Handle stop request - this permanently stops training
        if (stop_requested_.load()) {
            std::println("\nStopping training permanently at iteration {}...", iter);
            std::println("Saving final model...");
            strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/true);
            is_running_ = false;
        }
    }

    std::expected<Trainer::StepResult, std::string> Trainer::train_step(
        int iter,
        Camera* cam,
        torch::Tensor gt_image,
        torch::Tensor weights,
        RenderMode render_mode,
        bool out_of_mask_penalty,
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
            RenderOutput r_output = fast_rasterize(*cam, strategy_->get_model(), background_);

            // Apply bilateral grid if enabled
            if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                r_output.image = bilateral_grid_->apply(r_output.image, cam->uid());
            }

            // Compute loss using the factored-out function
            std::expected<torch::Tensor, std::string> loss_result;
            
            const int total_iters = params_.optimization.iterations;
            const int warmup_end_iter = total_iters * 0.1f;

            if (!weights.defined() || iter < warmup_end_iter) {
                loss_result = compute_photometric_loss(r_output,
                                                        gt_image,
                                                        strategy_->get_model(),
                                                        params_.optimization);
            } 
            else {
                const int full_penalty_end_iter = total_iters * 0.5f;
                const int decay_end_iter = total_iters * 0.8f;
                float current_penalty_w = 0.0f;
                if (out_of_mask_penalty) {
                    if (iter <= full_penalty_end_iter) {
                        current_penalty_w = 1;
                    } else if (iter < decay_end_iter) {
                        const int decay_start_iter = full_penalty_end_iter;
                        const int decay_duration = decay_end_iter - decay_start_iter;
                        const float decay_progress = static_cast<float>(iter - decay_start_iter) / decay_duration;
                        current_penalty_w = 1.0f - decay_progress;
                    }
                }

                loss_result = compute_photometric_loss( r_output,
                                                        gt_image,
                                                        weights,
                                                        current_penalty_w,
                                                        strategy_->get_model(),
                                                        params_.optimization);
            }
            
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

            // Store the loss value immediately
            current_loss_ = loss_value;

            // Update progress synchronously if needed
            if (progress_) {
                progress_->update(iter, loss_value,
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // Emit training progress event (throttled to reduce GUI updates)
            if (iter % 10 == 0 || iter == 1) { // Only update every 10 iterations
                events::state::TrainingProgress{
                    .iteration = iter,
                    .loss = loss_value,
                    .num_gaussians = static_cast<int>(strategy_->get_model().size()),
                    .is_refining = strategy_->is_refining(iter)}
                    .emit();
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

                    // Emit model updated event
                    events::state::ModelUpdated{
                        .iteration = iter,
                        .num_gaussians = static_cast<size_t>(strategy_->get_model().size())}
                        .emit();
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

                            // Emit checkpoint saved event
                            events::state::CheckpointSaved{
                                .iteration = iter,
                                .path = save_path}
                                .emit();
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
        is_running_ = false;
        training_complete_ = false;

        // Event-based ready signaling
        if (!params_.optimization.headless) {
            std::atomic<bool> ready{false};

            // Subscribe temporarily to start signal
            train_started_handle_ = events::internal::TrainingReadyToStart::when([&ready](const auto&) {
                ready = true;
            });

            // Signal we're ready
            events::internal::TrainerReady{}.emit();

            // Wait for start signal
            while (!ready.load() && !stop_token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }

        is_running_ = true; // Now we can start

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


                std::expected<Trainer::StepResult, std::string> step_result;
                if (!params_.optimization.use_attention_mask || !camera_with_image.attentionMask.defined()) {
                    step_result = train_step(iter, cam, gt_image, torch::Tensor(), render_mode, false, stop_token);
                } else {
                    torch::Tensor attention_image = std::move(camera_with_image.attentionMask);
                    bool out_of_mask_penalty = true;
                    step_result = train_step(iter, cam, gt_image, attention_image, render_mode, out_of_mask_penalty, stop_token);
                }

                
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
                        std::cerr << "Warning: Failed to launch callback: " << cudaGetErrorString(err) << std::endl;
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

            training_complete:
                if (params_.optimization.use_attention_mask)
                    prune_after_training(0.8);
            
            // Final save if not already saved by stop request
            if (!stop_requested_.load() && !stop_token.stop_requested()) {
                auto final_path = params_.dataset.output_path;
                strategy_->get_model().save_ply(final_path, iter, /*join=*/true);

                // Emit final checkpoint saved event
                events::state::CheckpointSaved{
                    .iteration = iter,
                    .path = final_path}
                    .emit();

                events::notify::Log{
                    .level = events::notify::Log::Level::Info,
                    .message = std::format("Training completed. Final model saved at iteration {}", iter),
                    .source = "Trainer"}
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

            return {};

        } catch (const std::exception& e) {
            is_running_ = false;
            return std::unexpected(std::format("Training failed: {}", e.what()));
        }
    }

    std::shared_ptr<const Camera> Trainer::getCamById(int camId) const {
        const auto it = m_cam_id_to_cam.find(camId);
        if (it == m_cam_id_to_cam.end()) {
            std::cerr << "error: getCamById - could not find cam with cam id " << camId << std::endl;
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

    void Trainer::prune_after_training(float threshold) {
        torch::NoGradGuard no_grad;

        // 0) Access current Gaussian model
        SplatData& model = strategy_->get_model();
        const int64_t N = model.get_means().size(0);
        if (N == 0)
            return;

        // 1) Allocate vote buffers on GPU
        torch::Tensor pos = torch::zeros({N}, torch::kInt32).cuda(); // positive votes
        torch::Tensor tot = torch::zeros({N}, torch::kInt32).cuda(); // total votes

        // 2) Build the same DataLoader you already use (mask comes from batch.data.attentionMask)
        std::cout << "Optimized pruning (projection-only): Using DataLoader..." << std::endl;
        auto pruning_dataloader = torch::data::make_data_loader(
            *train_dataset_,
            torch::data::samplers::SequentialSampler(train_dataset_->size().value()),
            torch::data::DataLoaderOptions().batch_size(1).workers(4) // keep your current setting
        );

        // 3) Prepare model tensors once (CUDA)
        auto means3D = model.get_means();      // [N,3], CUDA
        auto scales = model.get_scaling();     // [N,3], CUDA
        auto rotations = model.get_rotation(); // [N,4] or [N,3x3], CUDA
        auto opacities = model.get_opacity();  // [N] or [N,1], CUDA / may be undefined but Tensor

        if (opacities.defined() && opacities.dim() == 2 && opacities.size(-1) == 1) {
            opacities = opacities.squeeze(-1);
        }

        // Projection numeric constants (keep in sync with rasterizer)
        const float eps2d = 0.3f;
        const float near_plane = 0.01f;
        const float far_plane = 10000.0f;
        const float radius_clip = 0.0f;
        const float scaling_mod = 1.0f;

        std::cout << "Optimized pruning: Fetched..." << std::endl;
        int index = 1;

        for (auto& batch : *pruning_dataloader) {
            // Progress heartbeat
            printf("\rPrunning image %i", index++);
            fflush(stdout);

            // 3.a) Unpack camera and attention mask from your batch
            auto camera_with_data = batch[0].data;
            Camera* cam = camera_with_data.camera;
            torch::Tensor float_weight_map = camera_with_data.attentionMask;
            if (!cam || !float_weight_map.defined()) {
                continue;
            }

            // Make a [H,W] boolean mask on CPU
            auto bool_mask_3d = (float_weight_map > 0.5f); // [1,H,W] or [H,W]
            auto bool_mask = (bool_mask_3d.dim() == 3 && bool_mask_3d.size(0) == 1)
                                 ? bool_mask_3d.squeeze(0)
                                 : bool_mask_3d;
            TORCH_CHECK(bool_mask.dim() == 2, "Attention mask must be [H,W] or [1,H,W]");
            bool_mask = bool_mask.contiguous(); // CPU bool [H,W]

            const int H = static_cast<int>(bool_mask.size(0));
            const int W = static_cast<int>(bool_mask.size(1));

            // 3.b) Camera tensors (CUDA)
            auto viewmat = cam->world_view_transform().to(torch::kCUDA); // [1,4,4]
            auto K = cam->K().to(torch::kCUDA);                          // [1,3,3] or [3,3]

            // Prefer camera's declared image size for projection
            const int image_width = static_cast<int>(cam->image_width());
            const int image_height = static_cast<int>(cam->image_height());

            // 3.c) Projection-only (no rendering)
            auto proj_settings = torch::tensor(
                {static_cast<float>(image_width),
                 static_cast<float>(image_height),
                 eps2d, near_plane, far_plane, radius_clip, scaling_mod},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

            // NOTE: This lives in the same namespace (gs) – call without "gs::" prefix,
            // exactly like in rasterizer.cpp.
            auto proj_out = ProjectionFunction::apply(
                means3D, rotations, scales, opacities, viewmat, K, proj_settings);

            torch::Tensor radii2 = proj_out[0];  // [1,N,2] or [N,2]
            torch::Tensor means2d = proj_out[1]; // [1,N,2] or [N,2]

            if (!radii2.defined() || !means2d.defined()) {
                printf("radii2 or means2d failed\n");
                continue; // Projection failed, skip gracefully
            }
            if (radii2.dim() == 3 && radii2.size(0) == 1)
                radii2 = radii2.squeeze(0);
            if (means2d.dim() == 3 && means2d.size(0) == 1)
                means2d = means2d.squeeze(0);

            // 3.d) Visibility: positive projected radius
            torch::Tensor visible;
            if (radii2.dim() == 2 && radii2.size(1) >= 1) {
                visible = (radii2 > 0.0f).all(-1); // [N]
            } else if (radii2.dim() == 1) {
                visible = (radii2 > 0.0f);
            } else {
                continue;
            }
            if (!visible.any().item<bool>())
                continue;

            auto idx = visible.nonzero().squeeze(); // [M], CUDA

            // 3.e) Gather 2D positions (CPU) and vote on the CPU mask
            auto xy_cuda = means2d.index({idx}); // [M,2], CUDA
            auto xy = xy_cuda.detach().to(torch::kCPU);

            // Round and clamp to mask bounds
            auto x = torch::round(xy.select(1, 0)).to(torch::kLong).clamp(0, W - 1);
            auto y = torch::round(xy.select(1, 1)).to(torch::kLong).clamp(0, H - 1);
            auto lin = y * W + x; // [M], CPU long

            // Sample mask and accumulate votes
            auto white_cpu = bool_mask.flatten().index({lin});                  // CPU bool
            auto white_i32_cuda = white_cpu.to(torch::kInt32).to(torch::kCUDA); // CUDA int32

            pos.index_add_(0, idx, white_i32_cuda);
            tot.index_add_(0, idx, torch::ones_like(white_i32_cuda, torch::kInt32));
        }

        // Newline after progress
        printf("\n");

        // 4) Final pruning by vote ratio
        const int min_visibility_count = 3;
        auto tot_safe = tot.to(torch::kFloat32).clamp_min(1.0f);
        auto ratio = pos.to(torch::kFloat32) / tot_safe;

        auto keep_mask = (tot >= min_visibility_count) & (ratio >= threshold);

        const int removed = (keep_mask == 0).sum().item<int>();
        model.filterByMask(keep_mask);

        std::cout << "[Trainer] prune_after_training (projection-only): removed "
                  << removed << " / " << N << " splats (thr=" << threshold
                  << ", min_vis=" << min_visibility_count << ")\n";
    }
} // namespace gs
