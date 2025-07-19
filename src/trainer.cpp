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

    torch::Tensor Trainer::compute_photometric_loss(const RenderOutput& render_output,
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
        return loss;
    }

    /* torch::Tensor Model::outsideMaskOpacityPenalty(const torch::Tensor& current_photometric_loss,
                                                   const torch::Tensor& mask) {
        // --- 1. Pre-computation checks (same as before) ----------------------------
        bool can_apply_loss = mask.defined() && mask.numel() > 0 &&
                              final_pixel_transmittance.defined() && xys.defined() &&
                              radii.defined() && opacities.defined() &&
                              opacities.size(0) == radii.size(0) && xys.size(0) == radii.size(0);

        if (!can_apply_loss) {
            return current_photometric_loss;
        }

        // --- 2. Identify splats projected on screen (same as before) ---------------
        torch::Tensor visible_splats_mask = (radii > 0.0f).squeeze(-1);
        if (!visible_splats_mask.any().item<bool>()) {
            return current_photometric_loss;
        }

        // --- 3. Get opacity and pixel coordinates (same as before) -----------------
        torch::Tensor visible_opacities_logits = opacities.squeeze(-1).index({visible_splats_mask});
        torch::Tensor visible_splats_alpha = torch::sigmoid(visible_opacities_logits);

        int H = final_pixel_transmittance.size(0);
        int W = final_pixel_transmittance.size(1);
        torch::Tensor visible_xys = xys.index({visible_splats_mask});
        torch::Tensor x_coords = torch::round(visible_xys.select(1, 0)).to(torch::kLong).clamp(0, W - 1);
        torch::Tensor y_coords = torch::round(visible_xys.select(1, 1)).to(torch::kLong).clamp(0, H - 1);
        torch::Tensor linear_indices = y_coords * W + x_coords;

        // --- 4. Create per-splat masks for inside/outside regions ------------------
        torch::Tensor is_outside_mask = mask.flatten().index({linear_indices}).to(torch::kFloat32);
        torch::Tensor is_inside_mask = 1.0f - is_outside_mask;

        // --- 5. Calculate the two loss components ----------------------------------
        // Penalty for being opaque OUTSIDE the mask
        torch::Tensor outside_penalty = visible_splats_alpha * is_outside_mask;

        // Reward for being opaque INSIDE the mask (implemented as a penalty on transparency)
        torch::Tensor inside_reward_as_penalty = (1.0f - visible_splats_alpha) * is_inside_mask;

        // --- 6. Combine losses with their respective weights -----------------------
        torch::Tensor combined_loss_per_splat =
            (this->outsideMaskPenaltyWeight * 20.0f * outside_penalty) +
            (this->outsideMaskPenaltyWeight * 0.10f * inside_reward_as_penalty);

        torch::Tensor mean_combined_loss = combined_loss_per_splat.mean();

        // --- 7. Apply final loss (additively) --------------------------------------
        // Additive loss is generally more stable than multiplicative factors.
        return current_photometric_loss * (1 + mean_combined_loss);
    }*/

    torch::Tensor Trainer::compute_photometric_loss(const RenderOutput& render_output,
                                                    const torch::Tensor& gt_image,
                                                    const torch::Tensor& mask_image,
                                                    const float outOfMaskAlphaPenalty,
                                                    const SplatData& splatData,
                                                    const param::OptimizationParameters& opt_params) {

        if (!mask_image.defined() || mask_image.numel() == 0) {
            // fallback to the previous mode
            return compute_photometric_loss(render_output, gt_image, splatData, opt_params);
        }

        // Ensure images have same dimensions
        torch::Tensor rendered = render_output.image;
        torch::Tensor gt = gt_image;

        // Ensure both tensors are 4D (batch, height, width, channels)
        rendered = rendered.dim() == 3 ? rendered.unsqueeze(0) : rendered;
        gt = gt.dim() == 3 ? gt.unsqueeze(0) : gt;

        const int Height = rendered.size(2);
        const int Width = rendered.size(3);
        TORCH_CHECK(rendered.sizes() == gt.sizes(), "ERROR: size mismatch - rendered ", rendered.sizes(), " vs. ground truth ", gt.sizes());
        TORCH_CHECK((Height == mask_image.size(0) && Width == mask_image.size(1)),
                    "ERROR: size mismatch - rendered ", rendered.sizes(), " vs. mask ", mask_image.sizes());

        torch::Tensor inv = mask_image;
        if (inv.dim() == 2) {
            inv = inv.unsqueeze(0); // [1,H,W]
        }
        // inv: true = invalid
        // convert to float wo weights: 1 = invalid
        torch::Tensor invF = inv.to(torch::kFloat32); // [B,H,W]

        // Params
        const float invalidPixelWeight = 1.0f / 20.0f;
        const int ExpandRimRadius = 0;

        torch::Tensor W;
        if (ExpandRimRadius > 0) {
            const int ExpandRimWindow = 2 * ExpandRimRadius + 1;

            // invF: [B,H,W] o [H,W]
            torch::Tensor invF4;
            if (invF.dim() == 2) {
                invF4 = invF.unsqueeze(0).unsqueeze(0);
            } else if (invF.dim() == 3) {
                invF4 = invF.unsqueeze(1);
            } else {
                throw std::runtime_error("Unexpected mask dimensionality");
            }

            const int K = 2 * ExpandRimRadius + 1;
            // Exterior dilation
            auto dil4 = torch::nn::functional::max_pool2d(
                invF4,
                torch::nn::functional::MaxPool2dFuncOptions({K, K})
                    .stride(1)
                    .padding(ExpandRimRadius));
            // Back to the original shape
            torch::Tensor dil = (invF.dim() == 2)
                                    ? dil4.squeeze(0).squeeze(0)
                                    : dil4.squeeze(1);

            // Same for avg:
            auto avg4 = torch::nn::functional::avg_pool2d(
                invF4,
                torch::nn::functional::AvgPool2dFuncOptions({K, K})
                    .stride(1)
                    .padding(ExpandRimRadius));
            torch::Tensor soft = (invF.dim() == 2)
                                     ? avg4.squeeze(0).squeeze(0)
                                     : avg4.squeeze(1);

            /* Final map */
            W = torch::ones_like(invF);              // weight 1 as default
            W.masked_fill_(inv, invalidPixelWeight); // invalid interior

            torch::Tensor rim = (dil > 0.5f) & (~inv); // exterior border
            if (rim.any().item<bool>()) {
                torch::Tensor wRim = 1.f - soft * (1.f - invalidPixelWeight);
                W.masked_scatter_(rim, wRim.masked_select(rim));
            }
        } else {
            // no rim
            W = torch::where(inv,
                             torch::full_like(invF, invalidPixelWeight, torch::kFloat),
                             torch::ones_like(invF, torch::kFloat));
        }
        int totalPixels = (Height * Width);

        // Pixel-wise L1 map and weighted L1 loss
        auto l1_map = torch::abs(rendered - gt).mean(/*dim=*/1); // Resultado: [B, H, W]
        auto wSum = W.sum().clamp_min(1e-6f);
        auto l1_loss = (l1_map * W).sum() / totalPixels;

        // 2) pixel-wise SSIM map and weighted SSIM loss

#define weigthed_ssim
#ifdef weigthed_ssim
        // Compute the SSIM map. Using "valid" padding results in a smaller map.
        auto ssim_map_raw = fused_ssim_map(rendered, gt, "valid", /*train=*/true);

        // Process the raw SSIM map: average over channels and clamp values to [0, 1].
        torch::Tensor ssim_map;
        if (ssim_map_raw.dim() == 4) {
            ssim_map = ssim_map_raw.mean(1); // Average channel dimension -> [B, H, W]
        } else {
            ssim_map = ssim_map_raw;
        }
        ssim_map = ssim_map.clamp(0.0f, 1.0f);

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
#else
        auto ssim_loss = 1.f - fused_ssim(rendered, gt, "valid", /*train=*/true);
#endif

        // 3) combined loss
        auto loss = (1.0f - opt_params.lambda_dssim) * l1_loss + opt_params.lambda_dssim * ssim_loss;
        return loss;
    }

    torch::Tensor Trainer::compute_scale_reg_loss(const SplatData& splatData,
                                                  const param::OptimizationParameters& opt_params) {
        if (opt_params.scale_reg > 0.0f) {
            auto scale_l1 = splatData.get_scaling().mean();
            return opt_params.scale_reg * scale_l1;
        }
        return torch::zeros({1}, torch::kFloat32).requires_grad_();
    }

    torch::Tensor Trainer::compute_opacity_reg_loss(const SplatData& splatData,
                                                    const param::OptimizationParameters& opt_params) {
        if (opt_params.opacity_reg > 0.0f) {
            auto opacity_l1 = splatData.get_opacity().mean();
            return opt_params.opacity_reg * opacity_l1;
        }
        return torch::zeros({1}, torch::kFloat32).requires_grad_();
    }

    torch::Tensor Trainer::compute_bilateral_grid_tv_loss(const std::unique_ptr<gs::BilateralGrid>& bilateral_grid,
                                                          const param::OptimizationParameters& opt_params) {
        if (opt_params.use_bilateral_grid) {
            return opt_params.tv_loss_weight * bilateral_grid->tv_loss();
        }
        return torch::zeros({1}, torch::kFloat32).requires_grad_();
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

    bool Trainer::train_step(int iter, Camera* cam, torch::Tensor gt_image, torch::Tensor mask_image, RenderMode render_mode, float out_of_mask_penalty) {
        if (cam->radial_distortion().numel() != 0 ||
            cam->tangential_distortion().numel() != 0) {
            throw std::runtime_error("Training on cameras with distortion is not supported yet.");
        }
        if (cam->camera_model_type() != gsplat::CameraModelType::PINHOLE) {
            throw std::runtime_error("Training on cameras with non-pinhole model is not supported yet.");
        }
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
        torch::Tensor loss;
        if (!mask_image.defined() || params_.optimization.iterations * 0.1 > iter)
            loss = compute_photometric_loss(r_output, gt_image, strategy_->get_model(), params_.optimization);
        else
            loss = compute_photometric_loss(r_output, gt_image, mask_image, out_of_mask_penalty, strategy_->get_model(), params_.optimization);

        loss.backward();
        float loss_value = loss.item<float>();

        loss = compute_scale_reg_loss(strategy_->get_model(), params_.optimization);
        loss.backward();
        loss_value += loss.item<float>();

        loss = compute_opacity_reg_loss(strategy_->get_model(), params_.optimization);
        loss.backward();
        loss_value += loss.item<float>();

        loss = compute_bilateral_grid_tv_loss(bilateral_grid_, params_.optimization);
        loss.backward();
        loss_value += loss.item<float>();

        current_loss_ = loss_value;

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
            if (!params_.optimization.skip_intermediate_saving) {
                for (size_t save_step : params_.optimization.save_steps) {
                    if (iter == static_cast<int>(save_step) && iter != params_.optimization.iterations) {
                        const bool join_threads = (iter == params_.optimization.save_steps.back());
                        strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/join_threads);
                    }
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

        progress_->update(iter, current_loss_,
                          static_cast<int>(strategy_->get_model().size()),
                          strategy_->is_refining(iter));

        if (viewer_) {
            if (viewer_->info_) {
                auto& info = viewer_->info_;
                std::lock_guard<std::mutex> lock(viewer_->info_->mtx);
                info->updateProgress(iter, params_.optimization.iterations);
                info->updateNumSplats(static_cast<size_t>(strategy_->get_model().size()));
                info->updateLoss(current_loss_);
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
                torch::Tensor gt_image = std::move(camera_with_image.imageTensor);
                if (!params_.optimization.use_attention_mask) {
                    should_continue = train_step(iter, cam, gt_image, torch::Tensor(), render_mode, 0.0f);
                } else {
                    torch::Tensor mask_image = std::move(camera_with_image.maskTensor);
                    float out_of_mask_penalty = epoch * 4 < epochs_needed ? 1.0f : 0.0f;
                    should_continue = train_step(iter, cam, gt_image, mask_image, render_mode, out_of_mask_penalty);
                }

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