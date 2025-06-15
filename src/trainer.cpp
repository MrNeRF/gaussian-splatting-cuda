#include "core/trainer.hpp"
#include "core/image_io.hpp"
#include "core/rasterizer.hpp"
#include "kernels/fused_ssim.cuh"
#include <chrono>
#include <iostream>
#include <numeric>
#include <torch/torch.h>

namespace gs {

    // Helper function to convert string render mode to enum
    inline RenderMode stringToRenderMode(const std::string& mode) {
        if (mode == "RGB")
            return RenderMode::RGB;
        else if (mode == "D")
            return RenderMode::D;
        else if (mode == "ED")
            return RenderMode::ED;
        else if (mode == "RGB_D")
            return RenderMode::RGB_D;
        else if (mode == "RGB_ED")
            return RenderMode::RGB_ED;
        else
            throw std::runtime_error("Invalid render mode: " + mode);
    }

    // Helper function to check if render mode includes depth
    inline bool renderModeHasDepth(RenderMode mode) {
        return mode != RenderMode::RGB;
    }

    // Helper function to check if render mode includes RGB
    inline bool renderModeHasRGB(RenderMode mode) {
        return mode == RenderMode::RGB ||
               mode == RenderMode::RGB_D ||
               mode == RenderMode::RGB_ED;
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

                    // Simple evaluation - just print without progress bar interaction
                    if (params_.optimization.enable_eval) {
                        for (size_t eval_step : params_.optimization.eval_steps) {
                            if (iter == static_cast<int>(eval_step)) {
                                std::cout << std::endl;
                                std::cout << "[Evaluation at step " << iter << "]" << std::endl;
                                auto metrics = evaluate(iter, params_.optimization.enable_save_eval_images);
                                std::cout << metrics.to_string() << std::endl;
                            }
                        }
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

        // Final evaluation and save final depth
        if (params_.optimization.enable_eval) {
            progress_->complete();
            std::cout << "\n[Final Evaluation]" << std::endl;
            auto final_metrics = evaluate(iter, params_.optimization.enable_save_eval_images);
            std::cout << final_metrics.to_string() << std::endl;
            metrics_reporter_->save_report();
        } else {
            progress_->complete();
        }

        strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/true);
        progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));
    }

    torch::Tensor Trainer::apply_depth_colormap(const torch::Tensor& depth_normalized) {
        // depth_normalized should be [H, W] with values in [0, 1]
        auto H = depth_normalized.size(0);
        auto W = depth_normalized.size(1);

        // Create RGB tensor
        auto colormap = torch::zeros({3, H, W}, torch::kFloat32);

        // Simple jet-like colormap
        auto depth_flat = depth_normalized.flatten();
        auto r = colormap[0].flatten();
        auto g = colormap[1].flatten();
        auto b = colormap[2].flatten();

        for (int i = 0; i < depth_flat.size(0); i++) {
            float val = depth_flat[i].item<float>();

            // Jet colormap approximation
            if (val < 0.25f) {
                r[i] = 0.0f;
                g[i] = 4.0f * val;
                b[i] = 1.0f;
            } else if (val < 0.5f) {
                r[i] = 0.0f;
                g[i] = 1.0f;
                b[i] = 1.0f - 4.0f * (val - 0.25f);
            } else if (val < 0.75f) {
                r[i] = 4.0f * (val - 0.5f);
                g[i] = 1.0f;
                b[i] = 0.0f;
            } else {
                r[i] = 1.0f;
                g[i] = 1.0f - 4.0f * (val - 0.75f);
                b[i] = 0.0f;
            }
        }

        return colormap;
    }

    metrics::EvalMetrics Trainer::evaluate(int iteration, bool save_images) {
        metrics::EvalMetrics result;
        result.num_gaussians = static_cast<int>(strategy_->get_model().size());
        result.iteration = iteration;

        auto val_dataloader = make_val_dataloader();

        std::vector<float> psnr_values, ssim_values, lpips_values;
        auto start_time = std::chrono::steady_clock::now();

        // Create directory for evaluation images
        std::filesystem::path eval_dir = params_.dataset.output_path /
                                         ("eval_step_" + std::to_string(iteration));
        std::filesystem::create_directories(eval_dir);

        // Convert string render mode to enum
        RenderMode render_mode = stringToRenderMode(params_.optimization.render_mode);
        const bool has_rgb = renderModeHasRGB(render_mode);
        const bool has_depth = renderModeHasDepth(render_mode);

        // Create subdirectory for depth maps only if we're saving depth
        std::filesystem::path depth_dir;
        if (has_depth) {
            depth_dir = eval_dir / "depth";
            std::filesystem::create_directories(depth_dir);
        }

        int image_idx = 0;
        for (auto& batch : *val_dataloader) {
            auto camera_with_image = batch[0].data;
            Camera* cam = camera_with_image.camera;
            torch::Tensor gt_image = std::move(camera_with_image.image);

            // Render with configured mode
            auto r_output = gs::rasterize(
                *cam,
                strategy_->get_model(),
                background_,
                1.0f,
                false,
                false,
                render_mode);

            // Only compute metrics if we have RGB output
            if (has_rgb) {
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

                // Save side-by-side RGB images
                if (save_images) {
                    save_image(eval_dir / (std::to_string(image_idx) + ".png"),
                               {gt_image.squeeze(0), r_output.image.squeeze(0)},
                               true, // horizontal
                               4);   // separator width
                }
            }

            // Only save depth if enabled and render mode includes depth
            if (has_depth && save_images) {
                if (r_output.depth.defined()) {
                    auto depth_vis = r_output.depth.clone().squeeze(0).to(torch::kCPU); // [H, W]

                    // Normalize depth
                    auto min_depth = depth_vis.min();
                    auto max_depth = depth_vis.max();
                    auto depth_normalized = (depth_vis - min_depth) / (max_depth - min_depth).clamp_min(1e-10);

                    // Apply colormap
                    auto depth_colormap = apply_depth_colormap(depth_normalized);

                    // Optionally save RGB + Depth side by side (only if we have RGB)
                    if (has_rgb) {
                        save_image(depth_dir / (std::to_string(image_idx) + "_rgb_depth.png"),
                                   {r_output.image.squeeze(0), depth_colormap},
                                   true, // horizontal
                                   4);   // separator width
                    } else {
                        // Save depth alone if no RGB
                        auto depth_gray_rgb = depth_normalized.unsqueeze(0).repeat({3, 1, 1});
                        save_image(depth_dir / (std::to_string(image_idx) + "_gray.png"), depth_gray_rgb);
                        save_image(depth_dir / (std::to_string(image_idx) + "_color.png"), depth_colormap);
                    }
                }
            }

            image_idx++;
        }

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<float>(end_time - start_time).count();

        // Compute averages only if we have RGB metrics
        if (has_rgb && !psnr_values.empty()) {
            result.psnr = std::accumulate(psnr_values.begin(), psnr_values.end(), 0.0f) / psnr_values.size();
            result.ssim = std::accumulate(ssim_values.begin(), ssim_values.end(), 0.0f) / ssim_values.size();
            result.lpips = std::accumulate(lpips_values.begin(), lpips_values.end(), 0.0f) / lpips_values.size();
        } else {
            // Set default values for depth-only modes
            result.psnr = 0.0f;
            result.ssim = 0.0f;
            result.lpips = 0.0f;
        }
        result.elapsed_time = elapsed / val_dataset_size_;

        // Add metrics to reporter
        metrics_reporter_->add_metrics(result);

        std::cout << "Saved " << image_idx << " evaluation images to: " << eval_dir << std::endl;
        if (has_depth) {
            std::cout << "Saved depth maps to: " << depth_dir << std::endl;
        }

        return result;
    }

} // namespace gs