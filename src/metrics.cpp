#include "core/metrics.hpp"
#include "core/image_io.hpp"
#include "core/splat_data.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

namespace gs {
    namespace metrics {

        // 1D Gaussian kernel
        torch::Tensor gaussian(int window_size, float sigma) {
            torch::Tensor gauss = torch::empty(window_size);
            for (int x = 0; x < window_size; ++x) {
                gauss[x] = std::exp(-(std::pow(std::floor(static_cast<float>(x - window_size) / 2.f), 2)) / (2.f * sigma * sigma));
            }
            return gauss / gauss.sum();
        }

        torch::Tensor create_window(int window_size, int channel) {
            auto _1D_window = gaussian(window_size, 1.5).unsqueeze(1);
            auto _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0);
            return _2D_window.expand({channel, 1, window_size, window_size}).contiguous();
        }

        // PSNR Implementation
        float PSNR::compute(const torch::Tensor& pred, const torch::Tensor& target) {
            TORCH_CHECK(pred.sizes() == target.sizes(),
                        "Prediction and target must have the same shape");

            // Make tensors contiguous before operations
            auto pred_cont = pred.contiguous();
            auto target_cont = target.contiguous();

            // Compute MSE
            torch::Tensor squared_diff = (pred_cont - target_cont).pow(2);

            // Use reshape instead of view to handle non-contiguous tensors
            torch::Tensor mse_val = squared_diff.reshape({pred.size(0), -1}).mean(1, true);

            // Avoid log(0)
            mse_val = torch::clamp_min(mse_val, 1e-10);

            // PSNR = 20 * log10(data_range / sqrt(MSE))
            return (20.f * torch::log10(data_range_ / mse_val.sqrt())).mean().item<float>();
        }

        // SSIM Implementation
        SSIM::SSIM(int window_size, int channel)
            : window_size_(window_size),
              channel_(channel) {
            window_ = create_window(window_size, channel).to(torch::kFloat32);
        }

        float SSIM::compute(const torch::Tensor& pred, const torch::Tensor& target) {
            TORCH_CHECK(pred.dim() == 4, "Expected 4D tensor [B, C, H, W]");
            TORCH_CHECK(pred.sizes() == target.sizes(),
                        "Prediction and target must have the same shape");

            // Ensure window is on the same device as input
            if (window_.device() != pred.device()) {
                window_ = window_.to(pred.device());
            }

            const int pad = window_size_ / 2;

            // Compute local means
            auto mu1 = torch::nn::functional::conv2d(pred, window_,
                                                     torch::nn::functional::Conv2dFuncOptions()
                                                         .padding(pad)
                                                         .groups(channel_));
            auto mu2 = torch::nn::functional::conv2d(target, window_,
                                                     torch::nn::functional::Conv2dFuncOptions()
                                                         .padding(pad)
                                                         .groups(channel_));

            auto mu1_sq = mu1.pow(2);
            auto mu2_sq = mu2.pow(2);
            auto mu1_mu2 = mu1 * mu2;

            // Compute local variances and covariance
            auto sigma1_sq = torch::nn::functional::conv2d(pred * pred, window_,
                                                           torch::nn::functional::Conv2dFuncOptions()
                                                               .padding(pad)
                                                               .groups(channel_)) -
                             mu1_sq;
            auto sigma2_sq = torch::nn::functional::conv2d(target * target, window_,
                                                           torch::nn::functional::Conv2dFuncOptions()
                                                               .padding(pad)
                                                               .groups(channel_)) -
                             mu2_sq;
            auto sigma12 = torch::nn::functional::conv2d(pred * target, window_,
                                                         torch::nn::functional::Conv2dFuncOptions()
                                                             .padding(pad)
                                                             .groups(channel_)) -
                           mu1_mu2;

            // SSIM formula
            auto ssim_map = ((2.f * mu1_mu2 + C1) * (2.f * sigma12 + C2)) /
                            ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

            return ssim_map.mean().item<float>();
        }

        // LPIPS Implementation
        LPIPS::LPIPS(const std::string& model_path) {
            if (!model_path.empty()) {
                load_model(model_path);
            } else {
                // Try default paths
                std::vector<std::string> default_paths = {
                    "weights/lpips_vgg.pt",
                    "../weights/lpips_vgg.pt",
                    "../../weights/lpips_vgg.pt",
                    std::string(std::getenv("HOME") ? std::getenv("HOME") : "") + "/.cache/gaussian_splatting/lpips_vgg.pt"};

                for (const auto& path : default_paths) {
                    if (std::filesystem::exists(path)) {
                        load_model(path);
                        break;
                    }
                }
            }

            if (!model_loaded_) {
                throw std::runtime_error(
                    "LPIPS model not found! \n"
                    "Searched paths: weights/lpips_vgg.pt, ../weights/lpips_vgg.pt");
            }
        }

        void LPIPS::load_model(const std::string& model_path) {
            try {
                model_ = torch::jit::load(model_path);
                model_.eval();
                model_.to(torch::kCUDA);
                model_loaded_ = true;
                std::cout << "LPIPS model loaded from: " << model_path << std::endl;
            } catch (const c10::Error& e) {
                throw std::runtime_error(
                    "Failed to load LPIPS model from " + model_path + ": " + e.what());
            }
        }

        float LPIPS::compute(const torch::Tensor& pred, const torch::Tensor& target) {
            TORCH_CHECK(pred.dim() == 4, "Expected 4D tensor [B, C, H, W]");
            TORCH_CHECK(pred.sizes() == target.sizes(),
                        "Prediction and target must have the same shape");
            TORCH_CHECK(model_loaded_, "LPIPS model not loaded!");

            torch::NoGradGuard no_grad;

            // LPIPS expects inputs in range [-1, 1], but our inputs are in [0, 1]
            // Convert from [0, 1] to [-1, 1]
            auto pred_normalized = 2.0f * pred - 1.0f;
            auto target_normalized = 2.0f * target - 1.0f;

            // Ensure inputs are on CUDA and contiguous
            pred_normalized = pred_normalized.to(torch::kCUDA).contiguous();
            target_normalized = target_normalized.to(torch::kCUDA).contiguous();

            // Forward pass through LPIPS model
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(pred_normalized);
            inputs.push_back(target_normalized);

            auto output = model_.forward(inputs).toTensor();

            // LPIPS returns a single value per batch item
            return output.mean().item<float>();
        }

        // MetricsReporter Implementation
        MetricsReporter::MetricsReporter(const std::filesystem::path& output_dir)
            : output_dir_(output_dir) {
            csv_path_ = output_dir_ / "metrics.csv";
            txt_path_ = output_dir_ / "metrics_report.txt";

            // Create CSV header if file doesn't exist
            if (!std::filesystem::exists(csv_path_)) {
                std::ofstream csv_file(csv_path_);
                if (csv_file.is_open()) {
                    csv_file << EvalMetrics{}.to_csv_header() << std::endl;
                    csv_file.close();
                }
            }
        }

        void MetricsReporter::add_metrics(const EvalMetrics& metrics) {
            all_metrics_.push_back(metrics);

            // Append to CSV immediately
            std::ofstream csv_file(csv_path_, std::ios::app);
            if (csv_file.is_open()) {
                csv_file << metrics.to_csv_row() << std::endl;
                csv_file.close();
            }
        }

        void MetricsReporter::save_report() {
            std::ofstream report_file(txt_path_);
            if (!report_file.is_open()) {
                std::cerr << "Failed to open report file: " << txt_path_ << std::endl;
                return;
            }

            // Write header
            report_file << "==============================================\n";
            report_file << "3D Gaussian Splatting Evaluation Report\n";
            report_file << "==============================================\n";
            report_file << "Output Directory: " << output_dir_ << "\n";

            // Get current time
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            report_file << "Generated: " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "\n\n";

            // Summary statistics
            if (!all_metrics_.empty()) {
                report_file << "Summary Statistics:\n";
                report_file << "------------------\n";

                // Find best metrics
                auto best_psnr = std::max_element(all_metrics_.begin(), all_metrics_.end(),
                                                  [](const EvalMetrics& a, const EvalMetrics& b) { return a.psnr < b.psnr; });
                auto best_ssim = std::max_element(all_metrics_.begin(), all_metrics_.end(),
                                                  [](const EvalMetrics& a, const EvalMetrics& b) { return a.ssim < b.ssim; });
                auto best_lpips = std::min_element(all_metrics_.begin(), all_metrics_.end(),
                                                   [](const EvalMetrics& a, const EvalMetrics& b) { return a.lpips > b.lpips; });

                report_file << std::fixed << std::setprecision(4);
                report_file << "Best PSNR:  " << best_psnr->psnr << " (at iteration " << best_psnr->iteration << ")\n";
                report_file << "Best SSIM:  " << best_ssim->ssim << " (at iteration " << best_ssim->iteration << ")\n";
                report_file << "Best LPIPS: " << best_lpips->lpips << " (at iteration " << best_lpips->iteration << ")\n";

                // Final metrics
                const auto& final = all_metrics_.back();
                report_file << "\nFinal Metrics (iteration " << final.iteration << "):\n";
                report_file << "PSNR:  " << final.psnr << "\n";
                report_file << "SSIM:  " << final.ssim << "\n";
                report_file << "LPIPS: " << final.lpips << "\n";
                report_file << "Time per image: " << final.elapsed_time << " seconds\n";
                report_file << "Number of Gaussians: " << final.num_gaussians << "\n";
            }

            // Detailed results
            report_file << "\nDetailed Results:\n";
            report_file << "-----------------\n";
            report_file << std::setw(10) << "Iteration"
                        << std::setw(10) << "PSNR"
                        << std::setw(10) << "SSIM"
                        << std::setw(10) << "LPIPS"
                        << std::setw(15) << "Time(s/img)"
                        << std::setw(15) << "#Gaussians"
                        << "\n";
            report_file << std::string(75, '-') << "\n";

            for (const auto& m : all_metrics_) {
                report_file << std::setw(10) << m.iteration
                            << std::setw(10) << std::fixed << std::setprecision(4) << m.psnr
                            << std::setw(10) << m.ssim
                            << std::setw(10) << m.lpips
                            << std::setw(15) << m.elapsed_time
                            << std::setw(15) << m.num_gaussians << "\n";
            }

            report_file.close();
            std::cout << "Evaluation report saved to: " << txt_path_ << std::endl;
            std::cout << "Metrics CSV saved to: " << csv_path_ << std::endl;
        }

        // MetricsEvaluator Implementation
        MetricsEvaluator::MetricsEvaluator(const param::TrainingParameters& params)
            : enabled_(params.optimization.enable_eval),
              save_images_(params.optimization.enable_save_eval_images),
              render_mode_str_(params.optimization.render_mode),
              output_path_(params.dataset.output_path),
              eval_steps_(params.optimization.eval_steps) {

            if (!enabled_) {
                return;
            }

            // Convert render mode string to enum
            render_mode_ = stringToRenderMode(render_mode_str_);

            // Initialize metrics
            psnr_metric_ = std::make_unique<PSNR>(1.0f);
            ssim_metric_ = std::make_unique<SSIM>(11, 3);

            // Find LPIPS model
            std::filesystem::path lpips_path = params.dataset.output_path.parent_path() / "weights" / "lpips_vgg.pt";
            if (!std::filesystem::exists(lpips_path)) {
                lpips_path = "weights/lpips_vgg.pt";
            }
            lpips_metric_ = std::make_unique<LPIPS>(lpips_path.string());

            // Initialize reporter
            reporter_ = std::make_unique<MetricsReporter>(params.dataset.output_path);

            std::cout << "MetricsEvaluator initialized with render mode: " << render_mode_str_ << std::endl;
        }

        bool MetricsEvaluator::should_evaluate(int iteration) const {
            if (!enabled_) return false;

            return std::find(eval_steps_.begin(), eval_steps_.end(), iteration) != eval_steps_.end();
        }

        bool MetricsEvaluator::has_rgb() const {
            return render_mode_ == RenderMode::RGB ||
                   render_mode_ == RenderMode::RGB_D ||
                   render_mode_ == RenderMode::RGB_ED;
        }

        bool MetricsEvaluator::has_depth() const {
            return render_mode_ != RenderMode::RGB;
        }

        torch::Tensor MetricsEvaluator::apply_depth_colormap(const torch::Tensor& depth_normalized) {
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

        auto MetricsEvaluator::make_dataloader(std::shared_ptr<CameraDataset> dataset, int workers) const {
            return create_dataloader_from_dataset(dataset, workers);
        }

        EvalMetrics MetricsEvaluator::evaluate(int iteration,
                                               const SplatData& splatData,
                                               std::shared_ptr<CameraDataset> val_dataset,
                                               torch::Tensor& background) {
            if (!enabled_) {
                throw std::runtime_error("Evaluation is not enabled");
            }

            EvalMetrics result;
            result.num_gaussians = static_cast<int>(splatData.size());

            result.iteration = iteration;

            auto val_dataloader = make_dataloader(val_dataset);

            std::vector<float> psnr_values, ssim_values, lpips_values;
            auto start_time = std::chrono::steady_clock::now();

            // Create directory for evaluation images
            std::filesystem::path eval_dir = output_path_ /
                                             ("eval_step_" + std::to_string(iteration));
            if (save_images_) {
                std::filesystem::create_directories(eval_dir);
            }

            // Create subdirectory for depth maps only if we're saving depth
            std::filesystem::path depth_dir;
            if (has_depth() && save_images_) {
                depth_dir = eval_dir / "depth";
                std::filesystem::create_directories(depth_dir);
            }

            int image_idx = 0;
            size_t val_dataset_size = val_dataset->size().value();

            for (auto& batch : *val_dataloader) {
                auto camera_with_image = batch[0].data;
                Camera* cam = camera_with_image.camera;
                torch::Tensor gt_image = std::move(camera_with_image.image);

                // Render with configured mode
                auto r_output = gs::rasterize(
                    *cam,
                    splatData,
                    background,
                    1.0f,
                    false,
                    false,
                    render_mode_);

                // Only compute metrics if we have RGB output
                if (has_rgb()) {
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
                    if (save_images_) {
                        save_image(eval_dir / (std::to_string(image_idx) + ".png"),
                                   {gt_image.squeeze(0), r_output.image.squeeze(0)},
                                   true,  // horizontal
                                   4);    // separator width
                    }
                }

                // Only save depth if enabled and render mode includes depth
                if (has_depth() && save_images_) {
                    if (r_output.depth.defined()) {
                        auto depth_vis = r_output.depth.clone().squeeze(0).to(torch::kCPU); // [H, W]

                        // Normalize depth
                        auto min_depth = depth_vis.min();
                        auto max_depth = depth_vis.max();
                        auto depth_normalized = (depth_vis - min_depth) / (max_depth - min_depth).clamp_min(1e-10);

                        // Apply colormap
                        auto depth_colormap = apply_depth_colormap(depth_normalized);

                        // Optionally save RGB + Depth side by side (only if we have RGB)
                        if (has_rgb()) {
                            save_image(depth_dir / (std::to_string(image_idx) + "_rgb_depth.png"),
                                       {r_output.image.squeeze(0), depth_colormap},
                                       true,  // horizontal
                                       4);    // separator width
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
            if (has_rgb() && !psnr_values.empty()) {
                result.psnr = std::accumulate(psnr_values.begin(), psnr_values.end(), 0.0f) / psnr_values.size();
                result.ssim = std::accumulate(ssim_values.begin(), ssim_values.end(), 0.0f) / ssim_values.size();
                result.lpips = std::accumulate(lpips_values.begin(), lpips_values.end(), 0.0f) / lpips_values.size();
            } else {
                // Set default values for depth-only modes
                result.psnr = 0.0f;
                result.ssim = 0.0f;
                result.lpips = 0.0f;
            }
            result.elapsed_time = elapsed / val_dataset_size;

            // Add metrics to reporter
            reporter_->add_metrics(result);

            if (save_images_) {
                std::cout << "Saved " << image_idx << " evaluation images to: " << eval_dir << std::endl;
                if (has_depth()) {
                    std::cout << "Saved depth maps to: " << depth_dir << std::endl;
                }
            }

            return result;
        }

    } // namespace metrics
} // namespace gs