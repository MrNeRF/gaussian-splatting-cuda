// =============================================================================
// NEW FILE: include/core/training_progress.hpp
// =============================================================================
#pragma once

#include "external/indicators.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

class TrainingProgress {
private:
    std::unique_ptr<indicators::ProgressBar> progress_bar_;
    std::chrono::steady_clock::time_point start_time_;
    int total_iterations_;
    int update_frequency_;
    bool early_stopping_enabled_;
    
public:
    TrainingProgress(int total_iterations, int update_frequency = 100, bool enable_early_stopping = false) 
        : total_iterations_(total_iterations), 
          update_frequency_(update_frequency),
          early_stopping_enabled_(enable_early_stopping) {
        
        using namespace indicators;
        progress_bar_ = std::make_unique<ProgressBar>(ProgressBar{
            option::BarWidth{40},
            option::Start{"["},
            option::Fill{"█"},
            option::Lead{"▌"},
            option::Remainder{"░"},
            option::End{"]"},
            option::PrefixText{"Training "},
            option::PostfixText{"Initializing..."},
            option::ForegroundColor{Color::cyan},
            option::ShowPercentage{true},
            option::ShowElapsedTime{true},
            option::ShowRemainingTime{true},
            option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
        });
        
        start_time_ = std::chrono::steady_clock::now();
    }
    
    void update(int current_iteration, float loss, int splat_count, 
                float convergence_rate = 0.0f, bool is_densifying = false) {
        if (current_iteration % update_frequency_ != 0) return;
        
        float progress = static_cast<float>(current_iteration) / total_iterations_ * 100;
        progress_bar_->set_progress(progress);
        
        std::ostringstream postfix;
        postfix << "Loss: " << std::fixed << std::setprecision(4) << loss
                << " | Splats: " << splat_count;
        
        if (is_densifying) {
            postfix << " (+)";
        }
        
        if (early_stopping_enabled_ && convergence_rate > 0) {
            postfix << " | CR: " << std::fixed << std::setprecision(5) << convergence_rate;
        }
        
        progress_bar_->set_option(indicators::option::PostfixText{postfix.str()});
    }
    
    void complete() {
        if (!progress_bar_->is_completed()) {
            progress_bar_->set_progress(100);
            progress_bar_->mark_as_completed();
            std::cout << std::endl;
        }
    }
    
    void print_final_summary(int final_splats, float psnr, int actual_iterations = -1) {
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(end_time - start_time_).count();
        
        int iterations_used = (actual_iterations > 0) ? actual_iterations : total_iterations_;
        
        std::cout << std::endl
                  << "✓ Training completed in "
                  << std::fixed << std::setprecision(3) << elapsed << "s"
                  << " (avg " << std::fixed << std::setprecision(1) 
                  << iterations_used / elapsed << " iter/s)"
                  << std::endl
                  << "✓ Final splats: " << final_splats
                  << std::endl
                  << "✓ PSNR: " << std::fixed << std::setprecision(4) << psnr
                  << std::endl << std::endl;
    }
    
    void print_early_convergence(int converged_at) {
        complete();
        std::cout << "🎯 Converged after " << converged_at << " iterations!" << std::endl;
    }
    
    // Destructor ensures completion
    ~TrainingProgress() {
        complete();
    }
};

// =============================================================================
// UPDATED main.cpp - MINIMAL BOILERPLATE VERSION
// =============================================================================

#include "core/debug_utils.hpp"
#include "core/gaussian.hpp"
#include "core/loss_monitor.hpp"
#include "core/parameters.hpp"
#include "core/render_utils.hpp"
#include "core/scene.hpp"
#include "core/training_progress.hpp"  // NEW INCLUDE
#include "kernels/fused_ssim.cuh"
#include "kernels/loss_utils.cuh"
#include <args.hxx>
#include <c10/cuda/CUDACachingAllocator.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <torch/torch.h>

void Write_model_parameters_to_file(const gs::param::ModelParameters& params) {
    std::filesystem::path outputPath = params.output_path;
    std::filesystem::create_directories(outputPath);

    std::ofstream cfg_log_f(outputPath / "cfg_args");
    if (!cfg_log_f.is_open()) {
        std::cerr << "Failed to open file for writing!" << std::endl;
        return;
    }

    cfg_log_f << "Namespace(";
    cfg_log_f << "eval=" << (params.eval ? "True" : "False") << ", ";
    cfg_log_f << "images='" << params.images << "', ";
    cfg_log_f << "model_path='" << params.output_path.string() << "', ";
    cfg_log_f << "resolution=" << params.resolution << ", ";
    cfg_log_f << "sh_degree=" << params.sh_degree << ", ";
    cfg_log_f << "source_path='" << params.source_path.string() << "', ";
    cfg_log_f << "white_background=" << (params.white_background ? "True" : "False") << ")";
    cfg_log_f.close();

    std::cout << "Output folder: " << params.output_path.string() << std::endl;
}

std::vector<int> get_random_indices(int max_index) {
    std::vector<int> indices(max_index);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine());
    return indices;
}

int parse_cmd_line_args(const std::vector<std::string>& args,
                        gs::param::ModelParameters& modelParams,
                        gs::param::OptimizationParameters& optimParams) {
    if (args.empty()) {
        std::cerr << "No command line arguments provided!" << std::endl;
        return -1;
    }
    args::ArgumentParser parser("3D Gaussian Splatting CUDA Implementation\n",
                                "This program provides a lightning-fast CUDA implementation of the 3D Gaussian Splatting algorithm for real-time radiance field rendering.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<float> convergence_rate(parser, "convergence_rate", "Set convergence rate", {'c', "convergence_rate"});
    args::ValueFlag<int> resolution(parser, "resolution", "Set resolutino", {'r', "resolution"});
    args::Flag enable_cr_monitoring(parser, "enable_cr_monitoring", "Enable convergence rate monitoring", {"enable-cr-monitoring"});
    args::Flag force_overwrite_output_path(parser, "force", "Forces to overwrite output folder", {'f', "force"});
    args::Flag empty_gpu_memory(parser, "empty_gpu_cache", "Forces to reset GPU Cache. Should be lighter on VRAM", {"empty-gpu-cache"});
    args::ValueFlag<std::string> data_path(parser, "data_path", "Path to the training data", {'d', "data-path"});
    args::ValueFlag<std::string> output_path(parser, "output_path", "Path to the training output", {'o', "output-path"});
    args::ValueFlag<uint32_t> iterations(parser, "iterations", "Number of iterations to train the model", {'i', "iter"});
    args::CompletionFlag completion(parser, {"complete"});

    try {
        parser.Prog(args.front());
        parser.ParseArgs(std::vector<std::string>(args.begin() + 1, args.end()));
    } catch (const args::Completion& e) {
        std::cout << e.what();
        return 0;
    } catch (const args::Help&) {
        std::cout << parser;
        return -1;
    } catch (const args::ParseError& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return -1;
    }

    if (data_path) {
        modelParams.source_path = args::get(data_path);
    } else {
        std::cerr << "No data path provided!" << std::endl;
        return -1;
    }
    if (output_path) {
        modelParams.output_path = args::get(output_path);
    } else {
        std::filesystem::path executablePath = std::filesystem::canonical("/proc/self/exe");
        std::filesystem::path parentDir = executablePath.parent_path().parent_path();
        std::filesystem::path outputDir = parentDir / "output";
        try {
            bool isCreated = std::filesystem::create_directory(outputDir);
            if (!isCreated) {
                if (!force_overwrite_output_path) {
                    std::cerr << "Directory already exists! Not overwriting it" << std::endl;
                    return -1;
                } else {
                    std::filesystem::create_directory(outputDir);
                    std::filesystem::remove_all(outputDir);
                }
            }
        } catch (...) {
            std::cerr << "Failed to create output directory!" << std::endl;
            return -1;
        }
        modelParams.output_path = outputDir;
    }

    if (iterations) {
        optimParams.iterations = args::get(iterations);
    }
    optimParams.early_stopping = args::get(enable_cr_monitoring);
    if (optimParams.early_stopping && convergence_rate) {
        optimParams.convergence_threshold = args::get(convergence_rate);
    }

    if (resolution) {
        modelParams.resolution = args::get(resolution);
    }

    optimParams.empty_gpu_cache = args::get(empty_gpu_memory);
    return 0;
}

float psnr_metric(const torch::Tensor& rendered_img, const torch::Tensor& gt_img) {
    torch::Tensor squared_diff = (rendered_img - gt_img).pow(2);
    torch::Tensor mse_val = squared_diff.view({rendered_img.size(0), -1}).mean(1, true);
    return (20.f * torch::log10(1.0 / mse_val.sqrt())).mean().item<float>();
}

int main(int argc, char* argv[]) {
    std::vector<std::string> args;
    args.reserve(argc);

    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    
    auto modelParams = gs::param::ModelParameters();
    auto optimParams = gs::param::read_optim_params_from_json();
    if (parse_cmd_line_args(args, modelParams, optimParams) < 0) {
        return -1;
    };
    Write_model_parameters_to_file(modelParams);

    auto gaussians = GaussianModel(modelParams.sh_degree);
    auto scene = Scene(gaussians, modelParams);
    gaussians.Training_setup(optimParams);
    
    if (!torch::cuda::is_available()) {
        std::cout << "CUDA is not available! Training on CPU." << std::endl;
        exit(-1);
    }
    
    auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
    auto background = modelParams.white_background ? torch::tensor({1.f, 1.f, 1.f}) : torch::tensor({0.f, 0.f, 0.f}, pointType).to(torch::kCUDA);

    const int camera_count = scene.Get_camera_count();
    std::vector<int> indices;
    float loss_add = 0.f;

    LossMonitor loss_monitor(200);
    float avg_converging_rate = 0.f;
    float psnr_value = 0.f;

    // 🎯 MINIMAL BOILERPLATE - Just create the progress tracker!
    TrainingProgress progress(optimParams.iterations, 100, optimParams.early_stopping);

    // Training loop
    for (int iter = 1; iter < optimParams.iterations + 1; ++iter) {
        if (indices.empty()) {
            indices = get_random_indices(camera_count);
        }
        const int camera_index = indices.back();
        auto& cam = scene.Get_training_camera(camera_index);
        auto gt_image = cam.Get_original_image().to(torch::kCUDA, true);
        indices.pop_back();
        
        if (iter % 1000 == 0) {
            gaussians.One_up_sh_degree();
        }
        
        // Render
        auto [image, viewspace_point_tensor, visibility_filter, radii] = render(cam, gaussians, background);

        // Ensure both images are 4D tensors [N, C, H, W] for SSIM
        if (image.dim() == 3) {
            image = image.unsqueeze(0);
        }
        if (gt_image.dim() == 3) {
            gt_image = gt_image.unsqueeze(0);
        }

        if (image.sizes() != gt_image.sizes()) {
            std::cerr << "ERROR: Image size mismatch - rendered: " << image.sizes()
                      << ", ground truth: " << gt_image.sizes() << std::endl;
            exit(-1);
        }

        // Loss Computations
        auto image_for_l1 = image.dim() == 4 && image.size(0) == 1 ? image.squeeze(0) : image;
        auto gt_for_l1 = gt_image.dim() == 4 && gt_image.size(0) == 1 ? gt_image.squeeze(0) : gt_image;

        auto l1l = gaussian_splatting::l1_loss(image_for_l1, gt_for_l1);
        auto ssim_loss = fused_ssim(image, gt_image, "same", true);
        auto loss = (1.f - optimParams.lambda_dssim) * l1l + optimParams.lambda_dssim * (1.f - ssim_loss);

        if (optimParams.early_stopping) {
            avg_converging_rate = loss_monitor.Update(loss.item<float>());
        }
        loss_add += loss.item<float>();
        loss.backward();

        {
            torch::NoGradGuard no_grad;
            auto visible_max_radii = gaussians._max_radii2D.masked_select(visibility_filter);
            auto visible_radii = radii.masked_select(visibility_filter);
            auto max_radii = torch::max(visible_max_radii, visible_radii);
            gaussians._max_radii2D.masked_scatter_(visibility_filter, max_radii);

            // Check for densification to show (+) indicator
            bool is_densifying = (iter < optimParams.densify_until_iter && 
                                 iter > optimParams.densify_from_iter && 
                                 iter % optimParams.densification_interval == 0);

            // 🎯 SINGLE LINE PROGRESS UPDATE!
            progress.update(iter, loss.item<float>(), static_cast<int>(gaussians.Get_xyz().size(0)), 
                           avg_converging_rate, is_densifying);

            if (iter == optimParams.iterations) {
                gaussians.Save_ply(modelParams.output_path, iter, true);
                psnr_value = psnr_metric(image, gt_image);
                break;
            }

            if (iter % 7'000 == 0) {
                gaussians.Save_ply(modelParams.output_path, iter, false);
            }

            // Densification
            if (iter < optimParams.densify_until_iter) {
                gaussians.Add_densification_stats(viewspace_point_tensor, visibility_filter);
                if (is_densifying) {
                    gaussians.Densify_and_prune(optimParams.densify_grad_threshold, optimParams.min_opacity, scene.Get_cameras_extent());
                }

                if (iter % optimParams.opacity_reset_interval == 0 || (modelParams.white_background && iter == optimParams.densify_from_iter)) {
                    gaussians.Reset_opacity();
                }
            }

            if (iter >= optimParams.densify_until_iter && loss_monitor.IsConverging(optimParams.convergence_threshold)) {
                progress.print_early_convergence(iter);
                gaussians.Save_ply(modelParams.output_path, iter, true);
                psnr_value = psnr_metric(image, gt_image);
                break;
            }

            if (iter < optimParams.iterations) {
                gaussians._optimizer->step();
                gaussians._optimizer->zero_grad(true);
                gaussians.Update_learning_rate(iter);
            }

            if (optimParams.empty_gpu_cache && iter % 100) {
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
        }
    }

    // 🎯 SINGLE LINE FINAL SUMMARY!
    progress.print_final_summary(static_cast<int>(gaussians.Get_xyz().size(0)), psnr_value);

    return 0;
}
