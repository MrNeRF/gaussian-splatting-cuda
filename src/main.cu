#include "args_parser.cuh"
#include "gaussian.cuh"
#include "loss_monitor.cuh"
#include "loss_utils.cuh"
#include "parameters.cuh"
#include "render_utils.cuh"
#include "progess_stats.cuh"
#include "scene.cuh"
#include <c10/cuda/CUDACachingAllocator.h>
#include <iostream>
#include <random>
#include <torch/torch.h>

std::vector<int> get_random_indices(int max_index) {
    std::vector<int> indices(max_index);
    std::iota(indices.begin(), indices.end(), 0);
    // Shuffle the vector
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine());
    std::reverse(indices.begin(), indices.end());
    return indices;
}


float psnr_metric(const torch::Tensor& rendered_img, const torch::Tensor& gt_img) {

    torch::Tensor squared_diff = (rendered_img - gt_img).pow(2);
    torch::Tensor mse_val = squared_diff.view({rendered_img.size(0), -1}).mean(1, true);
    return (20.f * torch::log10(1.0 / mse_val.sqrt())).mean().item<float>();
}

int main(int argc, const char* argv[]) {
    // TODO: read parameters from JSON file or command line
    auto modelParams = gs::param::ModelParameters();
    auto optimParams = gs::param::read_optim_params_from_json();
    if (ArgsParser::Parse(argc, argv, modelParams, optimParams) < 0) {
        return -1;
    };

    ArgsParser::Dump(modelParams);

    auto gaussians = GaussianModel(modelParams.sh_degree);
    auto scene = Scene(gaussians, modelParams);
    gaussians.Training_setup(optimParams);
    if (!torch::cuda::is_available()) {
        // At the moment, I want to make sure that my GPU is utilized.
        std::cout << "CUDA is not available! Training on CPU." << std::endl;
        exit(-1);
    }
    auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
    auto background = modelParams.white_background ? torch::tensor({1.f, 1.f, 1.f}) : torch::tensor({0.f, 0.f, 0.f}, pointType).to(torch::kCUDA);

    const int window_size = 11;
    const int channel = 3;
    const auto conv_window = gaussian_splatting::create_window(window_size, channel).to(torch::kFloat32).to(torch::kCUDA, true);
    const int camera_count = scene.Get_camera_count();

    std::vector<int> indices;
    LossMonitor loss_monitor(200);
    float avg_converging_rate = 0.f;

    float psnr_value = 0.f;
    auto progress_stats = ProgressStats();
    for (int iter = 1; iter < optimParams.iterations + 1; ++iter) {
        if (indices.empty()) {
            indices = get_random_indices(camera_count);
        }
        const int camera_index = indices.back();
        auto& cam = scene.Get_training_camera(camera_index);
        auto gt_image = cam.Get_original_image().to(torch::kCUDA, true);
        indices.pop_back(); // remove last element to iterate over all cameras randomly
        if (iter % 1000 == 0) {
            gaussians.One_up_sh_degree();
        }
        // Render
        auto [image, viewspace_point_tensor, visibility_filter, radii] = render(cam, gaussians, background);

        // Loss Computations
        auto l1l = gaussian_splatting::l1_loss(image, gt_image);
        auto ssim_loss = gaussian_splatting::ssim(image, gt_image, conv_window, window_size, channel);
        auto loss = (1.f - optimParams.lambda_dssim) * l1l + optimParams.lambda_dssim * (1.f - ssim_loss);

        // Update status line
        if (iter % 10 == 0) {
            progress_stats.print(iter, (int)gaussians.Get_xyz().size(0), loss.item<float>());
        }

        loss.backward();

        {
            torch::NoGradGuard no_grad;
            auto visible_max_radii = gaussians._max_radii2D.masked_select(visibility_filter);
            auto visible_radii = radii.masked_select(visibility_filter);
            auto max_radii = torch::max(visible_max_radii, visible_radii);
            gaussians._max_radii2D.masked_scatter_(visibility_filter, max_radii);

            if (iter == optimParams.iterations) {
                std::cout << std::endl;
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
                if (iter > optimParams.densify_from_iter && iter % optimParams.densification_interval == 0) {
                    // @TODO: Not sure about type
                    float size_threshold = iter > optimParams.opacity_reset_interval ? 20.f : -1.f;
                    gaussians.Densify_and_prune(optimParams.densify_grad_threshold, optimParams.min_opacity, scene.Get_cameras_extent(), size_threshold);
                }

                if (iter % optimParams.opacity_reset_interval == 0 || (modelParams.white_background && iter == optimParams.densify_from_iter)) {
                    gaussians.Reset_opacity();
                }
            }

            if (iter >= optimParams.densify_until_iter && loss_monitor.IsConverging(optimParams.convergence_threshold)) {
                std::cout << "Converged after " << iter << " iterations!" << std::endl;
                gaussians.Save_ply(modelParams.output_path, iter, true);
                break;
            }

            //  Optimizer step
            if (iter < optimParams.iterations) {
                gaussians._optimizer->step();
                gaussians._optimizer->zero_grad(true);
                // @TODO: Not sure about type
                gaussians.Update_learning_rate(iter);
            }

            if (optimParams.empty_gpu_cache && iter % 100) {
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
        }
    }

    progress_stats.stop(optimParams.iterations, gaussians.Get_xyz().size(0), psnr_value);

    return 0;
}
