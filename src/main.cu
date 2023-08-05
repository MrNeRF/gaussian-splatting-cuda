#include "gaussian.cuh"
#include "loss_utils.cuh"
#include "parameters.cuh"
#include "read_utils.cuh"
#include "render_utils.cuh"
#include "scene.cuh"
#include <iostream>
#include <random>
#include <torch/torch.h>

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cout << "Usage: ./readPly <ply file>" << std::endl;
        return 1;
    }
    // TODO: read parameters from JSON file or command line
    auto modelParams = ModelParameters();
    modelParams.source_path = argv[1];
    const auto optimParams = OptimizationParameters();
    const auto pipelineParams = PipelineParameters();
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

    const int camera_count = scene.Get_camera_count();
    // Initialize random engine
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, camera_count - 1);
    // training loop
    for (int iter = 1; iter < optimParams.iterations; ++iter) {
        if (iter % 1000 == 0) {
            gaussians.One_up_sh_degree();
        }

        // Render
        const int random_index = dis(gen);
        auto& cam = scene.Get_training_camera(random_index);
        auto [image, viewspace_point_tensor, visibility_filter, radii] = render(cam, gaussians, pipelineParams, background);

        // Loss Computations
        auto gt_image = cam.Get_original_image().to(torch::kCUDA);
        auto l1l = gaussian_splatting::l1_loss(image, gt_image);
        auto loss = (1.0 - optimParams.lambda_dssim) * l1l + optimParams.lambda_dssim * (1.0 - gaussian_splatting::ssim(image, gt_image));
        std::cout << "Iteration: " << iter << " Loss: " << loss.item<float>() << std::endl;
        loss.backward();

        if (!gaussians._opacity.grad().defined()) {
            std::cout << "Opacity gradient is not defined! Iter: " << iter << std::endl;
        }
        {
            torch::NoGradGuard no_grad;
            // Keep track of max radii in image-space for pruning
            auto visible_max_radii = gaussians._max_radii2D.masked_select(visibility_filter);
            auto visible_radii = radii.masked_select(visibility_filter);

            auto max_radii = torch::max(visible_max_radii, visible_radii);
            gaussians._max_radii2D.masked_scatter_(visibility_filter, max_radii);

            // TODO: support saving
            //          if (iteration in saving_iterations):
            //             print("\n[ITER {}] Saving Gaussians".format(iteration))
            //             scene.save(iteration)

            // Densification
            if (iter < optimParams.densify_until_iter) {
                gaussians.Add_densification_stats(viewspace_point_tensor, visibility_filter);
                if (iter > optimParams.densify_from_iter && iter % optimParams.densification_interval == 0) {
                    // @TODO: Not sure about type
                    float size_threshold = iter > optimParams.opacity_reset_interval ? 20.f : -1.f;
                    gaussians.Densify_and_prune(optimParams.densify_grad_threshold, 0.005, scene.Get_cameras_extent(), size_threshold);
                }

                if (iter % optimParams.opacity_reset_interval == 0 || (modelParams.white_background && iter == optimParams.densify_from_iter)) {
                    std::cout << "iteration " << iter << " resetting opacity" << std::endl;
                    gaussians.Reset_opacity();
                }
            }

            //  Optimizer step
            if (iter < optimParams.iterations) {
                gaussians._optimizer->step();
                gaussians._optimizer->zero_grad(true);
                // @TODO: Not sure about type
                gaussians.Update_learning_rate(iter);
            }
        }
    }
    return 0;
}