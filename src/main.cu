#include "debug_utils.cuh"
#include "gaussian.cuh"
#include "loss_utils.cuh"
#include "parameters.cuh"
#include "render_utils.cuh"
#include "scene.cuh"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <torch/torch.h>

std::filesystem::path createOutputDirectoryInParent() {
    std::filesystem::path executablePath = std::filesystem::canonical("/proc/self/exe");
    std::filesystem::path parentDir = executablePath.parent_path().parent_path();
    std::filesystem::path outputDir = parentDir / "output";
    std::filesystem::create_directory(outputDir);
    return outputDir;
}

void Write_model_parameters_to_file(const ModelParameters& params) {
    std::filesystem::path outputPath = params.model_path;
    std::filesystem::create_directories(outputPath); // Make sure the directory exists

    std::ofstream cfg_log_f(outputPath / "cfg_args");
    if (!cfg_log_f.is_open()) {
        std::cerr << "Failed to open file for writing!" << std::endl;
        return;
    }

    // Write the parameters in the desired format
    cfg_log_f << "Namespace(";
    cfg_log_f << "eval=" << (params.eval ? "True" : "False") << ", ";
    cfg_log_f << "images='" << params.images << "', ";
    cfg_log_f << "model_path='" << params.model_path.string() << "', ";
    cfg_log_f << "resolution=" << params.resolution << ", ";
    cfg_log_f << "sh_degree=" << params.sh_degree << ", ";
    cfg_log_f << "source_path='" << params.source_path.string() << "', ";
    cfg_log_f << "white_background=" << (params.white_background ? "True" : "False") << ")";
    cfg_log_f.close();

    std::cout << "Output folder: " << params.model_path.string() << std::endl;
}

std::vector<int> get_random_indices(int max_index) {
    std::vector<int> indices(max_index);
    std::iota(indices.begin(), indices.end(), 0);
    // Shuffle the vector
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine());
    return indices;
}

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cout << "Usage: ./readPly <ply file>" << std::endl;
        return 1;
    }
    // TODO: read parameters from JSON file or command line
    auto modelParams = ModelParameters();
    modelParams.source_path = argv[1];
    modelParams.model_path = createOutputDirectoryInParent();
    Write_model_parameters_to_file(modelParams);

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
    std::vector<int> indices;
    for (int iter = 1; iter < optimParams.iterations + 1; ++iter) {
        if (iter % 1000 == 0) {
            gaussians.One_up_sh_degree();
        }

        if (indices.empty()) {
            indices = get_random_indices(camera_count);
        }
        const int camera_index = indices.back();
        indices.pop_back(); // remove last element to iterate over all cameras randomly
        auto& cam = scene.Get_training_camera(camera_index);
        // Render
        auto [image, viewspace_point_tensor, visibility_filter, radii] = render(cam, gaussians, pipelineParams, background);

        // Loss Computations
        auto gt_image = cam.Get_original_image().to(torch::kCUDA);
        auto l1l = gaussian_splatting::l1_loss(image, gt_image);
        auto loss = (1.f - optimParams.lambda_dssim) * l1l + optimParams.lambda_dssim * (1.f - gaussian_splatting::ssim(image, gt_image));
        std::cout << "Iteration: " << iter << " Loss: " << loss.item<float>() << " gaussian splats: " << gaussians.Get_xyz().size(0) << std::endl;
        loss.backward();

        {
            torch::NoGradGuard no_grad;
            auto visible_max_radii = gaussians._max_radii2D.masked_select(visibility_filter);
            auto visible_radii = radii.masked_select(visibility_filter);
            auto max_radii = torch::max(visible_max_radii, visible_radii);
            gaussians._max_radii2D.masked_scatter_(visibility_filter, max_radii);

            if (iter % 7'000 == 0) {
                gaussians.Save_ply(modelParams.model_path, iter, false);
            }
            if (iter == 30'000) {
                gaussians.Save_ply(modelParams.model_path, iter, true);
                return 0;
            }

            // Densification
            if (iter < optimParams.densify_until_iter) {
                gaussians.Add_densification_stats(viewspace_point_tensor, visibility_filter);
                if (iter > optimParams.densify_from_iter && iter % optimParams.densification_interval == 0) {
                    // @TODO: Not sure about type
                    float size_threshold = iter > optimParams.opacity_reset_interval ? 20.f : -1.f;
                    gaussians.Densify_and_prune(optimParams.densify_grad_threshold, 0.005f, scene.Get_cameras_extent(), size_threshold);
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
