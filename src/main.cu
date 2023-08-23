#include "debug_utils.cuh"
#include "gaussian.cuh"
#include "loss_monitor.cuh"
#include "loss_utils.cuh"
#include "parameters.cuh"
#include "render_utils.cuh"
#include "scene.cuh"
#include <args.hxx>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <torch/torch.h>

void Write_model_parameters_to_file(const ModelParameters& params) {
    std::filesystem::path outputPath = params.output_path;
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
    // Shuffle the vector
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine());
    return indices;
}

int parse_cmd_line_args(const std::vector<std::string>& args,
                        ModelParameters& modelParams,
                        OptimizationParameters& optimParams,
                        PipelineParameters& pipelineParams) {
    if (args.empty()) {
        std::cerr << "No command line arguments provided!" << std::endl;
        return -1;
    }
    args::ArgumentParser parser("3D Gaussian Splatting CUDA Implementation\n",
                                "This program provides a lightning-fast CUDA implementation of the 3D Gaussian Splatting algorithm for real-time radiance field rendering.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<float> convergence_rate(parser, "convergence_rate", "Set convergence rate", {'c', "convergence_rate"});
    args::Flag enable_cr_monitoring(parser, "enable_cr_monitoring", "Enable convergence rate monitoring", {"enable-cr-monitoring"});
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
            if (!std::filesystem::create_directory(outputDir)) {
                std::cerr << "Directory already exists! Not overwriting it" << std::endl;
                return -1;
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
    return 0;
}

int main(int argc, char* argv[]) {
    std::vector<std::string> args;
    args.reserve(argc);

    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    // TODO: read parameters from JSON file or command line
    auto modelParams = ModelParameters();
    auto optimParams = OptimizationParameters();
    auto pipelineParams = PipelineParameters();
    if (parse_cmd_line_args(args, modelParams, optimParams, pipelineParams) < 0) {
        return -1;
    };
    Write_model_parameters_to_file(modelParams);

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
    float loss_add = 0.f;

    LossMonitor loss_monitor(200);
    float avg_converging_rate = 0.f;

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
        //        std::cout << "Iteration: " << iter << " Loss: " << loss.item<float>() << " gaussian splats: " << gaussians.Get_xyz().size(0) << std::endl;
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

            if (iter == optimParams.iterations) {
                gaussians.Save_ply(modelParams.output_path, iter, true);
                return 0;
            }

            if (iter % 7'000 == 0) {
                gaussians.Save_ply(modelParams.output_path, iter, false);
            }

            if (iter % 100 == 0) {
                std::cout << "Iteration: " << iter
                          << " Loss: " << loss_add / 100.f
                          << " Average Convergence rate: " << avg_converging_rate
                          << " gaussian splats: " << gaussians.Get_xyz().size(0) << "\n";
                loss_add = 0.f;
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

            if (iter >= optimParams.densify_until_iter && loss_monitor.IsConverging(optimParams.convergence_threshold)) {
                std::cout << "Converged after " << iter << " iterations!" << std::endl;
                gaussians.Save_ply(modelParams.output_path, iter, true);
                return 0;
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
