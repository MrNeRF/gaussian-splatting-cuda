// Copyright (c) 2023 Janusch Patas.

#include "core/argument_parser.hpp"
#include "core/parameters.hpp"
#include <args.hxx>
#include <filesystem>
#include <iostream>
#include <set>

namespace gs {
    namespace args {
        static void scale_steps_vector(std::vector<size_t>& steps, size_t scaler) {
            std::set<size_t> unique_steps(steps.begin(), steps.end());
            for (const auto& step : steps) {
                for (size_t i = 1; i <= scaler; ++i) {
                    unique_steps.insert(step * i);
                }
            }
            steps.assign(unique_steps.begin(), unique_steps.end());
        }

        std::vector<std::string> convert_args(int argc, char* argv[]) {
            return std::vector<std::string>(argv, argv + argc);
        }

        int parse_arguments(const std::vector<std::string>& args,
                            gs::param::TrainingParameters& params) {

            if (args.empty()) {
                std::cerr << "No command line arguments provided!" << std::endl;
                return -1;
            }

            // Set up argument parser
            ::args::ArgumentParser parser(
                "3D Gaussian Splatting CUDA Implementation\n",
                "This program provides a lightning-fast CUDA implementation of the 3D Gaussian Splatting algorithm for real-time radiance field rendering.");

            // Define all command line arguments
            ::args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
            ::args::ValueFlag<float> convergence_rate(parser, "convergence_rate", "Set convergence rate", {'c', "convergence_rate"});
            ::args::ValueFlag<int> resolution(parser, "resolution", "Set resolution", {'r', "resolution"});
            ::args::Flag force_overwrite_output_path(parser, "force", "Force overwrite of output folder", {'f', "force"});
            ::args::ValueFlag<std::string> data_path(parser, "data_path", "Path to the training data", {'d', "data-path"});
            ::args::ValueFlag<std::string> output_path(parser, "output_path", "Path to the training output", {'o', "output-path"});
            ::args::ValueFlag<uint32_t> iterations(parser, "iterations", "Number of iterations to train the model", {'i', "iter"});
            ::args::CompletionFlag completion(parser, {"complete"});
            ::args::ValueFlag<int> max_cap(parser, "max_cap", "Maximum number of Gaussians for MCMC", {"max-cap"});
            ::args::Flag use_bilateral_grid(parser, "bilateral_grid", "Enable bilateral grid filtering", {"bilateral-grid"});
            ::args::ValueFlag<std::string> images_folder(parser, "images", "Images folder name (e.g., images, images_2, images_4, images_8)", {"images"});
            ::args::ValueFlag<int> test_every(parser, "test_every", "Use every Nth image as a test image", {"test-every"});
            ::args::ValueFlag<int> steps_scaler(parser, "steps_scaler", "Scale all training steps by this factor", {"steps-scaler"});
            ::args::ValueFlag<int> sh_degree_interval(parser, "sh_degree_interval", "Interval for increasing spherical harmonics degree", {"sh-degree-interval"});
            ::args::Flag enable_save_eval_images(parser, "save_eval_images", "Save images and depth maps during evaluation (based on render mode)", {"save-eval-images"});
            ::args::Flag enable_eval(parser, "eval", "Enable evaluation during training", {"eval"});
            ::args::Flag selective_adam(parser, "selective_adam", "Enable selective adam", {"selective-adam"});

            // Add render mode arguments
            ::args::ValueFlag<std::string> render_mode(parser, "render_mode", "Render mode: RGB, D, ED, RGB_D, RGB_ED", {"render-mode"});
            ::args::Flag save_depth(parser, "save_depth", "Save depth maps during training", {"save-depth"});

            // Parse arguments
            try {
                parser.Prog(args.front());
                parser.ParseArgs(std::vector<std::string>(args.begin() + 1, args.end()));
            } catch (const ::args::Completion& e) {
                std::cout << e.what();
                return 0;
            } catch (const ::args::Help&) {
                std::cout << parser;
                return -1;
            } catch (const ::args::ParseError& e) {
                std::cerr << e.what() << std::endl;
                std::cerr << parser;
                return -1;
            }

            // Process parsed arguments and populate parameters

            // Data path is required
            if (::args::get(data_path).empty()) {
                std::cerr << "ERROR: No data path specified. Use --data-path to specify the path to the dataset.\n";
                return -1;
            }
            params.dataset.data_path = ::args::get(data_path);

            // Handle output path
            if (::args::get(output_path).empty()) {
                std::cerr << "ERROR: No output path specified. Use --output-path to specify the path for output files.\n";
                return -1;
            }
            params.dataset.output_path = ::args::get(output_path);

            // Create output directory if it doesn't exist
            std::filesystem::path outputDir = params.dataset.output_path;
            if (!std::filesystem::exists(outputDir)) {
                std::filesystem::create_directories(outputDir);
            }

            // Process optional arguments
            if (iterations) {
                params.optimization.iterations = ::args::get(iterations);
            }

            if (resolution) {
                params.dataset.resolution = ::args::get(resolution);
            }

            if (max_cap) {
                params.optimization.max_cap = ::args::get(max_cap);
            }

            if (use_bilateral_grid) {
                params.optimization.use_bilateral_grid = true;
            }

            if (images_folder) {
                params.dataset.images = ::args::get(images_folder);
            }

            if (test_every) {
                params.dataset.test_every = ::args::get(test_every);
            }

            if (steps_scaler) {
                params.optimization.steps_scaler = ::args::get(steps_scaler);
            }

            if (sh_degree_interval) {
                params.optimization.sh_degree_interval = ::args::get(sh_degree_interval);
            }

            if (enable_eval) {
                params.optimization.enable_eval = true;
            }

            if (selective_adam) {
                params.optimization.selective_adam = true;
            }

            if (enable_save_eval_images) {
                params.optimization.enable_save_eval_images = true;
            }

            // Process render mode arguments
            if (render_mode) {
                std::string mode = ::args::get(render_mode);
                // Validate render mode
                if (mode != "RGB" && mode != "D" && mode != "ED" &&
                    mode != "RGB_D" && mode != "RGB_ED") {
                    std::cerr << "ERROR: Invalid render mode '" << mode << "'. ";
                    std::cerr << "Valid modes are: RGB, D, ED, RGB_D, RGB_ED\n";
                    return -1;
                }
                params.optimization.render_mode = mode;
            }

            return 0; // Success
        }
    } // namespace args
} // namespace gs

gs::param::TrainingParameters gs::args::parse_args_and_params(int argc, char* argv[]) {
    gs::param::TrainingParameters params;
    params.optimization = gs::param::read_optim_params_from_json();

    if (parse_arguments(convert_args(argc, argv), params) < 0) {
        throw std::runtime_error("Failed to parse arguments");
    }

    // Apply step scaling if specified
    if (params.optimization.steps_scaler > 1) {
        std::cout << "Scaling training steps by factor: " << params.optimization.steps_scaler << std::endl;
        params.optimization.iterations *= params.optimization.steps_scaler;
        params.optimization.start_refine *= params.optimization.steps_scaler;
        params.optimization.stop_refine *= params.optimization.steps_scaler;
        params.optimization.refine_every *= params.optimization.steps_scaler;
        params.optimization.sh_degree_interval *= params.optimization.steps_scaler;

        scale_steps_vector(params.optimization.eval_steps, params.optimization.steps_scaler);
        scale_steps_vector(params.optimization.save_steps, params.optimization.steps_scaler);
    }

    return params;
}