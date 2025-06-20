// Copyright (c) 2023 Janusch Patas.

#include "core/argument_parser.hpp"
#include "core/parameters.hpp"
#include <args.hxx>
#include <filesystem>
#include <iostream>
#include <set>

namespace {

    constexpr int HELP_EXIT_CODE = 0;
    constexpr int ERROR_EXIT_CODE = -1;
    constexpr int SUCCESS_EXIT_CODE = 1;

    const std::set<std::string> VALID_RENDER_MODES = {"RGB", "D", "ED", "RGB_D", "RGB_ED"};

    void scale_steps_vector(std::vector<size_t>& steps, size_t scaler) {
        std::set<size_t> unique_steps(steps.begin(), steps.end());
        for (const auto& step : steps) {
            for (size_t i = 1; i <= scaler; ++i) {
                unique_steps.insert(step * i);
            }
        }
        steps.assign(unique_steps.begin(), unique_steps.end());
    }

    int parse_arguments(const std::vector<std::string>& args,
                        gs::param::TrainingParameters& params) {

        ::args::ArgumentParser parser(
            "3D Gaussian Splatting CUDA Implementation\n",
            "Lightning-fast CUDA implementation of 3D Gaussian Splatting algorithm.");

        // Define all arguments
        ::args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});
        ::args::CompletionFlag completion(parser, {"complete"});

        // Required arguments
        ::args::ValueFlag<std::string> data_path(parser, "data_path", "Path to training data", {'d', "data-path"});
        ::args::ValueFlag<std::string> output_path(parser, "output_path", "Path to output", {'o', "output-path"});

        // Optional value arguments
        ::args::ValueFlag<uint32_t> iterations(parser, "iterations", "Number of iterations", {'i', "iter"});
        ::args::ValueFlag<int> resolution(parser, "resolution", "Set resolution", {'r', "resolution"});
        ::args::ValueFlag<int> max_cap(parser, "max_cap", "Max Gaussians for MCMC", {"max-cap"});
        ::args::ValueFlag<std::string> images_folder(parser, "images", "Images folder name", {"images"});
        ::args::ValueFlag<int> test_every(parser, "test_every", "Use every Nth image as test", {"test-every"});
        ::args::ValueFlag<int> steps_scaler(parser, "steps_scaler", "Scale training steps by factor", {"steps-scaler"});
        ::args::ValueFlag<int> sh_degree_interval(parser, "sh_degree_interval", "SH degree interval", {"sh-degree-interval"});
        ::args::ValueFlag<std::string> render_mode(parser, "render_mode", "Render mode: RGB, D, ED, RGB_D, RGB_ED", {"render-mode"});

        // Optional flag arguments
        ::args::Flag use_bilateral_grid(parser, "bilateral_grid", "Enable bilateral grid filtering", {"bilateral-grid"});
        ::args::Flag enable_eval(parser, "eval", "Enable evaluation during training", {"eval"});
        ::args::Flag enable_viz(parser, "viz", "Enable visualization during training", {'v', "viz"});
        ::args::Flag selective_adam(parser, "selective_adam", "Enable selective adam", {"selective-adam"});
        ::args::Flag enable_save_eval_images(parser, "save_eval_images", "Save eval images and depth maps", {"save-eval-images"});
        ::args::Flag save_depth(parser, "save_depth", "Save depth maps during training", {"save-depth"});

        // Parse arguments
        try {
            parser.Prog(args.front());
            parser.ParseArgs(std::vector<std::string>(args.begin() + 1, args.end()));
        } catch (const ::args::Help&) {
            std::cout << parser;
            return HELP_EXIT_CODE;
        } catch (const ::args::Completion& e) {
            std::cout << e.what();
            return HELP_EXIT_CODE;
        } catch (const ::args::ParseError& e) {
            std::cerr << e.what() << "\n"
                      << parser;
            return ERROR_EXIT_CODE;
        }

        // Show help if no arguments provided
        if (args.size() == 1) {
            std::cout << parser;
            return HELP_EXIT_CODE;
        }

        // Check if just displaying help
        if (args.size() == 2 && (args[1] == "-h" || args[1] == "--help")) {
            return HELP_EXIT_CODE;
        }

        // Validate required arguments - check if flags were provided
        if (!data_path || !output_path) {
            std::cerr << "ERROR: Both --data-path and --output-path are required\n"
                      << "\n"
                      << parser;
            return ERROR_EXIT_CODE;
        }

        // Set required parameters
        params.dataset.data_path = ::args::get(data_path);
        params.dataset.output_path = ::args::get(output_path);

        // Create output directory
        std::filesystem::create_directories(params.dataset.output_path);

        // Lambdas for compact argument processing
        auto setVal = [](auto& flag, auto& target) {
            if (flag)
                target = ::args::get(flag);
        };

        auto setFlag = [](auto& flag, auto& target) {
            if (flag)
                target = true;
        };

        // Process all optional arguments compactly
        auto& opt = params.optimization;
        auto& ds = params.dataset;

        // Value arguments
        setVal(iterations, opt.iterations);
        setVal(resolution, ds.resolution);
        setVal(max_cap, opt.max_cap);
        setVal(images_folder, ds.images);
        setVal(test_every, ds.test_every);
        setVal(steps_scaler, opt.steps_scaler);
        setVal(sh_degree_interval, opt.sh_degree_interval);

        // Flag arguments
        setFlag(use_bilateral_grid, opt.use_bilateral_grid);
        setFlag(enable_eval, opt.enable_eval);
        setFlag(enable_viz, opt.enable_viz);
        setFlag(selective_adam, opt.selective_adam);
        setFlag(enable_save_eval_images, opt.enable_save_eval_images);

        // Special case: validate render mode
        if (render_mode) {
            const auto mode = ::args::get(render_mode);
            if (VALID_RENDER_MODES.find(mode) == VALID_RENDER_MODES.end()) {
                std::cerr << "ERROR: Invalid render mode '" << mode
                          << "'. Valid modes are: RGB, D, ED, RGB_D, RGB_ED\n";
                return ERROR_EXIT_CODE;
            }
            opt.render_mode = mode;
        }

        return SUCCESS_EXIT_CODE;
    }

    void apply_step_scaling(gs::param::TrainingParameters& params) {
        auto& opt = params.optimization;
        const size_t scaler = opt.steps_scaler;

        if (scaler > 1) {
            std::cout << "Scaling training steps by factor: " << scaler << std::endl;

            opt.iterations *= scaler;
            opt.start_refine *= scaler;
            opt.stop_refine *= scaler;
            opt.refine_every *= scaler;
            opt.sh_degree_interval *= scaler;

            scale_steps_vector(opt.eval_steps, scaler);
            scale_steps_vector(opt.save_steps, scaler);
        }
    }

    std::vector<std::string> convert_args(int argc, const char* const argv[]) {
        return std::vector<std::string>(argv, argv + argc);
    }

} // anonymous namespace

// Public interface
gs::param::TrainingParameters gs::args::parse_args_and_params(int argc, const char* const argv[]) {
    gs::param::TrainingParameters params;
    params.optimization = gs::param::read_optim_params_from_json();

    const int result = parse_arguments(convert_args(argc, argv), params);

    if (result < 0) {
        throw std::runtime_error("Failed to parse arguments");
    } else if (result == 0) {
        std::exit(0);
    }

    apply_step_scaling(params);

    return params;
}