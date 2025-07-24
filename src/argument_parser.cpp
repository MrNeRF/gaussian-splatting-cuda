// Copyright (c) 2023 Janusch Patas.

#include "core/argument_parser.hpp"
#include "core/parameters.hpp"
#include <args.hxx>
#include <expected>
#include <filesystem>
#include <format>
#include <print>
#include <set>

namespace {

    enum class ParseResult {
        Success,
        Help,
        Error,
        ViewerMode
    };

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

    std::expected<ParseResult, std::string> parse_arguments(
        const std::vector<std::string>& args,
        gs::param::TrainingParameters& params) {

        try {
            ::args::ArgumentParser parser(
                "3D Gaussian Splatting CUDA Implementation\n",
                "Lightning-fast CUDA implementation of 3D Gaussian Splatting algorithm.\n\n"
                "Usage:\n"
                "  Training:  gs_cuda --data-path <path> --output-path <path> [options]\n"
                "  Viewing:   gs_cuda --view <path_to_ply> [options]\n"
                "  Interactive: gs_cuda  (starts empty viewer)");

            // Define all arguments
            ::args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});
            ::args::CompletionFlag completion(parser, {"complete"});

            // PLY viewing mode
            ::args::ValueFlag<std::string> view_ply(parser, "ply_file", "View a PLY file", {'v', "view"});

            // Training mode arguments
            ::args::ValueFlag<std::string> data_path(parser, "data_path", "Path to training data", {'d', "data-path"});
            ::args::ValueFlag<std::string> output_path(parser, "output_path", "Path to output", {'o', "output-path"});

            // Optional value arguments
            ::args::ValueFlag<uint32_t> iterations(parser, "iterations", "Number of iterations", {'i', "iter"});
            ::args::ValueFlag<int> resolution(parser, "resolution", "Set resolution", {'r', "resolution"});
            ::args::ValueFlag<int> max_cap(parser, "max_cap", "Max Gaussians for MCMC", {"max-cap"});
            ::args::ValueFlag<std::string> images_folder(parser, "images", "Images folder name", {"images"});
            ::args::ValueFlag<int> test_every(parser, "test_every", "Use every Nth image as test", {"test-every"});
            ::args::ValueFlag<float> steps_scaler(parser, "steps_scaler", "Scale training steps by factor", {"steps-scaler"});
            ::args::ValueFlag<int> sh_degree_interval(parser, "sh_degree_interval", "SH degree interval", {"sh-degree-interval"});
            ::args::ValueFlag<int> sh_degree(parser, "sh_degree", "Max SH degree [1-3]", {"sh-degree"});
            ::args::ValueFlag<float> min_opacity(parser, "min_opacity", "Minimum opacity threshold", {"min-opacity"});
            ::args::ValueFlag<std::string> render_mode(parser, "render_mode", "Render mode: RGB, D, ED, RGB_D, RGB_ED", {"render-mode"});

            // Optional flag arguments
            ::args::Flag use_bilateral_grid(parser, "bilateral_grid", "Enable bilateral grid filtering", {"bilateral-grid"});
            ::args::Flag enable_eval(parser, "eval", "Enable evaluation during training", {"eval"});
            ::args::Flag headless(parser, "headless", "Disable visualization during training", {"headless"});
            ::args::Flag antialiasing(parser, "antialiasing", "Enable antialiasing", {'a', "antialiasing"});
            ::args::Flag selective_adam(parser, "selective_adam", "Enable selective adam", {"selective-adam"});
            ::args::Flag enable_save_eval_images(parser, "save_eval_images", "Save eval images and depth maps", {"save-eval-images"});
            ::args::Flag save_depth(parser, "save_depth", "Save depth maps during training", {"save-depth"});
            ::args::Flag skip_intermediate_saving(parser, "skip_intermediate", "Skip saving intermediate results and only save final output", {"skip-intermediate"});

            // Parse arguments
            try {
                parser.Prog(args.front());
                parser.ParseArgs(std::vector<std::string>(args.begin() + 1, args.end()));
            } catch (const ::args::Help&) {
                std::print("{}", parser.Help());
                return ParseResult::Help;
            } catch (const ::args::Completion& e) {
                std::print("{}", e.what());
                return ParseResult::Help;
            } catch (const ::args::ParseError& e) {
                return std::unexpected(std::format("Parse error: {}\n{}", e.what(), parser.Help()));
            }

            // Check if explicitly displaying help
            if (help) {
                return ParseResult::Help;
            }

            // NO ARGUMENTS = INTERACTIVE VIEWER MODE
            if (args.size() == 1) {
                params.viewer_mode = true;
                return ParseResult::ViewerMode;
            }

            // Check for viewer mode with PLY
            if (view_ply) {
                params.viewer_mode = true;
                const auto ply_path = ::args::get(view_ply);
                if (!ply_path.empty()) {
                    params.ply_path = ply_path;

                    // Check if PLY file exists
                    if (!std::filesystem::exists(params.ply_path)) {
                        return std::unexpected(std::format("PLY file does not exist: {}", params.ply_path.string()));
                    }
                }

                return ParseResult::ViewerMode;
            }

            // Training mode
            bool has_data_path = data_path && !::args::get(data_path).empty();
            bool has_output_path = output_path && !::args::get(output_path).empty();

            // If headless mode, require data path
            if (headless && !has_data_path) {
                return std::unexpected(std::format(
                    "ERROR: Headless mode requires --data-path\n\n{}",
                    parser.Help()));
            }

            // If both paths provided, it's training mode
            if (has_data_path && has_output_path) {
                params.dataset.data_path = ::args::get(data_path);
                params.dataset.output_path = ::args::get(output_path);

                // Create output directory
                std::error_code ec;
                std::filesystem::create_directories(params.dataset.output_path, ec);
                if (ec) {
                    return std::unexpected(std::format(
                        "Failed to create output directory '{}': {}",
                        params.dataset.output_path.string(), ec.message()));
                }
            } else if (has_data_path != has_output_path) {
                return std::unexpected(std::format(
                    "ERROR: Training mode requires both --data-path and --output-path\n\n{}",
                    parser.Help()));
            }

            // Validate render mode if provided
            if (render_mode) {
                const auto mode = ::args::get(render_mode);
                if (VALID_RENDER_MODES.find(mode) == VALID_RENDER_MODES.end()) {
                    return std::unexpected(std::format(
                        "ERROR: Invalid render mode '{}'. Valid modes are: RGB, D, ED, RGB_D, RGB_ED",
                        mode));
                }
            }

            // Create lambda to apply command line overrides after JSON loading
            params._apply_cmd_overrides = [&]() {
                auto& opt = params.optimization;
                auto& ds = params.dataset;

                // Simple lambdas to apply if flag/value exists
                auto setVal = [](auto& flag, auto& target) {
                    if (flag)
                        target = ::args::get(flag);
                };

                auto setFlag = [](auto& flag, auto& target) {
                    if (flag)
                        target = true;
                };

                // Apply all overrides
                setVal(iterations, opt.iterations);
                setVal(resolution, ds.resolution);
                setVal(max_cap, opt.max_cap);
                setVal(images_folder, ds.images);
                setVal(test_every, ds.test_every);
                setVal(steps_scaler, opt.steps_scaler);
                setVal(sh_degree_interval, opt.sh_degree_interval);
                setVal(sh_degree, opt.sh_degree);
                setVal(min_opacity, opt.min_opacity);
                setVal(render_mode, opt.render_mode);

                setFlag(use_bilateral_grid, opt.use_bilateral_grid);
                setFlag(enable_eval, opt.enable_eval);
                setFlag(headless, opt.headless);
                setFlag(antialiasing, opt.antialiasing);
                setFlag(selective_adam, opt.selective_adam);
                setFlag(enable_save_eval_images, opt.enable_save_eval_images);
                setFlag(skip_intermediate_saving, opt.skip_intermediate_saving);
            };

            return ParseResult::Success;

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Unexpected error during argument parsing: {}", e.what()));
        }
    }

    void apply_step_scaling(gs::param::TrainingParameters& params) {
        auto& opt = params.optimization;
        const float scaler = opt.steps_scaler;

        if (scaler > 0) {
            std::println("Scaling training steps by factor: {}", scaler);

            opt.iterations *= scaler;
            opt.start_refine *= scaler;
            opt.stop_refine *= scaler;
            opt.refine_every *= scaler;
            opt.sh_degree_interval *= scaler;

            scale_steps_vector(opt.eval_steps, scaler);
            scale_steps_vector(opt.save_steps, scaler);
        }
    }

    std::vector<std::string> convert_args(int argc, char* argv[]) {
        return std::vector<std::string>(argv, argv + argc);
    }

} // anonymous namespace

// Public interface
std::expected<gs::param::TrainingParameters, std::string>
gs::args::parse_args_and_params(int argc, char* argv[]) {

    gs::param::TrainingParameters params;

    // Parse command line arguments
    auto parse_result = parse_arguments(convert_args(argc, argv), params);
    if (!parse_result) {
        return std::unexpected(parse_result.error());
    }

    // Handle help case
    if (*parse_result == ParseResult::Help) {
        std::exit(0);
    }

    // Handle viewer mode
    if (*parse_result == ParseResult::ViewerMode) {
        if (params._apply_cmd_overrides) {
            params._apply_cmd_overrides();
        }
        return params;
    }

    // Training mode - load JSON first
    if (!params.dataset.data_path.empty()) {
        auto opt_params_result = gs::param::read_optim_params_from_json();
        if (!opt_params_result) {
            return std::unexpected(std::format("Failed to load optimization parameters: {}",
                                               opt_params_result.error()));
        }
        params.optimization = *opt_params_result;
    }

    // Apply command line overrides
    if (params._apply_cmd_overrides) {
        params._apply_cmd_overrides();
    }

    // Apply step scaling
    apply_step_scaling(params);

    return params;
}