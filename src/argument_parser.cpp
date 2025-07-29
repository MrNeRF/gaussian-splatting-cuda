// Copyright (c) 2025 Janusch Patas.

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
        Help
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

    std::expected<std::tuple<ParseResult, std::function<void()>>, std::string> parse_arguments(
        const std::vector<std::string>& args,
        gs::param::TrainingParameters& params) {

        try {
            ::args::ArgumentParser parser(
                "3D Gaussian Splatting CUDA Implementation\n",
                "Lightning-fast CUDA implementation of 3D Gaussian Splatting algorithm.\n\n"
                "Usage:\n"
                "  Training:  gs_cuda --data-path <path> --output-path <path> [options]\n"
                "  Viewing:   gs_cuda --view <path_to_ply> [options]\n");

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
            ::args::ValueFlag<std::string> attention_masks_folder(parser, "attention_masks_folder", "Attention masks folder name", {"attention-masks-folder"});
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
            ::args::Flag use_attention_mask(parser, "attention_masks", "Use attention masks on training", {"attention-masks"});
            ::args::Flag preload_to_ram(parser, "preload_to_ram", "Load the entire dataset into RAM at startup for maximum performance (uses more RAM)", {"preload-to-ram"});

            // Parse arguments
            try {
                parser.Prog(args.front());
                parser.ParseArgs(std::vector<std::string>(args.begin() + 1, args.end()));
            } catch (const ::args::Help&) {
                std::print("{}", parser.Help());
                return std::make_tuple(ParseResult::Help, std::function<void()>{});
            } catch (const ::args::Completion& e) {
                std::print("{}", e.what());
                return std::make_tuple(ParseResult::Help, std::function<void()>{});
            } catch (const ::args::ParseError& e) {
                return std::unexpected(std::format("Parse error: {}\n{}", e.what(), parser.Help()));
            }

            // Check if explicitly displaying help
            if (help) {
                return std::make_tuple(ParseResult::Help, std::function<void()>{});
            }

            // NO ARGUMENTS = VIEWER MODE (empty)
            if (args.size() == 1) {
                return std::make_tuple(ParseResult::Success, std::function<void()>{});
            }

            // Check for viewer mode with PLY
            if (view_ply) {
                const auto ply_path = ::args::get(view_ply);
                if (!ply_path.empty()) {
                    params.ply_path = ply_path;

                    // Check if PLY file exists
                    if (!std::filesystem::exists(params.ply_path)) {
                        return std::unexpected(std::format("PLY file does not exist: {}", params.ply_path.string()));
                    }
                }

                return std::make_tuple(ParseResult::Success, std::function<void()>{});
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
            auto apply_cmd_overrides = [&params,
                                        // Capture values, not references
                                        iterations_val = iterations ? std::optional<uint32_t>(::args::get(iterations)) : std::optional<uint32_t>(),
                                        resolution_val = resolution ? std::optional<int>(::args::get(resolution)) : std::optional<int>(),
                                        max_cap_val = max_cap ? std::optional<int>(::args::get(max_cap)) : std::optional<int>(),
                                        images_folder_val = images_folder ? std::optional<std::string>(::args::get(images_folder)) : std::optional<std::string>(),
                                        test_every_val = test_every ? std::optional<int>(::args::get(test_every)) : std::optional<int>(),
                                        steps_scaler_val = steps_scaler ? std::optional<float>(::args::get(steps_scaler)) : std::optional<float>(),
                                        sh_degree_interval_val = sh_degree_interval ? std::optional<int>(::args::get(sh_degree_interval)) : std::optional<int>(),
                                        sh_degree_val = sh_degree ? std::optional<int>(::args::get(sh_degree)) : std::optional<int>(),
                                        min_opacity_val = min_opacity ? std::optional<float>(::args::get(min_opacity)) : std::optional<float>(),
                                        render_mode_val = render_mode ? std::optional<std::string>(::args::get(render_mode)) : std::optional<std::string>(),
                                        // Capture flag states
                                        preload_to_ram_flag = bool(preload_to_ram),
                                        use_bilateral_grid_flag = bool(use_bilateral_grid),
                                        use_attention_mask_flag = bool(use_attention_mask),
                                        enable_eval_flag = bool(enable_eval),
                                        headless_flag = bool(headless),
                                        antialiasing_flag = bool(antialiasing),
                                        selective_adam_flag = bool(selective_adam),
                                        enable_save_eval_images_flag = bool(enable_save_eval_images),
                                        skip_intermediate_saving_flag = bool(skip_intermediate_saving)]() {
                auto& opt = params.optimization;
                auto& ds = params.dataset;

                // Simple lambdas to apply if flag/value exists
                auto setVal = [](const auto& flag, auto& target) {
                    if (flag)
                        target = *flag;
                };

                auto setFlag = [](bool flag, auto& target) {
                    if (flag)
                        target = true;
                };

                // Apply all overrides
                setVal(iterations_val, opt.iterations);
                setVal(resolution_val, ds.resolution);
                setVal(max_cap_val, opt.max_cap);
                setVal(images_folder_val, ds.images);
                setVal(test_every_val, ds.test_every);
                setVal(steps_scaler_val, opt.steps_scaler);
                setVal(sh_degree_interval_val, opt.sh_degree_interval);
                setVal(sh_degree_val, opt.sh_degree);
                setVal(min_opacity_val, opt.min_opacity);
                setVal(render_mode_val, opt.render_mode);

                setFlag(use_attention_mask_flag, opt.use_attention_mask);
                setFlag(use_bilateral_grid_flag, opt.use_bilateral_grid);
                setFlag(preload_to_ram_flag, opt.preload_to_ram);
                setFlag(enable_eval_flag, opt.enable_eval);
                setFlag(headless_flag, opt.headless);
                setFlag(antialiasing_flag, opt.antialiasing);
                setFlag(selective_adam_flag, opt.selective_adam);
                setFlag(enable_save_eval_images_flag, opt.enable_save_eval_images);
                setFlag(skip_intermediate_saving_flag, opt.skip_intermediate_saving);
            };

            return std::make_tuple(ParseResult::Success, apply_cmd_overrides);

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

    std::vector<std::string> convert_args(int argc, const char* const argv[]) {
        return std::vector<std::string>(argv, argv + argc);
    }

} // anonymous namespace

// Public interface
std::expected<std::unique_ptr<gs::param::TrainingParameters>, std::string>
gs::args::parse_args_and_params(int argc, const char* const argv[]) {

    auto params = std::make_unique<gs::param::TrainingParameters>();

    // Parse command line arguments
    auto parse_result = parse_arguments(convert_args(argc, argv), *params);
    if (!parse_result) {
        return std::unexpected(parse_result.error());
    }

    auto [result, apply_overrides] = *parse_result;

    // Handle help case
    if (result == ParseResult::Help) {
        std::exit(0);
    }

    // Training mode - load JSON first
    if (!params->dataset.data_path.empty()) {
        auto opt_params_result = gs::param::read_optim_params_from_json();
        if (!opt_params_result) {
            return std::unexpected(std::format("Failed to load optimization parameters: {}",
                                               opt_params_result.error()));
        }
        params->optimization = *opt_params_result;
    }

    // Apply command line overrides
    if (apply_overrides) {
        apply_overrides();
    }

    // Apply step scaling
    apply_step_scaling(*params);

    return params;
}