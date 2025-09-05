/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/argument_parser.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include <args.hxx>
#include <expected>
#include <filesystem>
#include <format>
#include <print>
#include <set>
#include <unordered_map>

namespace {

    enum class ParseResult {
        Success,
        Help
    };

    const std::set<std::string> VALID_RENDER_MODES = {"RGB", "D", "ED", "RGB_D", "RGB_ED"};
    const std::set<std::string> VALID_POSE_OPTS = {"none", "direct", "mlp"};
    const std::set<std::string> VALID_STRATEGIES = {"mcmc", "default"};

    void scale_steps_vector(std::vector<size_t>& steps, size_t scaler) {
        std::set<size_t> unique_steps(steps.begin(), steps.end());
        for (const auto& step : steps) {
            for (size_t i = 1; i <= scaler; ++i) {
                unique_steps.insert(step * i);
            }
        }
        steps.assign(unique_steps.begin(), unique_steps.end());
    }

    // Parse log level from string
    gs::core::LogLevel parse_log_level(const std::string& level_str) {
        if (level_str == "trace")
            return gs::core::LogLevel::Trace;
        if (level_str == "debug")
            return gs::core::LogLevel::Debug;
        if (level_str == "info")
            return gs::core::LogLevel::Info;
        if (level_str == "warn" || level_str == "warning")
            return gs::core::LogLevel::Warn;
        if (level_str == "error")
            return gs::core::LogLevel::Error;
        if (level_str == "critical")
            return gs::core::LogLevel::Critical;
        if (level_str == "off")
            return gs::core::LogLevel::Off;
        return gs::core::LogLevel::Info; // Default
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

            // LichtFeldStudio project arguments
            ::args::ValueFlag<std::string> project_name(parser, "proj_path", "LichtFeldStudio project path. Path must end with .ls", {"proj_path"});

            // Training mode arguments
            ::args::ValueFlag<std::string> data_path(parser, "data_path", "Path to training data", {'d', "data-path"});
            ::args::ValueFlag<std::string> output_path(parser, "output_path", "Path to output", {'o', "output-path"});

            // Optional value arguments
            ::args::ValueFlag<uint32_t> iterations(parser, "iterations", "Number of iterations", {'i', "iter"});
            ::args::ValueFlag<int> max_cap(parser, "max_cap", "Max Gaussians for MCMC", {"max-cap"});
            ::args::ValueFlag<std::string> images_folder(parser, "images", "Images folder name", {"images"});
            ::args::ValueFlag<int> test_every(parser, "test_every", "Use every Nth image as test", {"test-every"});
            ::args::ValueFlag<float> steps_scaler(parser, "steps_scaler", "Scale training steps by factor", {"steps-scaler"});
            ::args::ValueFlag<int> sh_degree_interval(parser, "sh_degree_interval", "SH degree interval", {"sh-degree-interval"});
            ::args::ValueFlag<int> sh_degree(parser, "sh_degree", "Max SH degree [1-3]", {"sh-degree"});
            ::args::ValueFlag<float> min_opacity(parser, "min_opacity", "Minimum opacity threshold", {"min-opacity"});
            ::args::ValueFlag<std::string> render_mode(parser, "render_mode", "Render mode: RGB, D, ED, RGB_D, RGB_ED", {"render-mode"});
            ::args::ValueFlag<std::string> pose_opt(parser, "pose_opt", "Enable pose optimization type: none, direct, mlp", {"pose-opt"});
            ::args::ValueFlag<std::string> strategy(parser, "strategy", "Optimization strategy: mcmc, default", {"strategy"});
            ::args::ValueFlag<int> init_num_pts(parser, "init_num_pts", "Number of random initialization points", {"init-num-pts"});
            ::args::ValueFlag<float> init_extent(parser, "init_extent", "Extent of random initialization", {"init-extent"});
            ::args::ValueFlagList<std::string> timelapse_images(parser, "timelapse_images", "Image filenames to render timelapse images for", {"timelapse-images"});
            ::args::ValueFlag<int> timelapse_every(parser, "timelapse_every", "Render timelapse image every N iterations (default: 50)", {"timelapse-every"});
            ::args::ValueFlag<std::string> init_ply(parser, "init_ply", "Optional PLY splat file for initialization", {"init-ply"});

            // Sparsity optimization arguments
            ::args::ValueFlag<int> sparsify_steps(parser, "sparsify_steps", "Number of steps for sparsification (default: 15000)", {"sparsify-steps"});
            ::args::ValueFlag<float> init_rho(parser, "init_rho", "Initial ADMM penalty parameter (default: 0.0005)", {"init-rho"});
            ::args::ValueFlag<float> prune_ratio(parser, "prune_ratio", "Final pruning ratio for sparsity (default: 0.6)", {"prune-ratio"});

            // SOG format arguments
            ::args::ValueFlag<int> sog_iterations(parser, "sog_iterations", "K-means iterations for SOG compression (default: 10)", {"sog-iterations"});

            // Logging options
            ::args::ValueFlag<std::string> log_level(parser, "level", "Log level: trace, debug, info, warn, error, critical, off (default: info)", {"log-level"});
            ::args::ValueFlag<std::string> log_file(parser, "file", "Optional log file path", {"log-file"});

            // Optional flag arguments
            ::args::Flag use_bilateral_grid(parser, "bilateral_grid", "Enable bilateral grid filtering", {"bilateral-grid"});
            ::args::Flag enable_eval(parser, "eval", "Enable evaluation during training", {"eval"});
            ::args::Flag headless(parser, "headless", "Disable visualization during training", {"headless"});
            ::args::Flag antialiasing(parser, "antialiasing", "Enable antialiasing", {'a', "antialiasing"});
            ::args::Flag enable_save_eval_images(parser, "save_eval_images", "Save eval images and depth maps", {"save-eval-images"});
            ::args::Flag save_depth(parser, "save_depth", "Save depth maps during training", {"save-depth"});
            ::args::Flag skip_intermediate_saving(parser, "skip_intermediate", "Skip saving intermediate results and only save final output", {"skip-intermediate"});
            ::args::Flag random(parser, "random", "Use random initialization instead of SfM", {"random"});
            ::args::Flag gut(parser, "gut", "Enable GUT mode", {"gut"});
            ::args::Flag enable_sparsity(parser, "enable_sparsity", "Enable sparsity optimization", {"enable-sparsity"});
            ::args::Flag rc(parser, "rc", "Workaround for reality captures - doesn't properly convert COLMAP camera model", {"rc"});
            ::args::Flag save_sog(parser, "sog", "Save in SOG format alongside PLY", {"sog"});

            ::args::MapFlag<std::string, int> resize_factor(parser, "resize_factor",
                                                            "resize resolution by this factor. Options: auto, 1, 2, 4, 8 (default: auto)",
                                                            {'r', "resize_factor"},
                                                            // load_image only supports those resizes
                                                            std::unordered_map<std::string, int>{
                                                                {"auto", 1},
                                                                {"1", 1},
                                                                {"2", 2},
                                                                {"4", 4},
                                                                {"8", 8}});

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

            // Initialize logger based on command line arguments
            {
                auto level = gs::core::LogLevel::Info; // Default level
                std::string log_file_path;

                if (log_level) {
                    level = parse_log_level(::args::get(log_level));
                }

                if (log_file) {
                    log_file_path = ::args::get(log_file);
                }

                // Initialize the logger with the specified level and optional file
                gs::core::Logger::get().init(level, log_file_path);

                // Log that the logger was initialized (without gs:: prefix)
                LOG_DEBUG("Logger initialized with level: {}", static_cast<int>(level));
                if (!log_file_path.empty()) {
                    LOG_DEBUG("Logging to file: {}", log_file_path);
                }
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

            if (init_ply) {
                const auto ply_path = ::args::get(init_ply);
                params.init_ply = ply_path;

                // Check if PLY file exists
                if (!std::filesystem::exists(ply_path)) {
                    return std::unexpected(std::format("Initialization PLY file does not exist: {}", ply_path));
                }
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
            if (strategy) {
                const auto strat = ::args::get(strategy);
                if (VALID_STRATEGIES.find(strat) == VALID_STRATEGIES.end()) {
                    return std::unexpected(std::format(
                        "ERROR: Invalid optimization strategy '{}'. Valid strategies are: mcmc, default",
                        strat));
                }

                // Set strategy immediately
                params.optimization.strategy = strat;
            }

            if (pose_opt) {
                const auto opt = ::args::get(pose_opt);
                if (VALID_POSE_OPTS.find(opt) == VALID_POSE_OPTS.end()) {
                    return std::unexpected(std::format(
                        "ERROR: Invalid pose optimization '{}'. Valid options are: none, direct, mlp",
                        opt));
                }
            }

            // Create lambda to apply command line overrides after JSON loading
            auto apply_cmd_overrides = [&params,
                                        // Capture values, not references
                                        iterations_val = iterations ? std::optional<uint32_t>(::args::get(iterations)) : std::optional<uint32_t>(),
                                        resize_factor_val = resize_factor ? std::optional<int>(::args::get(resize_factor)) : std::optional<int>(1),
                                        max_cap_val = max_cap ? std::optional<int>(::args::get(max_cap)) : std::optional<int>(),
                                        project_name_val = project_name ? std::optional<std::string>(::args::get(project_name)) : std::optional<std::string>(),
                                        images_folder_val = images_folder ? std::optional<std::string>(::args::get(images_folder)) : std::optional<std::string>(),
                                        test_every_val = test_every ? std::optional<int>(::args::get(test_every)) : std::optional<int>(),
                                        steps_scaler_val = steps_scaler ? std::optional<float>(::args::get(steps_scaler)) : std::optional<float>(),
                                        sh_degree_interval_val = sh_degree_interval ? std::optional<int>(::args::get(sh_degree_interval)) : std::optional<int>(),
                                        sh_degree_val = sh_degree ? std::optional<int>(::args::get(sh_degree)) : std::optional<int>(),
                                        min_opacity_val = min_opacity ? std::optional<float>(::args::get(min_opacity)) : std::optional<float>(),
                                        render_mode_val = render_mode ? std::optional<std::string>(::args::get(render_mode)) : std::optional<std::string>(),
                                        init_num_pts_val = init_num_pts ? std::optional<int>(::args::get(init_num_pts)) : std::optional<int>(),
                                        init_extent_val = init_extent ? std::optional<float>(::args::get(init_extent)) : std::optional<float>(),
                                        pose_opt_val = pose_opt ? std::optional<std::string>(::args::get(pose_opt)) : std::optional<std::string>(),
                                        strategy_val = strategy ? std::optional<std::string>(::args::get(strategy)) : std::optional<std::string>(),
                                        timelapse_images_val = timelapse_images ? std::optional<std::vector<std::string>>(::args::get(timelapse_images)) : std::optional<std::vector<std::string>>(),
                                        timelapse_every_val = timelapse_every ? std::optional<int>(::args::get(timelapse_every)) : std::optional<int>(),
                                        sog_iterations_val = sog_iterations ? std::optional<int>(::args::get(sog_iterations)) : std::optional<int>(),
                                        // Sparsity parameters
                                        sparsify_steps_val = sparsify_steps ? std::optional<int>(::args::get(sparsify_steps)) : std::optional<int>(),
                                        init_rho_val = init_rho ? std::optional<float>(::args::get(init_rho)) : std::optional<float>(),
                                        prune_ratio_val = prune_ratio ? std::optional<float>(::args::get(prune_ratio)) : std::optional<float>(),
                                        // Capture flag states
                                        use_bilateral_grid_flag = bool(use_bilateral_grid),
                                        enable_eval_flag = bool(enable_eval),
                                        rc_flag = bool(rc),
                                        headless_flag = bool(headless),
                                        antialiasing_flag = bool(antialiasing),
                                        enable_save_eval_images_flag = bool(enable_save_eval_images),
                                        skip_intermediate_saving_flag = bool(skip_intermediate_saving),
                                        random_flag = bool(random),
                                        gut_flag = bool(gut),
                                        save_sog_flag = bool(save_sog),
                                        enable_sparsity_flag = bool(enable_sparsity)]() {
                // Build JSON overrides object
                nlohmann::json overrides;

                // Dataset overrides
                if (resize_factor_val)
                    params.dataset.resize_factor = *resize_factor_val;
                if (project_name_val)
                    params.dataset.project_path = *project_name_val;
                if (images_folder_val)
                    params.dataset.images = *images_folder_val;
                if (test_every_val)
                    params.dataset.test_every = *test_every_val;
                if (timelapse_images_val)
                    params.dataset.timelapse_images = *timelapse_images_val;
                if (timelapse_every_val)
                    params.dataset.timelapse_every = *timelapse_every_val;

                // Optimization parameter overrides
                if (iterations_val)
                    overrides["iterations"] = *iterations_val;
                if (max_cap_val)
                    overrides["max_cap"] = *max_cap_val;
                if (steps_scaler_val)
                    overrides["steps_scaler"] = *steps_scaler_val;
                if (sh_degree_interval_val)
                    overrides["sh_degree_interval"] = *sh_degree_interval_val;
                if (sh_degree_val)
                    overrides["sh_degree"] = *sh_degree_val;
                if (min_opacity_val)
                    overrides["min_opacity"] = *min_opacity_val;
                if (render_mode_val)
                    overrides["render_mode"] = *render_mode_val;
                if (init_num_pts_val)
                    overrides["init_num_pts"] = *init_num_pts_val;
                if (init_extent_val)
                    overrides["init_extent"] = *init_extent_val;
                if (pose_opt_val)
                    overrides["pose_optimization"] = *pose_opt_val;
                if (strategy_val)
                    overrides["strategy"] = *strategy_val;
                if (sog_iterations_val)
                    overrides["sog_iterations"] = *sog_iterations_val;

                // Sparsity parameters
                if (sparsify_steps_val)
                    overrides["sparsify_steps"] = *sparsify_steps_val;
                if (init_rho_val)
                    overrides["init_rho"] = *init_rho_val;
                if (prune_ratio_val)
                    overrides["prune_ratio"] = *prune_ratio_val;

                // Flags
                if (use_bilateral_grid_flag)
                    overrides["use_bilateral_grid"] = true;
                if (enable_eval_flag)
                    overrides["enable_eval"] = true;
                if (rc_flag)
                    overrides["rc"] = true;
                if (headless_flag)
                    overrides["headless"] = true;
                if (antialiasing_flag)
                    overrides["antialiasing"] = true;
                if (enable_save_eval_images_flag)
                    overrides["enable_save_eval_images"] = true;
                if (skip_intermediate_saving_flag)
                    overrides["skip_intermediate"] = true;
                if (random_flag)
                    overrides["random"] = true;
                if (gut_flag)
                    overrides["gut"] = true;
                if (save_sog_flag)
                    overrides["save_sog"] = true;
                if (enable_sparsity_flag)
                    overrides["enable_sparsity"] = true;

                // Apply overrides to parameters
                params.optimization.params = params.optimization.params.with_overrides(overrides);
            };

            return std::make_tuple(ParseResult::Success, apply_cmd_overrides);

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Unexpected error during argument parsing: {}", e.what()));
        }
    }

    void apply_step_scaling(gs::param::TrainingParameters& params) {
        const float scaler = params.optimization.params.get<float>("steps_scaler", 0.0f);

        if (scaler > 0) {
            LOG_INFO("Scaling training steps by factor: {}", scaler);

            // Build overrides
            nlohmann::json overrides;
            overrides["iterations"] = params.optimization.iterations() * scaler;
            overrides["start_refine"] = params.optimization.params.get<size_t>("start_refine", 500) * scaler;
            overrides["reset_every"] = params.optimization.params.get<size_t>("reset_every", 3000) * scaler;
            overrides["stop_refine"] = params.optimization.params.get<size_t>("stop_refine", 15000) * scaler;
            overrides["refine_every"] = params.optimization.params.get<size_t>("refine_every", 100) * scaler;
            overrides["sh_degree_interval"] = params.optimization.params.get<size_t>("sh_degree_interval", 1000) * scaler;

            // Scale eval and save steps
            auto eval_steps = params.optimization.eval_steps();
            auto save_steps = params.optimization.save_steps();
            scale_steps_vector(eval_steps, scaler);
            scale_steps_vector(save_steps, scaler);

            // Convert back to JSON array
            nlohmann::json eval_json = nlohmann::json::array();
            nlohmann::json save_json = nlohmann::json::array();
            for (auto s : eval_steps)
                eval_json.push_back(s);
            for (auto s : save_steps)
                save_json.push_back(s);
            overrides["eval_steps"] = eval_json;
            overrides["save_steps"] = save_json;

            // Apply all overrides
            params.optimization.params = params.optimization.params.with_overrides(overrides);
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
        // Determine strategy (default if not set)
        if (params->optimization.strategy.empty()) {
            params->optimization.strategy = "default";
        }

        auto opt_params_result = gs::param::read_optim_params_from_json(params->optimization.strategy);
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