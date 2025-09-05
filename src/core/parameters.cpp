/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include <chrono>
#include <ctime>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <print>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace gs {
    namespace param {
        namespace {
            /**
             * @brief Get the path to a configuration file
             * @param filename Name of the configuration file
             * @return std::filesystem::path Full path to the configuration file
             */
            std::filesystem::path get_config_path(const std::string& filename) {
#ifdef _WIN32
                char executablePathWindows[MAX_PATH];
                GetModuleFileNameA(nullptr, executablePathWindows, MAX_PATH);
                std::filesystem::path executablePath = std::filesystem::path(executablePathWindows);
                std::filesystem::path searchDir = executablePath.parent_path();
                while (!searchDir.empty() && !std::filesystem::exists(searchDir / "parameter" / filename)) {
                    searchDir = searchDir.parent_path();
                }

                if (searchDir.empty()) {
                    throw std::runtime_error("could not find " + (std::filesystem::path("parameter") / filename).string());
                }
#else
                std::filesystem::path executablePath = std::filesystem::canonical("/proc/self/exe");
                std::filesystem::path searchDir = executablePath.parent_path().parent_path();
#endif
                return searchDir / "parameter" / filename;
            }

            /**
             * @brief Read and parse a JSON configuration file
             * @param path Path to the JSON file
             * @return Expected JSON object or error message
             */
            std::expected<nlohmann::json, std::string> read_json_file(const std::filesystem::path& path) {
                if (!std::filesystem::exists(path)) {
                    return std::unexpected(std::format("Configuration file does not exist: {}", path.string()));
                }

                std::ifstream file(path);
                if (!file.is_open()) {
                    return std::unexpected(std::format("Could not open configuration file: {}", path.string()));
                }

                try {
                    std::stringstream buffer;
                    buffer << file.rdbuf();
                    return nlohmann::json::parse(buffer.str());
                } catch (const nlohmann::json::parse_error& e) {
                    return std::unexpected(std::format("JSON parsing error in {}: {}", path.string(), e.what()));
                }
            }
        } // namespace

        nlohmann::json OptimizationParameters::to_json() const {
            nlohmann::json opt_json;

            // Add all parameters from both common and specific
            // Common parameters
            opt_json["iterations"] = iterations();
            opt_json["sh_degree_interval"] = params.get<size_t>("sh_degree_interval", 1000);
            opt_json["lambda_dssim"] = lambda_dssim();
            opt_json["sh_degree"] = sh_degree();
            opt_json["render_mode"] = render_mode();
            opt_json["enable_eval"] = enable_eval();
            opt_json["enable_save_eval_images"] = params.get<bool>("enable_save_eval_images", true);
            opt_json["skip_intermediate"] = params.get<bool>("skip_intermediate", false);
            opt_json["use_bilateral_grid"] = params.get<bool>("use_bilateral_grid", false);
            opt_json["bilateral_grid_X"] = params.get<int>("bilateral_grid_X", 16);
            opt_json["bilateral_grid_Y"] = params.get<int>("bilateral_grid_Y", 16);
            opt_json["bilateral_grid_W"] = params.get<int>("bilateral_grid_W", 8);
            opt_json["bilateral_grid_lr"] = params.get<float>("bilateral_grid_lr", 0.002f);
            opt_json["tv_loss_weight"] = params.get<float>("tv_loss_weight", 10.0f);
            opt_json["steps_scaler"] = params.get<float>("steps_scaler", 0.0f);
            opt_json["antialiasing"] = antialiasing();
            opt_json["random"] = params.get<bool>("random", false);
            opt_json["init_num_pts"] = params.get<int>("init_num_pts", 100000);
            opt_json["init_extent"] = params.get<float>("init_extent", 3.0f);
            opt_json["save_sog"] = save_sog();
            opt_json["sog_iterations"] = sog_iterations();
            opt_json["enable_sparsity"] = enable_sparsity();
            opt_json["sparsify_steps"] = sparsify_steps();
            opt_json["init_rho"] = init_rho();
            opt_json["prune_ratio"] = prune_ratio();
            opt_json["pose_optimization"] = params.get<std::string>("pose_optimization", "none");
            opt_json["strategy"] = strategy;

            // Strategy-specific parameters
            opt_json["means_lr"] = params.get<float>("means_lr", 0.00016f);
            opt_json["shs_lr"] = params.get<float>("shs_lr", 0.0025f);
            opt_json["opacity_lr"] = params.get<float>("opacity_lr", 0.05f);
            opt_json["scaling_lr"] = params.get<float>("scaling_lr", 0.005f);
            opt_json["rotation_lr"] = params.get<float>("rotation_lr", 0.001f);
            opt_json["min_opacity"] = params.get<float>("min_opacity", 0.005f);
            opt_json["refine_every"] = params.get<size_t>("refine_every", 100);
            opt_json["start_refine"] = params.get<size_t>("start_refine", 500);
            opt_json["stop_refine"] = params.get<size_t>("stop_refine", 15000);
            opt_json["grad_threshold"] = params.get<float>("grad_threshold", 0.0002f);
            opt_json["opacity_reg"] = params.get<float>("opacity_reg", 0.0f);
            opt_json["scale_reg"] = params.get<float>("scale_reg", 0.0f);
            opt_json["init_opacity"] = params.get<float>("init_opacity", 0.1f);
            opt_json["init_scaling"] = params.get<float>("init_scaling", 1.0f);
            opt_json["max_cap"] = params.get<int>("max_cap", 1000000);

            // Default-specific
            if (strategy == "default") {
                opt_json["prune_opacity"] = params.get<float>("prune_opacity", 0.005f);
                opt_json["grow_scale3d"] = params.get<float>("grow_scale3d", 0.01f);
                opt_json["grow_scale2d"] = params.get<float>("grow_scale2d", 0.05f);
                opt_json["prune_scale3d"] = params.get<float>("prune_scale3d", 0.1f);
                opt_json["prune_scale2d"] = params.get<float>("prune_scale2d", 0.15f);
                opt_json["reset_every"] = params.get<size_t>("reset_every", 3000);
                opt_json["pause_refine_after_reset"] = params.get<size_t>("pause_refine_after_reset", 0);
                opt_json["revised_opacity"] = params.get<bool>("revised_opacity", false);
            }

            // Add eval and save steps
            opt_json["eval_steps"] = eval_steps();
            opt_json["save_steps"] = save_steps();

            return opt_json;
        }

        std::expected<OptimizationParameters, std::string>
        OptimizationParameters::from_strategy(const std::string& strategy_name) {
            // Load the single JSON file
            auto json_path = get_config_path("optimization_params.json");
            auto json_result = read_json_file(json_path);
            if (!json_result) {
                return std::unexpected(json_result.error());
            }

            auto& j = *json_result;

            // Validate structure
            if (!j.contains("common")) {
                return std::unexpected("Invalid JSON structure: missing 'common' section");
            }
            if (!j.contains("strategies")) {
                return std::unexpected("Invalid JSON structure: missing 'strategies' section");
            }
            if (!j["strategies"].contains(strategy_name)) {
                return std::unexpected(std::format("Unknown strategy '{}' in configuration", strategy_name));
            }

            OptimizationParameters result;
            result.strategy = strategy_name;
            result.params = StrategyParameters(
                j["common"],
                j["strategies"][strategy_name]);

            return result;
        }

        /**
         * @brief Read optimization parameters from JSON file
         * @param[in] strategy Optimization strategy to load parameters for
         * @return Expected OptimizationParameters or error message
         */
        std::expected<OptimizationParameters, std::string> read_optim_params_from_json(const std::string strategy) {
            return OptimizationParameters::from_strategy(strategy);
        }

        /**
         * @brief Save full training parameters (dataset + optimization) to JSON
         * @param params The full training parameters
         * @param output_path Path to the output directory
         * @return Expected void or error message
         */
        std::expected<void, std::string> save_training_parameters_to_json(
            const TrainingParameters& params,
            const std::filesystem::path& output_path) {

            try {
                nlohmann::json json;

                // Dataset configuration
                json["dataset"]["data_path"] = params.dataset.data_path.string();
                json["dataset"]["output_path"] = params.dataset.output_path.string();
                json["dataset"]["images"] = params.dataset.images;
                json["dataset"]["resize_factor"] = params.dataset.resize_factor;
                json["dataset"]["test_every"] = params.dataset.test_every;

                // Optimization configuration
                json["optimization"] = params.optimization.to_json();

                // Add timestamp
                auto now = std::chrono::system_clock::now();
                auto time_t = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
                json["timestamp"] = ss.str();

                // Save to file
                std::filesystem::path filepath = output_path / "training_config.json";
                std::ofstream file(filepath);
                if (!file.is_open()) {
                    return std::unexpected(std::format("Could not open file for writing: {}", filepath.string()));
                }

                file << json.dump(4); // Pretty print with 4 spaces
                file.close();

                std::println("Saved training configuration to: {}", filepath.string());
                return {};

            } catch (const std::exception& e) {
                return std::unexpected(std::format("Error saving training parameters: {}", e.what()));
            }
        }

    } // namespace param
} // namespace gs