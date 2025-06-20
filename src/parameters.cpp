// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.

#include "core/parameters.hpp"
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <variant>
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
                std::filesystem::path parentDir = executablePath.parent_path().parent_path().parent_path();
#else
                std::filesystem::path executablePath = std::filesystem::canonical("/proc/self/exe");
                std::filesystem::path parentDir = executablePath.parent_path().parent_path();
#endif
                return parentDir / "parameter" / filename;
            }

            /**
             * @brief Read and parse a JSON configuration file
             * @param path Path to the JSON file
             * @return nlohmann::json Parsed JSON object
             * @throw std::runtime_error if file doesn't exist or can't be opened
             */
            nlohmann::json read_json_file(const std::filesystem::path& path) {
                if (!std::filesystem::exists(path)) {
                    throw std::runtime_error("Error: " + path.string() + " does not exist!");
                }

                std::ifstream file(path);
                if (!file.is_open()) {
                    throw std::runtime_error("Config file could not be opened: " + path.string());
                }

                std::stringstream buffer;
                buffer << file.rdbuf();
                try {
                    return nlohmann::json::parse(buffer.str());
                } catch (const nlohmann::json::parse_error& e) {
                    throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
                }
            }

            /**
             * @brief Verify optimization parameters between JSON and struct defaults
             *
             * This function performs a comprehensive verification of optimization parameters:
             * 1. Checks if all struct parameters exist in JSON
             * 2. Verifies values match between JSON and struct defaults
             * 3. Identifies unknown parameters in JSON (with warnings)
             * 4. Provides detailed error reporting
             *
             * @param defaults The default parameters from the struct
             * @param json The JSON configuration to verify
             * @param strict If true, unknown parameters are treated as errors
             * @return bool True if verification passed, false otherwise
             */
            bool verify_optimization_parameters(const OptimizationParameters& defaults,
                                                const nlohmann::json& json,
                                                bool strict = false) {
                bool all_match = true;
                std::vector<std::string> missing_in_json;
                std::vector<std::string> unknown_params;
                std::vector<std::string> mismatched_values;

                // Define all expected parameters and their types
                struct ParamInfo {
                    std::string name;
                    std::variant<int, size_t, float, int64_t, std::string, bool> value;
                    std::string description; // Added for better documentation
                };

                const std::vector<ParamInfo> expected_params = {
                    {"iterations", defaults.iterations, "Total number of training iterations"},
                    {"means_lr", defaults.means_lr, "Initial learning rate for position updates"},
                    {"shs_lr", defaults.shs_lr, "Learning rate for spherical harmonics updates"},
                    {"opacity_lr", defaults.opacity_lr, "Learning rate for opacity updates"},
                    {"scaling_lr", defaults.scaling_lr, "Learning rate for scaling updates"},
                    {"rotation_lr", defaults.rotation_lr, "Learning rate for rotation updates"},
                    {"lambda_dssim", defaults.lambda_dssim, "DSSIM loss weight"},
                    {"min_opacity", defaults.min_opacity, "Minimum opacity threshold"},
                    {"refine_every", defaults.refine_every, "Interval between densification steps"},
                    {"start_refine", defaults.start_refine, "Starting iteration for densification"},
                    {"stop_refine", defaults.stop_refine, "Ending iteration for densification"},
                    {"grad_threshold", defaults.grad_threshold, "Gradient threshold for densification"},
                    {"opacity_reg", defaults.opacity_reg, "Opacity L1 regularization weight"},
                    {"scale_reg", defaults.scale_reg, "Scale L1 regularization weight"},
                    {"init_opacity", defaults.init_opacity, "Initial opacity value for new Gaussians"},
                    {"init_scaling", defaults.init_scaling, "Initial scaling value for new Gaussians"},
                    {"sh_degree", defaults.sh_degree, "Spherical harmonics degree"},
                    {"max_cap", defaults.max_cap, "Maximum number of Gaussians for MCMC strategy"},
                    {"render_mode", defaults.render_mode, "Render mode: RGB, D, ED, RGB_D, RGB_ED"},
                    {"enable_eval", defaults.enable_eval, "Enable evaluation during training"},
                    {"enable_save_eval_images", defaults.enable_save_eval_images, "Save images during evaluation"},
                    {"use_bilateral_grid", defaults.use_bilateral_grid, "Enable bilateral grid for appearance modeling"},
                    {"bilateral_grid_X", defaults.bilateral_grid_X, "Bilateral grid X dimension"},
                    {"bilateral_grid_Y", defaults.bilateral_grid_Y, "Bilateral grid Y dimension"},
                    {"bilateral_grid_W", defaults.bilateral_grid_W, "Bilateral grid W dimension"},
                    {"bilateral_grid_lr", defaults.bilateral_grid_lr, "Learning rate for bilateral grid"},
                    {"tv_loss_weight", defaults.tv_loss_weight, "Weight for total variation loss"},
                    {"steps_scaler", defaults.steps_scaler, "Scales the training steps and values"},
                    {"sh_degree_interval", defaults.sh_degree_interval, "Interval for increasing SH degree"},
                    {"selective_adam", defaults.selective_adam, "Selective Adam optimizer flag"}};

                // Check all expected parameters
                for (const auto& param : expected_params) {
                    if (!json.contains(param.name)) {
                        // Skip eval_steps and save_steps as they are handled separately
                        if (param.name != "eval_steps" && param.name != "save_steps") {
                            missing_in_json.push_back(param.name);
                            all_match = false;
                        }
                        continue;
                    }

                    // Compare values based on type
                    std::visit([&](const auto& default_val) {
                        using T = std::decay_t<decltype(default_val)>;
                        try {
                            if (json[param.name].get<T>() != default_val) {
                                mismatched_values.push_back(param.name);
                                all_match = false;
                            }
                        } catch (...) {
                            // Type mismatch
                            mismatched_values.push_back(param.name + " (type mismatch)");
                            all_match = false;
                        }
                    },
                               param.value);
                }

                // Check for any unknown parameters in JSON
                for (const auto& [key, value] : json.items()) {
                    bool found = false;
                    for (const auto& param : expected_params) {
                        if (key == param.name) {
                            found = true;
                            break;
                        }
                    }
                    if (key == "eval_steps" || key == "save_steps") {
                        found = true;
                    }
                    if (!found) {
                        unknown_params.push_back(key);
                        if (strict) {
                            all_match = false;
                        }
                    }
                }

                // Print report
                if (!all_match || !unknown_params.empty()) {
                    std::cerr << "\nParameter verification report:\n";

                    if (!mismatched_values.empty()) {
                        std::cerr << "\nMismatched values:\n";
                        for (const auto& param : mismatched_values) {
                            std::cerr << "  - " << param << ": JSON=" << json[param];
                            // Find and print the default value and description
                            for (const auto& p : expected_params) {
                                if (p.name == param) {
                                    std::cerr << ", Default=";
                                    std::visit([&](const auto& val) {
                                        std::cerr << val;
                                    },
                                               p.value);
                                    std::cerr << " (" << p.description << ")\n";
                                    break;
                                }
                            }
                        }
                    }

                    if (!missing_in_json.empty()) {
                        std::cerr << "\nParameters in struct but not in JSON:\n";
                        for (const auto& param : missing_in_json) {
                            // Find and print the description
                            for (const auto& p : expected_params) {
                                if (p.name == param) {
                                    std::cerr << "  - " << param << " (" << p.description << ")\n";
                                    break;
                                }
                            }
                        }
                    }

                    if (!unknown_params.empty()) {
                        std::cerr << "\nUnknown parameters in JSON (will be ignored):\n";
                        for (const auto& param : unknown_params) {
                            std::cerr << "  - " << param << "\n";
                        }
                    }
                } else {
                    std::cout << "Parameter verification passed successfully!\n";
                }

                return all_match;
            }
        } // namespace

        /**
         * @brief Read optimization parameters from JSON file
         * @return OptimizationParameters The parsed parameters
         * @throw std::runtime_error if file doesn't exist or can't be parsed
         */
        OptimizationParameters read_optim_params_from_json() {
            auto json = read_json_file(get_config_path("optimization_params.json"));

            // Create default parameters for verification
            OptimizationParameters defaults;

            // Verify parameters before reading
            verify_optimization_parameters(defaults, json);

            OptimizationParameters params;
            params.iterations = json["iterations"];
            params.means_lr = json["means_lr"];
            params.shs_lr = json["shs_lr"];
            params.opacity_lr = json["opacity_lr"];
            params.scaling_lr = json["scaling_lr"];
            params.rotation_lr = json["rotation_lr"];
            params.lambda_dssim = json["lambda_dssim"];
            params.min_opacity = json["min_opacity"];
            params.refine_every = json["refine_every"];
            params.start_refine = json["start_refine"];
            params.stop_refine = json["stop_refine"];
            params.grad_threshold = json["grad_threshold"];
            params.sh_degree = json["sh_degree"];

            if (json.contains("opacity_reg")) {
                params.opacity_reg = json["opacity_reg"];
            }
            if (json.contains("scale_reg")) {
                params.scale_reg = json["scale_reg"];
            }
            if (json.contains("init_opacity")) {
                params.init_opacity = json["init_opacity"];
            }
            if (json.contains("init_scaling")) {
                params.init_scaling = json["init_scaling"];
            }
            if (json.contains("max_cap")) {
                params.max_cap = json["max_cap"];
            }

            // Handle render mode
            if (json.contains("render_mode")) {
                std::string mode = json["render_mode"];
                // Validate render mode
                if (mode == "RGB" || mode == "D" || mode == "ED" ||
                    mode == "RGB_D" || mode == "RGB_ED") {
                    params.render_mode = mode;
                } else {
                    std::cerr << "Warning: Invalid render mode '" << mode
                              << "' in JSON. Using default 'RGB'\n";
                }
            }

            if (json.contains("eval_steps")) {
                params.eval_steps.clear();
                for (const auto& step : json["eval_steps"]) {
                    params.eval_steps.push_back(step.get<size_t>());
                }
            }

            if (json.contains("save_steps")) {
                params.save_steps.clear();
                for (const auto& step : json["save_steps"]) {
                    params.save_steps.push_back(step.get<size_t>());
                }
            }

            if (json.contains("enable_eval")) {
                params.enable_eval = json["enable_eval"];
            }
            if (json.contains("enable_save_eval_images")) {
                params.enable_save_eval_images = json["enable_save_eval_images"];
            }
            if (json.contains("use_bilateral_grid")) {
                params.use_bilateral_grid = json["use_bilateral_grid"];
            }
            if (json.contains("bilateral_grid_X")) {
                params.bilateral_grid_X = json["bilateral_grid_X"];
            }
            if (json.contains("bilateral_grid_Y")) {
                params.bilateral_grid_Y = json["bilateral_grid_Y"];
            }
            if (json.contains("bilateral_grid_W")) {
                params.bilateral_grid_W = json["bilateral_grid_W"];
            }
            if (json.contains("bilateral_grid_lr")) {
                params.bilateral_grid_lr = json["bilateral_grid_lr"];
            }
            if (json.contains("tv_loss_weight")) {
                params.tv_loss_weight = json["tv_loss_weight"];
            }
            if (json.contains("steps_scaler")) {
                params.steps_scaler = json["steps_scaler"];
            }
            if (json.contains("sh_degree_interval")) {
                params.sh_degree_interval = json["sh_degree_interval"];
            }
            if (json.contains("selective_adam")) {
                params.selective_adam = json["selective_adam"];
            }
            return params;
        }

        /**
         * @brief Read model parameters from JSON file
         * @return ModelParameters The parsed parameters
         * @throw std::runtime_error if file doesn't exist or can't be parsed
         */
        DatasetConfig read_model_params_from_json() {
            auto json = read_json_file(get_config_path("model_params.json"));
            DatasetConfig params;
            if (json.contains("source_path"))
                params.data_path = json["source_path"].get<std::string>();
            if (json.contains("output_path"))
                params.output_path = json["output_path"].get<std::string>();
            if (json.contains("images"))
                params.images = json["images"];
            if (json.contains("resolution"))
                params.resolution = json["resolution"];
            return params;
        }

        /**
         * @brief Save full training parameters (dataset + optimization) to JSON
         * @param params The full training parameters
         * @param output_path Path to the output directory
         */
        void save_training_parameters_to_json(const TrainingParameters& params,
                                              const std::filesystem::path& output_path) {
            nlohmann::json json;

            // Dataset configuration
            json["dataset"]["data_path"] = params.dataset.data_path.string();
            json["dataset"]["output_path"] = params.dataset.output_path.string();
            json["dataset"]["images"] = params.dataset.images;
            json["dataset"]["resolution"] = params.dataset.resolution;
            json["dataset"]["test_every"] = params.dataset.test_every;

            // Optimization configuration
            nlohmann::json opt_json;
            opt_json["iterations"] = params.optimization.iterations;
            opt_json["means_lr"] = params.optimization.means_lr;
            opt_json["shs_lr"] = params.optimization.shs_lr;
            opt_json["opacity_lr"] = params.optimization.opacity_lr;
            opt_json["scaling_lr"] = params.optimization.scaling_lr;
            opt_json["rotation_lr"] = params.optimization.rotation_lr;
            opt_json["lambda_dssim"] = params.optimization.lambda_dssim;
            opt_json["min_opacity"] = params.optimization.min_opacity;
            opt_json["refine_every"] = params.optimization.refine_every;
            opt_json["start_refine"] = params.optimization.start_refine;
            opt_json["stop_refine"] = params.optimization.stop_refine;
            opt_json["grad_threshold"] = params.optimization.grad_threshold;
            opt_json["sh_degree"] = params.optimization.sh_degree;
            opt_json["opacity_reg"] = params.optimization.opacity_reg;
            opt_json["scale_reg"] = params.optimization.scale_reg;
            opt_json["init_opacity"] = params.optimization.init_opacity;
            opt_json["init_scaling"] = params.optimization.init_scaling;
            opt_json["max_cap"] = params.optimization.max_cap;
            opt_json["render_mode"] = params.optimization.render_mode;
            opt_json["eval_steps"] = params.optimization.eval_steps;
            opt_json["save_steps"] = params.optimization.save_steps;
            opt_json["enable_eval"] = params.optimization.enable_eval;
            opt_json["enable_save_eval_images"] = params.optimization.enable_save_eval_images;
            opt_json["use_bilateral_grid"] = params.optimization.use_bilateral_grid;
            opt_json["bilateral_grid_X"] = params.optimization.bilateral_grid_X;
            opt_json["bilateral_grid_Y"] = params.optimization.bilateral_grid_Y;
            opt_json["bilateral_grid_W"] = params.optimization.bilateral_grid_W;
            opt_json["bilateral_grid_lr"] = params.optimization.bilateral_grid_lr;
            opt_json["tv_loss_weight"] = params.optimization.tv_loss_weight;
            opt_json["steps_scaler"] = params.optimization.steps_scaler;
            opt_json["sh_degree_interval"] = params.optimization.sh_degree_interval;
            opt_json["selective_adam"] = params.optimization.selective_adam;

            json["optimization"] = opt_json;

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
                throw std::runtime_error("Could not open file for writing: " + filepath.string());
            }

            file << json.dump(4); // Pretty print with 4 spaces
            file.close();

            std::cout << "Saved training configuration to: " << filepath << std::endl;
        }

    } // namespace param
} // namespace gs