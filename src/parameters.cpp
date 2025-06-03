// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.

#include "core/parameters.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace gs {
    namespace param {
        namespace {
            /**
             * @brief Get the path to a configuration file
             * @param filename Name of the configuration file
             * @return std::filesystem::path Full path to the configuration file
             */
            std::filesystem::path get_config_path(const std::string& filename) {
                std::filesystem::path executablePath = std::filesystem::canonical("/proc/self/exe");
                std::filesystem::path parentDir = executablePath.parent_path().parent_path();
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
                    std::variant<size_t, float, int64_t> value;
                    std::string description; // Added for better documentation
                };

                const std::vector<ParamInfo> expected_params = {
                    {"iterations", defaults.iterations, "Total number of training iterations"},
                    {"position_lr_init", defaults.position_lr_init, "Initial learning rate for position updates"},
                    {"feature_lr", defaults.feature_lr, "Learning rate for feature updates"},
                    {"opacity_lr", defaults.opacity_lr, "Learning rate for opacity updates"},
                    {"scaling_lr", defaults.scaling_lr, "Learning rate for scaling updates"},
                    {"rotation_lr", defaults.rotation_lr, "Learning rate for rotation updates"},
                    {"lambda_dssim", defaults.lambda_dssim, "DSSIM loss weight"},
                    {"min_opacity", defaults.min_opacity, "Minimum opacity threshold"},
                    {"densification_interval", defaults.densification_interval, "Interval between densification steps"},
                    {"opacity_reset_interval", defaults.opacity_reset_interval, "Interval for opacity resets"},
                    {"densify_from_iter", defaults.densify_from_iter, "Starting iteration for densification"},
                    {"densify_until_iter", defaults.densify_until_iter, "Ending iteration for densification"},
                    {"densify_grad_threshold", defaults.densify_grad_threshold, "Gradient threshold for densification"},
                    {"opacity_reg", defaults.opacity_reg, "Opacity L1 regularization weight"},
                    {"scale_reg", defaults.scale_reg, "Scale L1 regularization weight"},
                    {"sh_degree", defaults.sh_degree, "Gradient threshold for densification"}};

                // Check all expected parameters
                for (const auto& param : expected_params) {
                    if (!json.contains(param.name)) {
                        missing_in_json.push_back(param.name);
                        all_match = false;
                        continue;
                    }

                    // Compare values based on type
                    std::visit([&](const auto& default_val) {
                        using T = std::decay_t<decltype(default_val)>;
                        if (json[param.name].get<T>() != default_val) {
                            mismatched_values.push_back(param.name);
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
            params.position_lr_init = json["position_lr_init"];
            params.feature_lr = json["feature_lr"];
            params.opacity_lr = json["opacity_lr"];
            params.scaling_lr = json["scaling_lr"];
            params.rotation_lr = json["rotation_lr"];
            params.lambda_dssim = json["lambda_dssim"];
            params.min_opacity = json["min_opacity"];
            params.densification_interval = json["densification_interval"];
            params.opacity_reset_interval = json["opacity_reset_interval"];
            params.densify_from_iter = json["densify_from_iter"];
            params.densify_until_iter = json["densify_until_iter"];
            params.densify_grad_threshold = json["densify_grad_threshold"];
            params.densify_grad_threshold = json["densify_grad_threshold"];

            if (json.contains("opacity_reg")) {
                params.opacity_reg = json["opacity_reg"];
            }
            if (json.contains("scale_reg")) {
                params.scale_reg = json["scale_reg"];
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
    } // namespace param
} // namespace gs
