/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <expected>
#include <filesystem>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace gs {
    namespace param {
        // New parameter container that wraps JSON data
        class StrategyParameters {
        public:
            StrategyParameters() = default;

            StrategyParameters(nlohmann::json common, nlohmann::json strategy_specific)
                : common_(std::move(common))
                , specific_(std::move(strategy_specific)) {}

            // Type-safe getter that checks strategy-specific first, then common
            template<typename T>
            T get(const std::string& key, T default_value = T{}) const {
                // First check strategy-specific
                if (auto it = specific_.find(key); it != specific_.end()) {
                    try {
                        return it->get<T>();
                    } catch (const nlohmann::json::type_error&) {
                        // Fall through to check common
                    }
                }
                // Then check common
                if (auto it = common_.find(key); it != common_.end()) {
                    try {
                        return it->get<T>();
                    } catch (const nlohmann::json::type_error&) {
                        // Fall through to return default
                    }
                }
                return default_value;
            }

            // Check if parameter exists
            bool has(const std::string& key) const {
                return specific_.contains(key) || common_.contains(key);
            }

            // Get strategy name
            std::string strategy_name() const {
                return get<std::string>("strategy", "default");
            }

            // For backward compatibility and special access
            const nlohmann::json& common() const { return common_; }
            const nlohmann::json& specific() const { return specific_; }

            // Create a mutable copy with overrides
            StrategyParameters with_overrides(const nlohmann::json& overrides) const {
                auto new_common = common_;
                auto new_specific = specific_;

                // Apply overrides to both common and specific
                for (const auto& [key, value] : overrides.items()) {
                    if (specific_.contains(key)) {
                        new_specific[key] = value;
                    } else {
                        new_common[key] = value;
                    }
                }

                return StrategyParameters(new_common, new_specific);
            }

        private:
            nlohmann::json common_;
            nlohmann::json specific_;
        };

        struct OptimizationParameters {
            std::string strategy;
            StrategyParameters params;

            // Convenience getters for commonly accessed parameters
            size_t iterations() const { return params.get<size_t>("iterations", 30000); }
            float lambda_dssim() const { return params.get<float>("lambda_dssim", 0.2f); }
            int sh_degree() const { return params.get<int>("sh_degree", 3); }
            std::string render_mode() const { return params.get<std::string>("render_mode", "RGB"); }
            bool enable_eval() const { return params.get<bool>("enable_eval", false); }
            bool headless() const { return params.get<bool>("headless", false); }
            bool antialiasing() const { return params.get<bool>("antialiasing", false); }
            bool gut() const { return params.get<bool>("gut", false); }
            bool rc() const { return params.get<bool>("rc", false); }
            bool save_sog() const { return params.get<bool>("save_sog", false); }
            int sog_iterations() const { return params.get<int>("sog_iterations", 10); }
            bool enable_sparsity() const { return params.get<bool>("enable_sparsity", false); }
            int sparsify_steps() const { return params.get<int>("sparsify_steps", 15000); }
            float init_rho() const { return params.get<float>("init_rho", 0.0005f); }
            float prune_ratio() const { return params.get<float>("prune_ratio", 0.6f); }

            // Get eval/save steps
            std::vector<size_t> eval_steps() const {
                std::vector<size_t> steps;
                try {
                    auto json_steps = params.get<nlohmann::json>("eval_steps", nlohmann::json::array());
                    for (const auto& s : json_steps) {
                        steps.push_back(s.get<size_t>());
                    }
                } catch (...) {
                    steps = {7000, 30000}; // defaults
                }
                return steps;
            }

            std::vector<size_t> save_steps() const {
                std::vector<size_t> steps;
                try {
                    auto json_steps = params.get<nlohmann::json>("save_steps", nlohmann::json::array());
                    for (const auto& s : json_steps) {
                        steps.push_back(s.get<size_t>());
                    }
                } catch (...) {
                    steps = {7000, 30000}; // defaults
                }
                return steps;
            }

            // Convert back to old format for compatibility
            nlohmann::json to_json() const;

            // Factory method
            static std::expected<OptimizationParameters, std::string>
            from_strategy(const std::string& strategy_name);
        };

        struct DatasetConfig {
            std::filesystem::path data_path = "";
            std::filesystem::path output_path = "";
            std::filesystem::path project_path = "";
            std::string images = "images";
            int resize_factor = -1;
            int test_every = 8;
            std::vector<std::string> timelapse_images = {};
            int timelapse_every = 50;
        };

        struct TrainingParameters {
            DatasetConfig dataset;
            OptimizationParameters optimization;

            // Viewer mode specific
            std::filesystem::path ply_path = "";

            // Optional PLY splat file for initialization
            std::optional<std::string> init_ply = std::nullopt;
        };

        // Functions
        std::expected<OptimizationParameters, std::string> read_optim_params_from_json(const std::string strategy);

        std::expected<void, std::string> save_training_parameters_to_json(
            const TrainingParameters& params,
            const std::filesystem::path& output_path);
    } // namespace param
} // namespace gs