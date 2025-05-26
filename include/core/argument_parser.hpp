// Copyright (c) 2023 Janusch Patas.
#pragma once

#include "core/parameters.hpp"
#include <string>
#include <vector>

namespace gs {
    namespace args {
        /**
         * @brief Parse command line arguments and populate model and optimization parameters
         *
         * @param args Vector of command line arguments (including program name as first element)
         * @param modelParams Reference to ModelParameters struct to be populated
         * @param optimParams Reference to OptimizationParameters struct to be populated
         * @return int 0 on success, negative value on error or help/completion requested
         */
        int parse_arguments(const std::vector<std::string>& args,
                            gs::param::ModelParameters& modelParams,
                            gs::param::OptimizationParameters& optimParams);

        /**
         * @brief Convert argc/argv to vector of strings for easier handling
         *
         * @param argc Argument count from main()
         * @param argv Argument values from main()
         * @return std::vector<std::string> Vector containing all arguments
         */
        std::vector<std::string> convert_args(int argc, char* argv[]);
    } // namespace args
} // namespace gs