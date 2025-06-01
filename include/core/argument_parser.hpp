// Copyright (c) 2023 Janusch Patas.
#pragma once

#include "core/parameters.hpp"
#include <string>
#include <vector>

namespace gs {
    namespace args {
        /**
         * @brief Convert command-line arguments to a vector of strings
         * @param argc Number of arguments
         * @param argv Array of argument strings
         * @return std::vector<std::string> Vector of argument strings
         */
        std::vector<std::string> convert_args(int argc, char* argv[]);

        /**
         * @brief Parse command-line arguments and populate parameter structs
         * @param args Vector of argument strings
         * @param params Reference to TrainingParameters struct to be populated
         * @return int 0 on success, negative value on error
         */
        int parse_arguments(const std::vector<std::string>& args,
                            gs::param::TrainingParameters& params);
    } // namespace args
} // namespace gs