// Copyright (c) 2023 Janusch Patas.
#pragma once

#include "core/parameters.hpp"
#include <expected>

namespace gs {
    namespace args {
        /**
         * @brief Parse command-line arguments and load parameters from JSON
         * @param argc Number of arguments
         * @param argv Array of argument strings
         * @return Expected TrainingParameters or error message
         */
        std::expected<gs::param::TrainingParameters, std::string>
        parse_args_and_params(int argc, char* argv[]);
    } // namespace args
} // namespace gs