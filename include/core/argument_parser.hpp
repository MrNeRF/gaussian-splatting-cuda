// Copyright (c) 2023 Janusch Patas.
#pragma once

#include "core/parameters.hpp"

namespace gs {
    namespace args {
        /**
         * @brief Parse command-line arguments and load parameters from JSON
         * @param argc Number of arguments
         * @param argv Array of argument strings (const-correct)
         * @return TrainingParameters Populated parameter structure
         * @throws std::runtime_error if argument parsing fails
         */
        gs::param::TrainingParameters parse_args_and_params(int argc, const char* const argv[]);
    } // namespace args
} // namespace gs
