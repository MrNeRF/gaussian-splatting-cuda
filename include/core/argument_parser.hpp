/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include <expected>
#include <memory>

namespace gs {
    namespace args {
        /**
         * @brief Parse command-line arguments and load parameters from JSON
         * @param argc Number of arguments
         * @param argv Array of argument strings (const-correct)
         * @return Expected TrainingParameters or error message
         */
        std::expected<std::unique_ptr<param::TrainingParameters>, std::string>
        parse_args_and_params(int argc, const char* const argv[]);
    } // namespace args
} // namespace gs