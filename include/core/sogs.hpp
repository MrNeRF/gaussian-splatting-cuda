/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include <expected>
#include <filesystem>
#include <string>

namespace gs {
    namespace core {

        struct SogWriteOptions {
            int iterations = 10;
            bool use_gpu = true;
            std::filesystem::path output_path;
        };

        std::expected<void, std::string> write_sog(
            const SplatData& splat_data,
            const SogWriteOptions& options);

    } // namespace core
} // namespace gs
