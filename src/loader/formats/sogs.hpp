/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include <expected>
#include <filesystem>
#include <string>

namespace gs::loader {

    std::expected<SplatData, std::string> load_sog(const std::filesystem::path& filepath);

} // namespace gs::loader
