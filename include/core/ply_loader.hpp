#pragma once

#include "core/splat_data.hpp"
#include <expected>
#include <filesystem>
#include <string>

namespace gs {

    std::expected<SplatData, std::string> load_ply(const std::filesystem::path& filepath);

} // namespace gs
