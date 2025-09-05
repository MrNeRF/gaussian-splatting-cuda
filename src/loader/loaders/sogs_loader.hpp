/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "loader/loader_interface.hpp"

namespace gs::loader {

    /**
     * @brief Loader for SOG compressed splat files
     */
    class SogLoader : public IDataLoader {
    public:
        SogLoader() = default;
        ~SogLoader() override = default;

        std::expected<LoadResult, std::string> load(
            const std::filesystem::path& path,
            const LoadOptions& options = {}) override;

        bool canLoad(const std::filesystem::path& path) const override;
        std::string name() const override;
        std::vector<std::string> supportedExtensions() const override;
        int priority() const override;
    };

} // namespace gs::loader
