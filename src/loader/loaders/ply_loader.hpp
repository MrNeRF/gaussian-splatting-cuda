#pragma once

#include "loader/loader_interface.hpp"

namespace gs::loader {

    /**
     * @brief Loader for PLY point cloud files
     */
    class PLYLoader : public IDataLoader {
    public:
        PLYLoader() = default;
        ~PLYLoader() override = default;

        std::expected<LoadResult, std::string> load(
            const std::filesystem::path& path,
            const LoadOptions& options = {}) override;

        bool canLoad(const std::filesystem::path& path) const override;
        std::string name() const override;
        std::vector<std::string> supportedExtensions() const override;
        int priority() const override;
    };

} // namespace gs::loader
