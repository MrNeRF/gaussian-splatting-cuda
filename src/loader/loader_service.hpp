#pragma once

#include "loader/loader_interface.hpp"
#include "loader/loader_registry.hpp"
#include <expected>
#include <memory>
#include <vector>

namespace gs::loader {

    /**
     * @brief Simple service for loading data files
     *
     * Provides a clean interface for loading any supported format.
     */
    class LoaderService {
    public:
        LoaderService();
        ~LoaderService() = default;

        // Delete copy operations
        LoaderService(const LoaderService&) = delete;
        LoaderService& operator=(const LoaderService&) = delete;

        /**
         * @brief Load data from any supported format
         * @param path File or directory to load
         * @param options Loading options
         * @return LoadResult on success, error string on failure
         */
        std::expected<LoadResult, std::string> load(
            const std::filesystem::path& path,
            const LoadOptions& options = {});

        /**
         * @brief Get information about available loaders
         */
        std::vector<std::string> getAvailableLoaders() const;

        /**
         * @brief Get supported extensions
         */
        std::vector<std::string> getSupportedExtensions() const;

    private:
        std::unique_ptr<DataLoaderRegistry> registry_;
    };

} // namespace gs::loader
