/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "loader/loader.hpp"
#include "core/logger.hpp"
#include "loader/filesystem_utils.hpp"
#include "loader_service.hpp"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <system_error>

namespace gs::loader {

    namespace {
        // Implementation class that hides all internal details
        class LoaderImpl : public Loader {
        public:
            LoaderImpl() : service_(std::make_unique<LoaderService>()) {
                LOG_TRACE("LoaderImpl created");
            }

            std::expected<LoadResult, std::string> load(
                const std::filesystem::path& path,
                const LoadOptions& options) override {

                LOG_DEBUG("Loading from path: {}", path.string());
                // Just delegate to the service
                return service_->load(path, options);
            }

            bool canLoad(const std::filesystem::path& path) const override {
                // Check if any registered loader can handle this path
                // We would need to expose this functionality from the service
                // For now, let's check common extensions
                if (!safe_exists(path)) {
                    LOG_TRACE("Path does not exist: {}", path.string());
                    return false;
                }

                auto ext = path.extension().string();
                auto extensions = service_->getSupportedExtensions();
                bool can_load = std::find(extensions.begin(), extensions.end(), ext) != extensions.end();
                LOG_TRACE("Can load {}: {}", path.string(), can_load);
                return can_load;
            }

            std::vector<std::string> getSupportedFormats() const override {
                return service_->getAvailableLoaders();
            }

            std::vector<std::string> getSupportedExtensions() const override {
                return service_->getSupportedExtensions();
            }

        private:
            std::unique_ptr<LoaderService> service_;
        };
    } // namespace

    // Factory method implementation
    std::unique_ptr<Loader> Loader::create() {
        LOG_DEBUG("Creating Loader instance");
        return std::make_unique<LoaderImpl>();
    }

    bool Loader::isDatasetPath(const std::filesystem::path& path) {
        if (!safe_exists(path)) {
            LOG_TRACE("Path does not exist for dataset check: {}", path.string());
            return false;
        }

        if (!safe_is_directory(path)) {
            auto ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            // JSON files might be datasets (transforms.json)
            if (ext == ".json") {
                LOG_TRACE("JSON file detected, treating as potential dataset: {}", path.string());
                return true;
            }

            // PLY files are definitely not datasets
            LOG_TRACE("Non-dataset file detected: {}", path.string());
            return false;
        }

        // Check for COLMAP markers in any standard location
        auto colmap_paths = get_colmap_search_paths(path);
        const std::vector<std::string> colmap_markers = {
            "cameras.bin", "cameras.txt", "images.bin", "images.txt"};

        for (const auto& marker : colmap_markers) {
            if (!find_file_in_paths(colmap_paths, marker).empty()) {
                LOG_TRACE("COLMAP dataset detected at: {}", path.string());
                return true;
            }
        }

        // Blender/NeRF markers
        if (safe_exists(path / "transforms.json") ||
            safe_exists(path / "transforms_train.json")) {
            LOG_TRACE("Blender/NeRF dataset detected at: {}", path.string());
            return true;
        }

        LOG_TRACE("No dataset markers found in directory: {}", path.string());
        return false;
    }

    // Static method to determine dataset type
    DatasetType Loader::getDatasetType(const std::filesystem::path& path) {
        if (!safe_exists(path)) {
            return DatasetType::Unknown;
        }

        if (!safe_is_directory(path)) {
            auto ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".json") {
                return DatasetType::Transforms;
            }
            return DatasetType::Unknown;
        }

        // Check for COLMAP markers
        auto colmap_paths = get_colmap_search_paths(path);
        const std::vector<std::string> colmap_markers = {
            "cameras.bin", "cameras.txt", "images.bin", "images.txt"};

        for (const auto& marker : colmap_markers) {
            if (!find_file_in_paths(colmap_paths, marker).empty()) {
                return DatasetType::COLMAP;
            }
        }

        // Check for Transforms markers
        if (safe_exists(path / "transforms.json") ||
            safe_exists(path / "transforms_train.json")) {
            return DatasetType::Transforms;
        }

        return DatasetType::Unknown;
    }

} // namespace gs::loader
