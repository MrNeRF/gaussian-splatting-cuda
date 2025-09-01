/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "loader/loader.hpp"
#include "core/logger.hpp"
#include "loader_service.hpp"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <system_error>

namespace gs::loader {

    namespace {
        // Helper function to safely check file existence
        bool safe_exists(const std::filesystem::path& path) {
            std::error_code ec;
            bool exists = std::filesystem::exists(path, ec);
            if (ec) {
                LOG_TRACE("Cannot access path {}: {}", path.string(), ec.message());
                return false;
            }
            return exists;
        }

        // Helper function to safely check if path is directory
        bool safe_is_directory(const std::filesystem::path& path) {
            std::error_code ec;
            bool is_dir = std::filesystem::is_directory(path, ec);
            if (ec) {
                LOG_TRACE("Cannot check directory status for {}: {}", path.string(), ec.message());
                return false;
            }
            return is_dir;
        }

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
                // Check if it's a transforms file by trying to read it
                // For now, assume .json files in the context are datasets
                LOG_TRACE("JSON file detected, treating as potential dataset: {}", path.string());
                return true;
            }

            // PLY files are definitely not datasets
            LOG_TRACE("Non-dataset file detected: {}", path.string());
            return false;
        }

        // Check for dataset markers - use safe_exists for all checks
        namespace fs = std::filesystem;

        // COLMAP markers - check both binary and text formats
        bool has_colmap =
            safe_exists(path / "sparse" / "0" / "cameras.bin") ||
            safe_exists(path / "sparse" / "cameras.bin") ||
            safe_exists(path / "sparse" / "0" / "cameras.txt") ||
            safe_exists(path / "sparse" / "cameras.txt");

        if (has_colmap) {
            LOG_TRACE("COLMAP dataset detected at: {}", path.string());
            return true;
        }

        // Blender/NeRF markers
        bool has_transforms =
            safe_exists(path / "transforms.json") ||
            safe_exists(path / "transforms_train.json");

        if (has_transforms) {
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

        // Check for COLMAP markers first
        if (safe_exists(path / "sparse" / "0" / "cameras.bin") ||
            safe_exists(path / "sparse" / "cameras.bin") ||
            safe_exists(path / "sparse" / "0" / "cameras.txt") ||
            safe_exists(path / "sparse" / "cameras.txt")) {
            return DatasetType::COLMAP;
        }

        // Check for Transforms markers
        if (safe_exists(path / "transforms.json") ||
            safe_exists(path / "transforms_train.json")) {
            return DatasetType::Transforms;
        }

        return DatasetType::Unknown;
    }

} // namespace gs::loader