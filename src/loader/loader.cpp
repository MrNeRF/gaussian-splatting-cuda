/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "loader/loader.hpp"
#include "core/logger.hpp"
#include "loader_service.hpp"
#include <memory>

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
                if (!std::filesystem::exists(path)) {
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

} // namespace gs::loader