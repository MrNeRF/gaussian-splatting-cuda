#include "loader/loader_service.hpp"
#include "core/logger.hpp"
#include "loader/loaders/blender_loader.hpp"
#include "loader/loaders/colmap_loader.hpp"
#include "loader/loaders/ply_loader.hpp"
#include <format>

namespace gs::loader {

    LoaderService::LoaderService()
        : registry_(std::make_unique<DataLoaderRegistry>()) {

        // Register default loaders
        registry_->registerLoader(std::make_unique<PLYLoader>());
        registry_->registerLoader(std::make_unique<ColmapLoader>());
        registry_->registerLoader(std::make_unique<BlenderLoader>());

        LOG_DEBUG("LoaderService initialized with {} loaders", registry_->size());
    }

    std::expected<LoadResult, std::string> LoaderService::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        // Find appropriate loader
        auto* loader = registry_->findLoader(path);
        if (!loader) {
            // Build detailed error message
            std::string error_msg = std::format(
                "No loader found for path: {}\n", path.string());

            // Try all loaders to get diagnostic info
            error_msg += "Tried loaders:\n";
            for (const auto& info : registry_->getLoaderInfo()) {
                error_msg += std::format("  - {}: ", info.name);

                // Get specific loader to check
                auto loaders = registry_->findAllLoaders(path);
                bool can_load = false;
                for (auto* l : loaders) {
                    if (l->name() == info.name) {
                        can_load = true;
                        break;
                    }
                }

                if (!can_load) {
                    if (info.extensions.empty()) {
                        error_msg += "directory-based format not detected\n";
                    } else {
                        error_msg += "extension not supported\n";
                    }
                }
            }

            LOG_ERROR("Failed to find loader: {}", error_msg);
            throw std::runtime_error(error_msg);
        }

        LOG_INFO("Using {} loader for: {}", loader->name(), path.string());

        // Perform the load
        try {
            return loader->load(path, options);
        } catch (const std::exception& e) {
            std::string error_msg = std::format(
                "{} loader failed: {}", loader->name(), e.what());
            LOG_ERROR("{}", error_msg);
            throw std::runtime_error(error_msg);
        }
    }

    std::vector<std::string> LoaderService::getAvailableLoaders() const {
        std::vector<std::string> names;
        for (const auto& info : registry_->getLoaderInfo()) {
            names.push_back(info.name);
        }
        return names;
    }

    std::vector<std::string> LoaderService::getSupportedExtensions() const {
        return registry_->getAllSupportedExtensions();
    }

} // namespace gs::loader