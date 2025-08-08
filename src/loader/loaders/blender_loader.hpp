#pragma once

#include "loader/loader_interface.hpp"
#include "loader/formats/colmap.hpp"

namespace gs::loader {

    /**
     * @brief Loader for Blender/NeRF dataset format (transforms.json)
     */
    class BlenderLoader : public IImageDataLoader {
    public:
        BlenderLoader() = default;
        ~BlenderLoader() override = default;

        std::expected<LoadResult, std::string> load(
            const std::filesystem::path& path,
            const LoadOptions& options = {}) override;

        bool canLoad(const std::filesystem::path& path) const override;
        std::string name() const override;
        std::vector<std::string> supportedExtensions() const override;
        int priority() const override;
        std::vector<CameraData> getImagesCams(const std::filesystem::path& path) const override;
    };

} // namespace gs::loader
