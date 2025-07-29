#include "ply_loader.hpp"
#include "formats/ply.hpp"
#include "core/splat_data.hpp"
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>

namespace gs::loader {

    std::expected<LoadResult, std::string> PLYLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        auto start_time = std::chrono::high_resolution_clock::now();

        // Report progress if callback provided
        if (options.progress) {
            options.progress(0.0f, "Loading PLY file...");
        }

        // Validate file exists
        if (!std::filesystem::exists(path)) {
            return std::unexpected(std::format("PLY file does not exist: {}", path.string()));
        }

        if (!std::filesystem::is_regular_file(path)) {
            return std::unexpected("Path is not a regular file");
        }

        // Validation only mode
        if (options.validate_only) {
            // Basic validation - check if it's a PLY file
            std::ifstream file(path, std::ios::binary);
            if (!file) {
                return std::unexpected("Cannot open file for reading");
            }

            std::string header;
            std::getline(file, header);
            if (header != "ply" && header != "ply\r") {
                return std::unexpected("File does not start with 'ply' header");
            }

            if (options.progress) {
                options.progress(100.0f, "PLY validation complete");
            }

            // Return empty result for validation only
            return LoadResult{
                .data = std::make_shared<SplatData>(), // Empty shared_ptr
                .scene_center = torch::zeros({3}),
                .loader_used = name(),
                .load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start_time),
                .warnings = {}};
        }

        // Load the PLY file using existing implementation
        if (options.progress) {
            options.progress(50.0f, "Parsing PLY data...");
        }

        auto splat_result = gs::load_ply(path);
        if (!splat_result) {
            return std::unexpected(splat_result.error());
        }

        if (options.progress) {
            options.progress(100.0f, "PLY loading complete");
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        LoadResult result{
            .data = std::make_shared<SplatData>(std::move(*splat_result)),
            .scene_center = torch::zeros({3}),
            .loader_used = name(),
            .load_time = load_time,
            .warnings = {}};

        std::println("PLY loaded successfully in {}ms", load_time.count());

        return result;
    }

    bool PLYLoader::canLoad(const std::filesystem::path& path) const {
        if (!std::filesystem::exists(path) || std::filesystem::is_directory(path)) {
            return false;
        }

        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return ext == ".ply";
    }

    std::string PLYLoader::name() const {
        return "PLY";
    }

    std::vector<std::string> PLYLoader::supportedExtensions() const {
        return {".ply", ".PLY"};
    }

    int PLYLoader::priority() const {
        return 10; // Higher priority for PLY files
    }

} // namespace gs::loader
