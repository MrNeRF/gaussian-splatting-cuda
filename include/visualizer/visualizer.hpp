#pragma once

#include <expected>
#include <filesystem>
#include <memory>
#include <string>

namespace gs {
    namespace param {
        struct TrainingParameters;
    }
} // namespace gs

namespace gs::visualizer {

    struct ViewerOptions {
        std::string title = "3DGS Viewer";
        int width = 1280;
        int height = 720;
        bool antialiasing = false;
        int target_fps = 30;
        bool enable_cuda_interop = true;
    };

    class Visualizer {
    public:
        static std::unique_ptr<Visualizer> create(const ViewerOptions& options = {});

        virtual void run() = 0;
        virtual void setParameters(const param::TrainingParameters& params) = 0;
        virtual std::expected<void, std::string> loadPLY(const std::filesystem::path& path) = 0;
        virtual std::expected<void, std::string> loadDataset(const std::filesystem::path& path) = 0;
        virtual void clearScene() = 0;

        virtual ~Visualizer() = default;
    };

} // namespace gs::visualizer
