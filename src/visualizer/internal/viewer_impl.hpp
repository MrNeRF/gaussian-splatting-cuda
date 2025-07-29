#pragma once

#include "visualizer/visualizer.hpp"
#include <memory>

namespace gs::visualizer {

    class ViewerImpl : public Visualizer {
    public:
        explicit ViewerImpl(const ViewerOptions& options);
        ~ViewerImpl() override;

        void run() override;
        void setParameters(const param::TrainingParameters& params) override;
        std::expected<void, std::string> loadPLY(const std::filesystem::path& path) override;
        std::expected<void, std::string> loadDataset(const std::filesystem::path& path) override;
        void clearScene() override;

    private:
        class Impl;
        std::unique_ptr<Impl> pImpl;
    };

} // namespace gs::visualizer
