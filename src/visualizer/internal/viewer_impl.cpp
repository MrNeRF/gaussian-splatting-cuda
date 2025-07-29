#include "viewer_impl.hpp"
#include "core/parameters.hpp"
#include "core/training_setup.hpp"
#include "legacy/detail.hpp"
#include "viewer_service.hpp"
#include <print>

namespace gs::visualizer {

    class ViewerImpl::Impl {
    public:
        ViewerOptions options;
        std::unique_ptr<GSViewer> legacy_viewer;
        std::unique_ptr<ViewerService> service;
        param::TrainingParameters params;

        Impl(const ViewerOptions& opt)
            : options(opt),
              service(std::make_unique<ViewerService>()) {
            // Defer viewer creation until run() for proper GL context
        }

        void ensureViewer() {
            if (!legacy_viewer) {
                legacy_viewer = std::make_unique<GSViewer>(
                    options.title,
                    options.width,
                    options.height);
                legacy_viewer->setAntiAliasing(options.antialiasing);
                if (!params.dataset.data_path.empty()) {
                    legacy_viewer->setParameters(params);
                }
            }
        }
    };

    ViewerImpl::ViewerImpl(const ViewerOptions& options)
        : pImpl(std::make_unique<Impl>(options)) {
    }

    ViewerImpl::~ViewerImpl() = default;

    void ViewerImpl::run() {
        pImpl->ensureViewer();
        pImpl->legacy_viewer->run();
    }

    void ViewerImpl::setParameters(const param::TrainingParameters& params) {
        pImpl->params = params;
        if (pImpl->legacy_viewer) {
            pImpl->legacy_viewer->setParameters(params);
        }
    }

    std::expected<void, std::string> ViewerImpl::loadPLY(const std::filesystem::path& path) {
        try {
            pImpl->ensureViewer();
            pImpl->legacy_viewer->loadPLYFile(path);
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to load PLY: {}", e.what()));
        }
    }

    std::expected<void, std::string> ViewerImpl::loadDataset(const std::filesystem::path& path) {
        try {
            pImpl->ensureViewer();
            pImpl->legacy_viewer->loadDataset(path);
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to load dataset: {}", e.what()));
        }
    }

    void ViewerImpl::clearScene() {
        if (pImpl->legacy_viewer) {
            pImpl->legacy_viewer->clearCurrentData();
        }
    }

} // namespace gs::visualizer
