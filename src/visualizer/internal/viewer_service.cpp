#include "internal/viewer_service.hpp"

namespace gs::visualizer {

    class ViewerService::Impl {
        // Implementation details
    };

    ViewerService::ViewerService() : pImpl(std::make_unique<Impl>()) {}

    ViewerService::~ViewerService() = default;

    void ViewerService::initialize() {
        // TODO: Implementation
    }

    void ViewerService::shutdown() {
        // TODO: Implementation
    }

} // namespace gs::visualizer
