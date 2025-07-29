#include "visualizer/visualizer.hpp"
#include "internal/viewer_impl.hpp"

namespace gs::visualizer {

    std::unique_ptr<Visualizer> Visualizer::create(const ViewerOptions& options) {
        return std::make_unique<ViewerImpl>(options);
    }

} // namespace gs::visualizer
