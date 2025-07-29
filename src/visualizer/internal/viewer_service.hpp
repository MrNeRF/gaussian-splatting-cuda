#pragma once

#include <memory>
#include <string>

namespace gs::visualizer {

    // Internal service for coordinating visualizer components
    class ViewerService {
    public:
        ViewerService();
        ~ViewerService();

        void initialize();
        void shutdown();

        // Future: Add internal coordination methods

    private:
        class Impl;
        std::unique_ptr<Impl> pImpl;
    };

} // namespace gs::visualizer
