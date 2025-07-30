#pragma once

#include <atomic>

namespace gs {

    // Simple notifier for training start synchronization
    struct ViewerNotifier {
        std::atomic<bool> ready{false};
    };

} // namespace gs