#pragma once

#include <condition_variable>
#include <mutex>

namespace gs {

    // Notifier struct for viewer synchronization
    struct ViewerNotifier {
        bool ready = false;
        std::mutex mtx;
        std::condition_variable cv;
    };

} // namespace gs
