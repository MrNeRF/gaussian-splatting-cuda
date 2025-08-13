#include "core/main_loop.hpp"
#include <glad/glad.h>
#include <iostream>
#include <thread>

namespace gs::visualizer {

    MainLoop::MainLoop() = default;

    void MainLoop::run() {
        // Initialize
        if (init_callback_ && !init_callback_()) {
            std::cerr << "Application initialization failed!" << std::endl;
            return;
        }

        // Main loop
        while (should_close_callback_ ? !should_close_callback_() : true) {
            // Clear
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Update
            if (update_callback_) {
                update_callback_();
            }

            // Render
            if (render_callback_) {
                render_callback_();
            }
        }

        // Shutdown
        if (shutdown_callback_) {
            shutdown_callback_();
        }
    }

} // namespace gs::visualizer
