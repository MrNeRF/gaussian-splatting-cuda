#include "core/main_loop.hpp"
#include <glad/glad.h>
#include <iostream>
#include <thread>

namespace gs::visualizer {

    MainLoop::MainLoop() {
        setTargetFPS(30);
    }

    void MainLoop::setTargetFPS(int fps) {
        target_fps_ = fps;
        frame_time_ms_ = 1000 / target_fps_;
    }

    void MainLoop::controlFrameRate() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now - last_frame_time_)
                            .count();

        if (duration < frame_time_ms_) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(frame_time_ms_ - duration));
        }

        last_frame_time_ = std::chrono::high_resolution_clock::now();
    }

    void MainLoop::run() {
        // Initialize
        if (init_callback_ && !init_callback_()) {
            std::cerr << "Application initialization failed!" << std::endl;
            return;
        }

        // Initialize frame time
        last_frame_time_ = std::chrono::high_resolution_clock::now();

        // Main loop
        while (should_close_callback_ ? !should_close_callback_() : true) {
            // Clear
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Frame rate control
            controlFrameRate();

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
