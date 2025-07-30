#pragma once

#include <chrono>
#include <functional>
#include <memory>

namespace gs::visualizer {

    class MainLoop {
    public:
        using InitCallback = std::function<bool()>;
        using UpdateCallback = std::function<void()>;
        using RenderCallback = std::function<void()>;
        using ShutdownCallback = std::function<void()>;
        using ShouldCloseCallback = std::function<bool()>;

        MainLoop();

        // Set callbacks
        void setInitCallback(InitCallback cb) { init_callback_ = cb; }
        void setUpdateCallback(UpdateCallback cb) { update_callback_ = cb; }
        void setRenderCallback(RenderCallback cb) { render_callback_ = cb; }
        void setShutdownCallback(ShutdownCallback cb) { shutdown_callback_ = cb; }
        void setShouldCloseCallback(ShouldCloseCallback cb) { should_close_callback_ = cb; }

        // Frame rate control
        void setTargetFPS(int fps);
        int getTargetFPS() const { return target_fps_; }

        // Main run loop
        void run();

    private:
        void controlFrameRate();

        InitCallback init_callback_;
        UpdateCallback update_callback_;
        RenderCallback render_callback_;
        ShutdownCallback shutdown_callback_;
        ShouldCloseCallback should_close_callback_;

        int target_fps_ = 30;
        int frame_time_ms_;
        std::chrono::time_point<std::chrono::high_resolution_clock> last_frame_time_;
    };

} // namespace gs::visualizer
