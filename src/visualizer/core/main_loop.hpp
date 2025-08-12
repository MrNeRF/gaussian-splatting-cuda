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

        // Main run loop
        void run();

    private:
        InitCallback init_callback_;
        UpdateCallback update_callback_;
        RenderCallback render_callback_;
        ShutdownCallback shutdown_callback_;
        ShouldCloseCallback should_close_callback_;
    };

} // namespace gs::visualizer
