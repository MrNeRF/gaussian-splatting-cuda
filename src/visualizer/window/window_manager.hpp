/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <atomic>
#include <filesystem>
#include <glm/glm.hpp>
#include <string>

// Forward declarations
struct GLFWwindow;

namespace gs {

    class WindowManager {
    public:
        WindowManager(const std::string& title, int width, int height);
        ~WindowManager();

        // Delete copy operations
        WindowManager(const WindowManager&) = delete;
        WindowManager& operator=(const WindowManager&) = delete;

        // Initialize GLFW and create window
        bool init();

        // Window operations
        void updateWindowSize();
        void swapBuffers();
        void pollEvents();
        bool shouldClose() const;
        void cancelClose();
        void setVSync(bool enabled);
        [[nodiscard]] bool getVSync() const { return vsync_enabled_; }
        void requestRedraw();
        bool needsRedraw() const;

        // Getters
        GLFWwindow* getWindow() const { return window_; }
        glm::ivec2 getWindowSize() const { return window_size_; }
        glm::ivec2 getFramebufferSize() const { return framebuffer_size_; }

        // Set the callback handler (typically the viewer instance)
        void setCallbackHandler(void* handler) { callback_handler_ = handler; }

    private:
        GLFWwindow* window_ = nullptr;
        std::string title_;
        glm::ivec2 window_size_;
        glm::ivec2 framebuffer_size_;

        // Static callback handler pointer
        static void* callback_handler_;
        bool vsync_enabled_ = true;                     // Track VSync state
        mutable std::atomic<bool> needs_redraw_{false}; // Redraw flag

        // Static GLFW callbacks
        static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
        static void cursorPosCallback(GLFWwindow* window, double x, double y);
        static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
        static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
        static void dropCallback(GLFWwindow* window, int count, const char** paths);
    };

} // namespace gs