#pragma once

#include <functional>
#include <glm/glm.hpp>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declare GLFW types to avoid including GLFW headers
struct GLFWwindow;

namespace gs {

    class InputHandler {
    public:
        // Event types
        struct MouseButtonEvent {
            int button;
            int action;
            int mods;
            glm::dvec2 position;
        };

        struct MouseMoveEvent {
            glm::dvec2 position;
            glm::dvec2 delta;
        };

        struct MouseScrollEvent {
            double xoffset;
            double yoffset;
        };

        struct KeyEvent {
            int key;
            int scancode;
            int action;
            int mods;
        };

        struct FileDropEvent {
            std::vector<std::string> paths;
        };

        // Callback types - now return void instead of bool
        using MouseButtonCallback = std::function<void(const MouseButtonEvent&)>;
        using MouseMoveCallback = std::function<void(const MouseMoveEvent&)>;
        using MouseScrollCallback = std::function<void(const MouseScrollEvent&)>;
        using KeyCallback = std::function<void(const KeyEvent&)>;
        using FileDropCallback = std::function<void(const FileDropEvent&)>;

        // Viewport check callback
        using ViewportCheckCallback = std::function<bool(double x, double y)>;

        explicit InputHandler(GLFWwindow* window);
        ~InputHandler();

        // Set the current input consumer (GUI or Viewport)
        enum class InputConsumer {
            None,
            GUI,
            Viewport
        };
        void setInputConsumer(InputConsumer consumer) { current_consumer_ = consumer; }
        InputConsumer getInputConsumer() const { return current_consumer_; }

        // Set callbacks for different consumers
        void setGUICallbacks(
            MouseButtonCallback mouse_button = nullptr,
            MouseMoveCallback mouse_move = nullptr,
            MouseScrollCallback mouse_scroll = nullptr,
            KeyCallback key = nullptr);

        void setViewportCallbacks(
            MouseButtonCallback mouse_button = nullptr,
            MouseMoveCallback mouse_move = nullptr,
            MouseScrollCallback mouse_scroll = nullptr,
            KeyCallback key = nullptr);

        void setFileDropCallback(FileDropCallback callback);

        // Set viewport check callback
        void setViewportCheckCallback(ViewportCheckCallback callback) {
            viewport_check_callback_ = callback;
        }

        // Input state queries
        bool isKeyPressed(int key) const;
        bool isMouseButtonPressed(int button) const;
        glm::dvec2 getMousePosition() const { return current_mouse_pos_; }
        glm::dvec2 getMouseDelta() const { return mouse_delta_; }

        // Enable/disable input processing
        void setEnabled(bool enabled) { enabled_ = enabled; }
        bool isEnabled() const { return enabled_; }

    private:
        // Callback storage
        struct Callbacks {
            MouseButtonCallback mouse_button;
            MouseMoveCallback mouse_move;
            MouseScrollCallback mouse_scroll;
            KeyCallback key;
        };

        // GLFW callbacks
        static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
        static void cursorPosCallback(GLFWwindow* window, double x, double y);
        static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
        static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
        static void dropCallback(GLFWwindow* window, int count, const char** paths);

        GLFWwindow* window_;
        bool enabled_ = true;
        InputConsumer current_consumer_ = InputConsumer::None;

        // Callbacks for different consumers
        Callbacks gui_callbacks_;
        Callbacks viewport_callbacks_;
        FileDropCallback file_drop_callback_;
        ViewportCheckCallback viewport_check_callback_;

        // Input state
        mutable std::mutex state_mutex_;
        std::unordered_map<int, bool> key_states_;
        std::unordered_map<int, bool> mouse_button_states_;
        glm::dvec2 current_mouse_pos_{0.0, 0.0};
        glm::dvec2 last_mouse_pos_{0.0, 0.0};
        glm::dvec2 mouse_delta_{0.0, 0.0};

        static InputHandler* instance_; // For GLFW callbacks
    };

} // namespace gs