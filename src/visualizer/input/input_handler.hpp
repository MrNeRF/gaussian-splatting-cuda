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

        // Callback types
        using MouseButtonCallback = std::function<bool(const MouseButtonEvent&)>;
        using MouseMoveCallback = std::function<bool(const MouseMoveEvent&)>;
        using MouseScrollCallback = std::function<bool(const MouseScrollEvent&)>;
        using KeyCallback = std::function<bool(const KeyEvent&)>;
        using FileDropCallback = std::function<bool(const FileDropEvent&)>;

        explicit InputHandler(GLFWwindow* window);
        ~InputHandler();

        // Subscribe to events (returns true if event was consumed)
        void addMouseButtonHandler(MouseButtonCallback handler);
        void addMouseMoveHandler(MouseMoveCallback handler);
        void addMouseScrollHandler(MouseScrollCallback handler);
        void addKeyHandler(KeyCallback handler);
        void addFileDropHandler(FileDropCallback handler);

        // Input state queries
        bool isKeyPressed(int key) const;
        bool isMouseButtonPressed(int button) const;
        glm::dvec2 getMousePosition() const { return current_mouse_pos_; }
        glm::dvec2 getMouseDelta() const { return mouse_delta_; }

        // Enable/disable input processing
        void setEnabled(bool enabled) { enabled_ = enabled; }
        bool isEnabled() const { return enabled_; }

    private:
        // GLFW callbacks
        static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
        static void cursorPosCallback(GLFWwindow* window, double x, double y);
        static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
        static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
        static void dropCallback(GLFWwindow* window, int count, const char** paths);

        // Dispatch events to handlers (returns true if consumed)
        bool dispatchMouseButton(const MouseButtonEvent& event);
        bool dispatchMouseMove(const MouseMoveEvent& event);
        bool dispatchMouseScroll(const MouseScrollEvent& event);
        bool dispatchKey(const KeyEvent& event);
        bool dispatchFileDrop(const FileDropEvent& event);

        GLFWwindow* window_;
        bool enabled_ = true;

        // Handler lists (processed in order, stops when one returns true)
        std::vector<MouseButtonCallback> mouse_button_handlers_;
        std::vector<MouseMoveCallback> mouse_move_handlers_;
        std::vector<MouseScrollCallback> mouse_scroll_handlers_;
        std::vector<KeyCallback> key_handlers_;
        std::vector<FileDropCallback> file_drop_handlers_;

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