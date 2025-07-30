#include "input/input_handler.hpp"

// clang-format off
// CRITICAL: GLAD must be included before GLFW to avoid OpenGL header conflicts
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include <algorithm>
#include <print>

namespace gs {

    InputHandler* InputHandler::instance_ = nullptr;

    InputHandler::InputHandler(GLFWwindow* window) : window_(window) {
        instance_ = this;

        // Set GLFW callbacks
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetCursorPosCallback(window, cursorPosCallback);
        glfwSetScrollCallback(window, scrollCallback);
        glfwSetKeyCallback(window, keyCallback);
        glfwSetDropCallback(window, dropCallback);

        // Initialize mouse position
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        current_mouse_pos_ = glm::dvec2(x, y);
        last_mouse_pos_ = current_mouse_pos_;
    }

    InputHandler::~InputHandler() {
        if (instance_ == this) {
            instance_ = nullptr;
        }
    }

    void InputHandler::addMouseButtonHandler(MouseButtonCallback handler) {
        mouse_button_handlers_.push_back(handler);
    }

    void InputHandler::addMouseMoveHandler(MouseMoveCallback handler) {
        mouse_move_handlers_.push_back(handler);
    }

    void InputHandler::addMouseScrollHandler(MouseScrollCallback handler) {
        mouse_scroll_handlers_.push_back(handler);
    }

    void InputHandler::addKeyHandler(KeyCallback handler) {
        key_handlers_.push_back(handler);
    }

    void InputHandler::addFileDropHandler(FileDropCallback handler) {
        file_drop_handlers_.push_back(handler);
    }

    bool InputHandler::isKeyPressed(int key) const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto it = key_states_.find(key);
        return it != key_states_.end() && it->second;
    }

    bool InputHandler::isMouseButtonPressed(int button) const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto it = mouse_button_states_.find(button);
        return it != mouse_button_states_.end() && it->second;
    }

    bool InputHandler::dispatchMouseButton(const MouseButtonEvent& event) {
        for (const auto& handler : mouse_button_handlers_) {
            if (handler(event)) {
                return true; // Event consumed
            }
        }
        return false;
    }

    bool InputHandler::dispatchMouseMove(const MouseMoveEvent& event) {
        for (const auto& handler : mouse_move_handlers_) {
            if (handler(event)) {
                return true; // Event consumed
            }
        }
        return false;
    }

    bool InputHandler::dispatchMouseScroll(const MouseScrollEvent& event) {
        for (const auto& handler : mouse_scroll_handlers_) {
            if (handler(event)) {
                return true; // Event consumed
            }
        }
        return false;
    }

    bool InputHandler::dispatchKey(const KeyEvent& event) {
        for (const auto& handler : key_handlers_) {
            if (handler(event)) {
                return true; // Event consumed
            }
        }
        return false;
    }

    bool InputHandler::dispatchFileDrop(const FileDropEvent& event) {
        for (const auto& handler : file_drop_handlers_) {
            if (handler(event)) {
                return true; // Event consumed
            }
        }
        return false;
    }

    // GLFW Callbacks
    void InputHandler::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
        if (!instance_ || !instance_->enabled_)
            return;

        double x, y;
        glfwGetCursorPos(window, &x, &y);

        MouseButtonEvent event{
            .button = button,
            .action = action,
            .mods = mods,
            .position = glm::dvec2(x, y)};

        // Update state
        {
            std::lock_guard<std::mutex> lock(instance_->state_mutex_);
            instance_->mouse_button_states_[button] = (action == GLFW_PRESS);
        }

        instance_->dispatchMouseButton(event);
    }

    void InputHandler::cursorPosCallback(GLFWwindow* window, double x, double y) {
        if (!instance_ || !instance_->enabled_)
            return;

        glm::dvec2 new_pos(x, y);

        // Calculate delta
        glm::dvec2 delta = new_pos - instance_->current_mouse_pos_;

        // Update state
        {
            std::lock_guard<std::mutex> lock(instance_->state_mutex_);
            instance_->last_mouse_pos_ = instance_->current_mouse_pos_;
            instance_->current_mouse_pos_ = new_pos;
            instance_->mouse_delta_ = delta;
        }

        MouseMoveEvent event{
            .position = new_pos,
            .delta = delta};

        instance_->dispatchMouseMove(event);
    }

    void InputHandler::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        if (!instance_ || !instance_->enabled_)
            return;

        MouseScrollEvent event{
            .xoffset = xoffset,
            .yoffset = yoffset};

        instance_->dispatchMouseScroll(event);
    }

    void InputHandler::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (!instance_ || !instance_->enabled_)
            return;

        KeyEvent event{
            .key = key,
            .scancode = scancode,
            .action = action,
            .mods = mods};

        // Update state
        {
            std::lock_guard<std::mutex> lock(instance_->state_mutex_);
            instance_->key_states_[key] = (action == GLFW_PRESS || action == GLFW_REPEAT);
        }

        instance_->dispatchKey(event);
    }

    void InputHandler::dropCallback(GLFWwindow* window, int count, const char** paths) {
        if (!instance_ || !instance_->enabled_)
            return;

        std::vector<std::string> file_paths;
        file_paths.reserve(count);
        for (int i = 0; i < count; ++i) {
            file_paths.emplace_back(paths[i]);
        }

        FileDropEvent event{
            .paths = std::move(file_paths)};

        instance_->dispatchFileDrop(event);
    }

} // namespace gs