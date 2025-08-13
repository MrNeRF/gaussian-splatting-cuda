#include "input/input_handler.hpp"

// clang-format off
// CRITICAL: GLAD must be included before GLFW to avoid OpenGL header conflicts
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include <algorithm>
#include <imgui.h> // Add ImGui for checking its state
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

    void InputHandler::setGUICallbacks(
        MouseButtonCallback mouse_button,
        MouseMoveCallback mouse_move,
        MouseScrollCallback mouse_scroll,
        KeyCallback key) {
        gui_callbacks_.mouse_button = mouse_button;
        gui_callbacks_.mouse_move = mouse_move;
        gui_callbacks_.mouse_scroll = mouse_scroll;
        gui_callbacks_.key = key;
    }

    void InputHandler::setViewportCallbacks(
        MouseButtonCallback mouse_button,
        MouseMoveCallback mouse_move,
        MouseScrollCallback mouse_scroll,
        KeyCallback key) {
        viewport_callbacks_.mouse_button = mouse_button;
        viewport_callbacks_.mouse_move = mouse_move;
        viewport_callbacks_.mouse_scroll = mouse_scroll;
        viewport_callbacks_.key = key;
    }

    void InputHandler::setFileDropCallback(FileDropCallback callback) {
        file_drop_callback_ = callback;
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

        bool imgui_wants_mouse = ImGui::GetIO().WantCaptureMouse;
        bool in_viewport = instance_->viewport_check_callback_ ? instance_->viewport_check_callback_(x, y) : false;

        // For mouse button events, we need to check if ImGui wants the mouse BEFORE we process
        // But we also need to handle releases for buttons we're tracking
        bool send_to_viewport = false;
        bool send_to_gui = false;

        if (action == GLFW_PRESS) {
            // On press, check where the mouse is and if ImGui wants it
            if (!imgui_wants_mouse && in_viewport) {
                send_to_viewport = true;
            } else {
                send_to_gui = true;
            }
        } else if (action == GLFW_RELEASE) {
            // On release, send to both - let them handle their own state
            send_to_viewport = true;
            send_to_gui = true;
        }

        // Send events based on decision
        if (send_to_viewport && instance_->viewport_callbacks_.mouse_button) {
            instance_->viewport_callbacks_.mouse_button(event);
        }
        if (send_to_gui && instance_->gui_callbacks_.mouse_button) {
            instance_->gui_callbacks_.mouse_button(event);
        }
    }

    void InputHandler::cursorPosCallback([[maybe_unused]] GLFWwindow* window, double x, double y) {
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

        // Send to both consumers - let them decide what to handle
        if (instance_->viewport_callbacks_.mouse_move) {
            instance_->viewport_callbacks_.mouse_move(event);
        }
        if (instance_->gui_callbacks_.mouse_move) {
            instance_->gui_callbacks_.mouse_move(event);
        }
    }

    void InputHandler::scrollCallback([[maybe_unused]] GLFWwindow* window, double xoffset, double yoffset) {
        if (!instance_ || !instance_->enabled_)
            return;

        MouseScrollEvent event{
            .xoffset = xoffset,
            .yoffset = yoffset};

        // Send to both consumers based on current consumer state
        // Scroll should respect focus more strictly than mouse buttons
        const Callbacks* callbacks = nullptr;
        switch (instance_->current_consumer_) {
        case InputConsumer::GUI:
            callbacks = &instance_->gui_callbacks_;
            break;
        case InputConsumer::Viewport:
            callbacks = &instance_->viewport_callbacks_;
            break;
        default:
            return;
        }

        if (callbacks && callbacks->mouse_scroll) {
            callbacks->mouse_scroll(event);
        }
    }

    void InputHandler::keyCallback([[maybe_unused]] GLFWwindow* window, int key, int scancode, int action, int mods) {
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

        // Route to appropriate consumer based on focus
        const Callbacks* callbacks = nullptr;
        switch (instance_->current_consumer_) {
        case InputConsumer::GUI:
            callbacks = &instance_->gui_callbacks_;
            break;
        case InputConsumer::Viewport:
            callbacks = &instance_->viewport_callbacks_;
            break;
        default:
            return;
        }

        if (callbacks && callbacks->key) {
            callbacks->key(event);
        }
    }

    void InputHandler::dropCallback([[maybe_unused]] GLFWwindow* window, int count, const char** paths) {
        if (!instance_ || !instance_->enabled_)
            return;

        std::vector<std::string> file_paths;
        file_paths.reserve(count);
        for (int i = 0; i < count; ++i) {
            file_paths.emplace_back(paths[i]);
        }

        FileDropEvent event{
            .paths = std::move(file_paths)};

        // File drops are always handled regardless of consumer
        if (instance_->file_drop_callback_) {
            instance_->file_drop_callback_(event);
        }
    }

} // namespace gs