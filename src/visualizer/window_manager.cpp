#include "visualizer/window_manager.hpp"
#include "visualizer/detail.hpp"
// clang-format off
// CRITICAL: GLAD must be included before GLFW to avoid OpenGL header conflicts
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on
#include <iostream>
#include <print>

namespace gs {

    // Initialize static member
    void* WindowManager::callback_handler_ = nullptr;

    WindowManager::WindowManager(const std::string& title, int width, int height)
        : title_(title),
          window_size_(width, height),
          framebuffer_size_(width, height) {
    }

    WindowManager::~WindowManager() {
        if (window_) {
            glfwDestroyWindow(window_);
        }
        glfwTerminate();
    }

    bool WindowManager::init() {
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW!" << std::endl;
            return false;
        }

        glfwWindowHint(GLFW_SAMPLES, 8);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_FALSE);
        glfwWindowHint(GLFW_DEPTH_BITS, 24);

        window_ = glfwCreateWindow(
            window_size_.x,
            window_size_.y,
            title_.c_str(),
            nullptr,
            nullptr);

        if (!window_) {
            std::cerr << "Failed to create GLFW window!" << std::endl;
            glfwTerminate();
            return false;
        }

        glfwMakeContextCurrent(window_);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::cerr << "GLAD init failed" << std::endl;
            glfwTerminate();
            return false;
        }

        // Set up callbacks
        glfwSetMouseButtonCallback(window_, mouseButtonCallback);
        glfwSetCursorPosCallback(window_, cursorPosCallback);
        glfwSetScrollCallback(window_, scrollCallback);
        glfwSetKeyCallback(window_, keyCallback);
        glfwSetDropCallback(window_, dropCallback);

        // Enable vsync by default
        glfwSwapInterval(1);

        // Set up OpenGL state
        glEnable(GL_LINE_SMOOTH);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBlendEquation(GL_FUNC_ADD);
        glEnable(GL_PROGRAM_POINT_SIZE);

        return true;
    }

    void WindowManager::updateWindowSize() {
        int winW, winH, fbW, fbH;
        glfwGetWindowSize(window_, &winW, &winH);
        glfwGetFramebufferSize(window_, &fbW, &fbH);
        window_size_ = glm::ivec2(winW, winH);
        framebuffer_size_ = glm::ivec2(fbW, fbH);
        glViewport(0, 0, fbW, fbH);
    }

    void WindowManager::swapBuffers() {
        glfwSwapBuffers(window_);
    }

    void WindowManager::pollEvents() {
        glfwPollEvents();
    }

    bool WindowManager::shouldClose() const {
        return glfwWindowShouldClose(window_);
    }

    void WindowManager::setVSync(bool enabled) {
        glfwSwapInterval(enabled ? 1 : 0);
    }

    // Static callback implementations
    void WindowManager::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
        if (callback_handler_) {
            ViewerDetail::mouseButtonCallback(window, button, action, mods);
        }
    }

    void WindowManager::cursorPosCallback(GLFWwindow* window, double x, double y) {
        if (callback_handler_) {
            ViewerDetail::cursorPosCallback(window, x, y);
        }
    }

    void WindowManager::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        if (callback_handler_) {
            ViewerDetail::scrollCallback(window, xoffset, yoffset);
        }
    }

    void WindowManager::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (callback_handler_) {
            ViewerDetail::wsad_callback(window, key, scancode, action, mods);
        }
    }

    void WindowManager::dropCallback(GLFWwindow* window, int count, const char** paths) {
        if (callback_handler_) {
            ViewerDetail::dropCallback(window, count, paths);
        }
    }

} // namespace gs
