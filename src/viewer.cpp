#include "visualizer/detail.hpp"

namespace gs {

    ViewerDetail* ViewerDetail::detail_ = nullptr;

    ViewerDetail::ViewerDetail(std::string title, int width, int height)
        : title_(title),
          viewport_(width, height) {
        detail_ = this;
    }

    ViewerDetail::~ViewerDetail() {
        std::cout << "Viewer destroyed." << std::endl;
    }

    bool ViewerDetail::init() {

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
#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

        window_ = glfwCreateWindow(
            detail_->viewport_.windowSize.x,
            detail_->viewport_.windowSize.y,
            detail_->title_.c_str(), NULL, NULL);

        if (window_ == NULL) {
            std::cerr << "Failed to create GLFW window!" << std::endl;
            glfwTerminate();
            return false;
        }

        glfwMakeContextCurrent(window_);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::cerr << "GLAD init failed" << std::endl;
            glfwTerminate();
            return -1;
        }

        glfwSwapInterval(1); // Enable vsync

        glfwSetMouseButtonCallback(window_, mouseButtonCallback);
        glfwSetCursorPosCallback(window_, cursorPosCallback);
        glfwSetScrollCallback(window_, scrollCallback);

        glEnable(GL_LINE_SMOOTH);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBlendEquation(GL_FUNC_ADD);
        glEnable(GL_PROGRAM_POINT_SIZE);

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
        io.ConfigWindowsMoveFromTitleBarOnly = true;
        ImGui::StyleColorsLight();

        // Setup Platform/Renderer backends
        const char* glsl_version = "#version 430";
        ImGui_ImplGlfw_InitForOpenGL(window_, true);
        ImGui_ImplOpenGL3_Init(glsl_version);

        // Set Fonts
        // concat two const char*
        std::string font_path = std::string(PROJECT_ROOT_PATH) +
                                "/include/visualizer/assets/JetBrainsMono-Regular.ttf";
        io.Fonts->AddFontFromFileTTF(font_path.c_str(), 14.0f);

        // Set Windows option
        window_flags |= ImGuiWindowFlags_NoScrollbar;
        window_flags |= ImGuiWindowFlags_NoResize;

        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 2.0f);

        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
        style.WindowPadding = ImVec2(6.0f, 6.0f);
        style.WindowRounding = 6.0f;
        style.WindowBorderSize = 0.0f;

        return true;
    }

    void ViewerDetail::updateWindowSize() {
        int winW, winH, fbW, fbH;
        glfwGetWindowSize(window_, &winW, &winH);
        glfwGetFramebufferSize(window_, &fbW, &fbH);
        viewport_.windowSize = glm::ivec2(winW, winH);
        viewport_.frameBufferSize = glm::ivec2(fbW, fbH);
        glViewport(0, 0, fbW, fbH);
    }

    float ViewerDetail::getGPUUsage() {
        size_t free_byte, total_byte;
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free_byte, &total_byte);
        size_t used_byte = total_byte - free_byte;
        float gpuUsage = used_byte / (float)total_byte * 100;

        return gpuUsage;
    }

    void ViewerDetail::setFrameRate(const int fps) {
        targetFPS = fps;
        frameTime = 1000 / targetFPS;
    }

    void ViewerDetail::controlFrameRate() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime).count();
        if (duration < frameTime) {
            std::this_thread::sleep_for(std::chrono::milliseconds(frameTime - duration));
        }
        lastTime = std::chrono::high_resolution_clock::now();
    }

    void ViewerDetail::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
        if (detail_->any_window_active)
            return;

        if ((button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_RIGHT) && action == GLFW_PRESS) {
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            detail_->viewport_.camera.initScreenPos(glm::vec2(xpos, ypos));
        }
    }

    void ViewerDetail::cursorPosCallback(GLFWwindow* window, double x, double y) {
        if (detail_->any_window_active)
            return;

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            detail_->viewport_.camera.translate(glm::vec2(x, y));
        } else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            detail_->viewport_.camera.rotate(glm::vec2(x, y));
        }
    }

    void ViewerDetail::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        if (detail_->any_window_active)
            return;

        float delta = static_cast<float>(yoffset);
        if (std::abs(delta) < 1.0e-2f)
            return;

        detail_->viewport_.camera.zoom(delta);
    }

    void ViewerDetail::run() {

        if (!init()) {
            std::cerr << "Viewer initialization failed!" << std::endl;
            return;
        }

        std::string shader_path = std::string(PROJECT_ROOT_PATH) + "/include/visualizer/shaders/";
        quadShader_ = std::make_shared<Shader>(
            (shader_path + "/screen_quad.vert").c_str(),
            (shader_path + "/screen_quad.frag").c_str(),
            true);

        screen_renderer_ = std::make_shared<ScreenQuadRenderer>();

        while (!glfwWindowShouldClose(window_)) {

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            controlFrameRate();

            updateWindowSize();

            draw();

            glfwSwapBuffers(window_);
            glfwPollEvents();
        }

        // Cleanup
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(window_);
        glfwTerminate();
    }

    GSViewer::GSViewer(std::string title, int width, int height)
        : ViewerDetail(title, width, height) {

        config_ = std::make_shared<RenderingConfig>();

        info_ = std::make_shared<TrainingInfo>();

        notifier_ = std::make_shared<Notifier>();

        setFrameRate(30);
    }

    void GSViewer::setTrainer(Trainer* trainer) {
        trainer_ = trainer;
    }

    void GSViewer::drawFrame() {

        glm::mat3 R = viewport_.getRotationMatrix();
        glm::vec3 t = viewport_.getTranslation();

        torch::Tensor R_tensor = torch::tensor({R[0][0], R[1][0], R[2][0],
                                                R[0][1], R[1][1], R[2][1],
                                                R[0][2], R[1][2], R[2][2]},
                                               torch::TensorOptions().dtype(torch::kFloat32))
                                     .reshape({3, 3});

        torch::Tensor t_tensor = torch::tensor({t[0],
                                                t[1],
                                                t[2]},
                                               torch::TensorOptions().dtype(torch::kFloat32))
                                     .reshape({3, 1});

        R_tensor = R_tensor.transpose(0, 1);
        t_tensor = -R_tensor.mm(t_tensor).squeeze();

        glm::ivec2& reso = viewport_.windowSize;
        glm::vec2 fov = config_->getFov(reso.x, reso.y);

        Camera cam = Camera(
            R_tensor,
            t_tensor,
            fov.x,
            fov.y,
            "online",
            "none image",
            reso.x,
            reso.y,
            -1);

        torch::Tensor background = torch::zeros({3});

        RenderOutput output;
        {
            std::lock_guard<std::mutex> lock(splat_mtx_);

            output = gs::rasterize(
                cam,
                trainer_->get_strategy().get_model(),
                background,
                config_->scaling_modifier,
                false,
                false,
                RenderMode::RGB);
        }

        auto image = (output.image * 255).to(torch::kCPU).to(torch::kU8).permute({1, 2, 0}).contiguous();

        screen_renderer_->uploadData(image.data_ptr<uchar>(), reso.x, reso.y);

        screen_renderer_->render(quadShader_, viewport_);
    }

    void GSViewer::drawControlPanel() {
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.3f, 0.3f, 0.3f, 0.9f));
        ImGui::Begin("Training Control", &show_control_panel_, window_flags);
        ImGui::SetWindowSize(ImVec2(250, 0));
        ImGui::SetWindowPos(ImVec2(10, viewport_.windowSize.y - 200), ImGuiCond_FirstUseEver);

        bool is_training = trainer_->is_running();
        bool is_paused = trainer_->is_paused();
        bool is_complete = trainer_->is_training_complete();

        // Stop/Resume button
        if (!is_complete) {
            if (is_paused) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
                if (ImGui::Button("Resume Training", ImVec2(-1, 30))) {
                    trainer_->request_resume();
                }
                ImGui::PopStyleColor(2);
            } else if (is_training) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.5f, 0.1f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.6f, 0.2f, 1.0f));
                if (ImGui::Button("Pause Training", ImVec2(-1, 30))) {
                    trainer_->request_pause();
                }
                ImGui::PopStyleColor(2);
            } else {
                ImGui::BeginDisabled();
                ImGui::Button("Waiting...", ImVec2(-1, 30));
                ImGui::EndDisabled();
            }

            ImGui::Spacing();

            // Save button
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.4f, 0.7f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.5f, 0.8f, 1.0f));
            if (ImGui::Button("Save Model", ImVec2(-1, 30))) {
                trainer_->request_save();
                save_in_progress_ = true;
                save_start_time_ = std::chrono::steady_clock::now();
            }
            ImGui::PopStyleColor(2);

            // Show save progress feedback
            if (save_in_progress_) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - save_start_time_).count();
                if (elapsed < 2000) {
                    ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Model saved!");
                } else {
                    save_in_progress_ = false;
                }
            }

            ImGui::Spacing();

            // Exit button
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.3f, 0.3f, 1.0f));
            if (ImGui::Button("Stop & Exit", ImVec2(-1, 30))) {
                trainer_->request_stop();
            }
            ImGui::PopStyleColor(2);
        } else {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Training Complete!");

            ImGui::Spacing();

            // Exit button
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
            if (ImGui::Button("Exit", ImVec2(-1, 30))) {
                glfwSetWindowShouldClose(glfwGetCurrentContext(), GLFW_TRUE);
            }
            ImGui::PopStyleColor(2);
        }

        ImGui::Separator();
        ImGui::Spacing();

        // Status display
        int current_iter = trainer_->get_current_iteration();
        float current_loss = trainer_->get_current_loss();

        ImGui::Text("Status: %s", is_complete ? "Complete" : (is_paused ? "Paused" : (is_training ? "Training" : "Idle")));
        ImGui::Text("Iteration: %d", current_iter);
        ImGui::Text("Loss: %.6f", current_loss);

        ImGui::End();
        ImGui::PopStyleColor();
    }

    void GSViewer::configuration() {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        any_window_active = ImGui::IsAnyItemActive();

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.5f, 0.5f, 0.5f, 0.8f));
        ImGui::Begin("Rendering Setting", nullptr, window_flags);
        ImGui::SetWindowSize(ImVec2(300, 0));

        if (notifier_) {
            // Signal that we're ready to start training
            if (!isRunning) {
                std::lock_guard<std::mutex> lock(notifier_->mtx);
                notifier_->ready = true;
                notifier_->cv.notify_one();
                isRunning = true;
            }
        }

        ImGui::SetNextItemWidth(200);
        ImGui::SliderFloat("##scale_slider", &config_->scaling_modifier, 0.01f, 3.0f, "Scale=%.2f");
        ImGui::SameLine();
        if (ImGui::Button("Reset##scale", ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
            config_->scaling_modifier = 1.0f;
        }

        ImGui::SetNextItemWidth(200);
        ImGui::SliderFloat("##fov_slider", &config_->fov, 45.0f, 120.0f, "FoV=%.2f");
        ImGui::SameLine();
        if (ImGui::Button("Reset##fov", ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
            config_->fov = 75.0f;
        }

        int current_iter;
        int total_iter;
        int num_splats;
        std::vector<float> loss_data;
        {
            std::lock_guard<std::mutex> lock(info_->mtx);
            current_iter = info_->curr_iterations_;
            total_iter = info_->total_iterations_;
            num_splats = info_->num_splats_;
            loss_data.assign(info_->loss_buffer_.begin(), info_->loss_buffer_.end());
        }

        float fraction = float(current_iter) / float(total_iter);
        char overlay_text[64];
        std::snprintf(overlay_text, sizeof(overlay_text), "%d / %d", current_iter, total_iter);
        ImGui::ProgressBar(fraction, ImVec2(-1, 20), overlay_text);

        if (loss_data.size() > 0) {
            auto [min_it, max_it] = std::minmax_element(loss_data.begin(), loss_data.end());
            float min_val = *min_it, max_val = *max_it;

            if (min_val == max_val) {
                min_val -= 1.0f;
                max_val += 1.0f;
            } else {
                float margin = (max_val - min_val) * 0.05f;
                min_val -= margin;
                max_val += margin;
            }

            char loss_label[64];
            std::snprintf(loss_label, sizeof(loss_label), "Loss: %.4f", loss_data.back());

            ImGui::PlotLines(
                "##Loss",
                loss_data.data(),
                static_cast<int>(loss_data.size()),
                0,
                loss_label,
                min_val,
                max_val,
                ImVec2(-1, 50));
        }

        float gpuUsage = getGPUUsage();
        std::string gpuUsageText = "GPU Usage: " + std::to_string(gpuUsage) + "%";
        ImGui::ProgressBar(gpuUsage / 100.0f, ImVec2(-1, 20), gpuUsageText.c_str());

        ImGui::Text("num Splats: %d", num_splats);

        ImGui::End();
        ImGui::PopStyleColor();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    void GSViewer::draw() {

        drawFrame();

        configuration();

        // Draw control panel
        drawControlPanel();
    }

} // namespace gs