#include "config.h" // Include generated config
#include "core/ply_loader.hpp"
#include "core/training_setup.hpp"
#include "visualizer/detail.hpp"
#include "visualizer/gui_manager.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <print>
#include <sstream>
#include <thread>

#ifdef CUDA_GL_INTEROP_ENABLED
#include "visualizer/cuda_gl_interop.hpp"
#endif

namespace gs {

    ViewerDetail::ViewerDetail(std::string title, int width, int height)
        : title_(title),
          viewport_(width, height),
          window_manager_(std::make_unique<WindowManager>(title, width, height)) {
    }

    ViewerDetail::~ViewerDetail() {
        std::cout << "Viewer destroyed." << std::endl;
    }

    bool ViewerDetail::init() {
        if (!window_manager_->init()) {
            return false;
        }

        // Create input handler
        input_handler_ = std::make_unique<InputHandler>(window_manager_->getWindow());

        // Create camera controller
        camera_controller_ = std::make_unique<CameraController>(viewport_);
        camera_controller_->connectToInputHandler(*input_handler_);

        return true;
    }

    void ViewerDetail::updateWindowSize() {
        window_manager_->updateWindowSize();
        viewport_.windowSize = window_manager_->getWindowSize();
        viewport_.frameBufferSize = window_manager_->getFramebufferSize();
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

        // Initialize screen renderer with interop support if available
#ifdef CUDA_GL_INTEROP_ENABLED
        screen_renderer_ = std::make_shared<ScreenQuadRendererInterop>(true);
        std::cout << "CUDA-OpenGL interop enabled for rendering" << std::endl;
#else
        screen_renderer_ = std::make_shared<ScreenQuadRenderer>();
        std::cout << "Using CPU copy for rendering (interop not available)" << std::endl;
#endif

        while (!window_manager_->shouldClose()) {

            // Clear with a dark background
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            controlFrameRate();

            updateWindowSize();

            draw();

            window_manager_->swapBuffers();
            window_manager_->pollEvents();
        }
    }

    bool GSViewer::handleFileDrop(const InputHandler::FileDropEvent& event) {
        // Process each dropped file
        for (const auto& path_str : event.paths) {
            std::filesystem::path filepath(path_str);

            // Check if it's a PLY file
            if (filepath.extension() == ".ply" || filepath.extension() == ".PLY") {
                std::println("Dropped PLY file: {}", filepath.string());

                // Load the PLY file
                loadPLYFile(filepath);

                // Log the action
                if (gui_manager_) {
                    gui_manager_->showScriptingConsole(true);
                    gui_manager_->addConsoleLog("Info: Loaded PLY file via drag-and-drop: %s",
                                                filepath.filename().string().c_str());
                }

                // Only process the first PLY file if multiple files were dropped
                return true;
            }
            if (std::filesystem::is_directory(filepath)) {
                // Check if it's a dataset directory
                bool is_colmap_dataset = false;
                bool is_transforms_dataset = false;

                // Check for COLMAP dataset structure
                if (std::filesystem::exists(filepath / "sparse" / "0" / "cameras.bin") ||
                    std::filesystem::exists(filepath / "sparse" / "cameras.bin")) {
                    is_colmap_dataset = true;
                }

                // Check for transforms dataset
                if (std::filesystem::exists(filepath / "transforms.json") ||
                    std::filesystem::exists(filepath / "transforms_train.json")) {
                    is_transforms_dataset = true;
                }

                if (is_colmap_dataset || is_transforms_dataset) {
                    std::println("Dropped dataset directory: {}", filepath.string());

                    // Load the dataset
                    loadDataset(filepath);

                    // Log the action
                    if (gui_manager_) {
                        gui_manager_->showScriptingConsole(true);
                        gui_manager_->addConsoleLog("Info: Loaded %s dataset via drag-and-drop: %s",
                                                    is_colmap_dataset ? "COLMAP" : "Transforms",
                                                    filepath.filename().string().c_str());
                    }

                    // Only process the first valid dataset if multiple were dropped
                    return true;
                }
            }
        }
        return false;
    }

    GSViewer::GSViewer(std::string title, int width, int height)
        : ViewerDetail(title, width, height),
          trainer_(nullptr) {

        config_ = std::make_shared<RenderingConfig>();
        info_ = std::make_shared<TrainingInfo>();
        notifier_ = std::make_shared<ViewerNotifier>();

        setFrameRate(30);

        // Create GUI manager
        gui_manager_ = std::make_unique<gui::GuiManager>(this);

        // Set up default script executor with basic functionality
        gui_manager_->setScriptExecutor([this](const std::string& command) -> std::string {
            std::ostringstream result;

            // Basic command parsing
            if (command.empty()) {
                return "";
            }

            // Handle basic commands
            if (command == "help" || command == "h") {
                result << "Available commands:\n";
                result << "  help, h - Show this help\n";
                result << "  clear - Clear console\n";
                result << "  status - Show training status\n";
                result << "  model_info - Show model information\n";
                result << "  tensor_info <name> - Show tensor information\n";
                result << "  gpu_info - Show GPU information\n";
                return result.str();
            }

            if (command == "clear") {
                // Handled internally by the console
                return "";
            }

            if (command == "status") {
                if (!trainer_) {
                    return "No trainer available (viewer mode)";
                }
                result << "Training Status:\n";
                result << "  Running: " << (trainer_->is_running() ? "Yes" : "No") << "\n";
                result << "  Paused: " << (trainer_->is_paused() ? "Yes" : "No") << "\n";
                result << "  Complete: " << (trainer_->is_training_complete() ? "Yes" : "No") << "\n";
                result << "  Current Iteration: " << trainer_->get_current_iteration() << "\n";
                result << "  Current Loss: " << std::fixed << std::setprecision(6) << trainer_->get_current_loss();
                return result.str();
            }

            if (command == "model_info") {
                if (!trainer_ && !standalone_model_) {
                    return "No model available";
                }

                result << "Model Information:\n";

                if (trainer_) {
                    std::lock_guard<std::mutex> lock(splat_mtx_);
                    auto& model = trainer_->get_strategy().get_model();
                    result << "  Number of Gaussians: " << model.size() << "\n";
                    result << "  Positions shape: [" << model.get_means().size(0) << ", " << model.get_means().size(1) << "]\n";
                    result << "  Device: " << model.get_means().device() << "\n";
                    result << "  Dtype: " << model.get_means().dtype() << "\n";
                    result << "  Active SH degree: " << model.get_active_sh_degree() << "\n";
                    result << "  Scene scale: " << model.get_scene_scale();
                } else if (standalone_model_) {
                    std::lock_guard<std::mutex> lock(splat_mtx_);
                    result << "  Number of Gaussians: " << standalone_model_->size() << "\n";
                    result << "  Positions shape: [" << standalone_model_->get_means().size(0) << ", " << standalone_model_->get_means().size(1) << "]\n";
                    result << "  Device: " << standalone_model_->get_means().device() << "\n";
                    result << "  Dtype: " << standalone_model_->get_means().dtype() << "\n";
                    result << "  Active SH degree: " << standalone_model_->get_active_sh_degree() << "\n";
                    result << "  Scene scale: " << standalone_model_->get_scene_scale();
                    result << "\n  Mode: Viewer (no training)";
                }

                return result.str();
            }

            if (command == "gpu_info") {
                size_t free_byte, total_byte;
                cudaDeviceSynchronize();
                cudaMemGetInfo(&free_byte, &total_byte);

                double free_gb = free_byte / 1024.0 / 1024.0 / 1024.0;
                double total_gb = total_byte / 1024.0 / 1024.0 / 1024.0;
                double used_gb = total_gb - free_gb;

                result << "GPU Memory Info:\n";
                result << "  Total: " << std::fixed << std::setprecision(2) << total_gb << " GB\n";
                result << "  Used: " << used_gb << " GB\n";
                result << "  Free: " << free_gb << " GB\n";
                result << "  Usage: " << std::setprecision(1) << (used_gb / total_gb * 100.0) << "%";
                return result.str();
            }

            // Handle tensor_info command
            if (command.substr(0, 11) == "tensor_info") {
                if (!trainer_ && !standalone_model_) {
                    return "No model available";
                }

                std::string tensor_name = "";
                if (command.length() > 12) {
                    tensor_name = command.substr(12); // Get parameter after "tensor_info "
                }

                if (tensor_name.empty()) {
                    return "Usage: tensor_info <tensor_name>\nAvailable: means, scaling, rotation, shs, opacity";
                }

                std::lock_guard<std::mutex> lock(splat_mtx_);

                // Get model reference
                SplatData* model = nullptr;
                if (trainer_) {
                    model = const_cast<SplatData*>(&trainer_->get_strategy().get_model());
                } else if (standalone_model_) {
                    model = standalone_model_.get();
                }

                if (!model) {
                    return "Model not available";
                }

                torch::Tensor tensor;
                if (tensor_name == "means" || tensor_name == "positions") {
                    tensor = model->get_means();
                } else if (tensor_name == "scales" || tensor_name == "scaling") {
                    tensor = model->get_scaling();
                } else if (tensor_name == "rotations" || tensor_name == "rotation" || tensor_name == "quats") {
                    tensor = model->get_rotation();
                } else if (tensor_name == "features" || tensor_name == "colors" || tensor_name == "shs") {
                    tensor = model->get_shs();
                } else if (tensor_name == "opacities" || tensor_name == "opacity") {
                    tensor = model->get_opacity();
                } else {
                    return "Unknown tensor: " + tensor_name + "\nAvailable: means, scaling, rotation, shs, opacity";
                }

                result << "Tensor '" << tensor_name << "' info:\n";
                result << "  Shape: [";
                for (int i = 0; i < tensor.dim(); i++) {
                    if (i > 0)
                        result << ", ";
                    result << tensor.size(i);
                }
                result << "]\n";
                result << "  Device: " << tensor.device() << "\n";
                result << "  Dtype: " << tensor.dtype() << "\n";
                result << "  Requires grad: " << (tensor.requires_grad() ? "Yes" : "No") << "\n";

                // Show some statistics if tensor is on CPU or we can move it
                try {
                    auto cpu_tensor = tensor.cpu();
                    auto flat = cpu_tensor.flatten();
                    if (flat.numel() > 0) {
                        result << "  Min: " << torch::min(flat).item<float>() << "\n";
                        result << "  Max: " << torch::max(flat).item<float>() << "\n";
                        result << "  Mean: " << torch::mean(flat).item<float>() << "\n";
                        result << "  Std: " << torch::std(flat).item<float>();
                    }
                } catch (...) {
                    result << "  (Statistics unavailable)";
                }

                return result.str();
            }

            return "Unknown command: '" + command + "'. Type 'help' for available commands.";
        });

        // Set up file selection callback
        gui_manager_->setFileSelectedCallback([this](const std::filesystem::path& path, bool is_dataset) {
            if (is_dataset) {
                loadDataset(path);
            } else {
                loadPLYFile(path);
            }
        });
    }

    GSViewer::~GSViewer() {
        // Stop training thread if running
        if (training_thread_ && training_thread_->joinable()) {
            std::cout << "Viewer closing - stopping training thread..." << std::endl;
            training_thread_->request_stop();
            training_thread_->join();
        }

        // If trainer is still running, request it to stop
        if (trainer_ && trainer_->is_running()) {
            std::cout << "Viewer closing - stopping training..." << std::endl;
            trainer_->request_stop();

            // Give the training thread a moment to acknowledge the stop request
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Cleanup GUI
        if (gui_manager_) {
            gui_manager_->shutdown();
        }

        std::cout << "GSViewer destroyed." << std::endl;
    }

    void GSViewer::setTrainer(Trainer* trainer) {
        trainer_ = trainer;
    }

    void GSViewer::setStandaloneModel(std::unique_ptr<SplatData> model) {
        standalone_model_ = std::move(model);
    }

    void GSViewer::setAntiAliasing(bool enable) {
        anti_aliasing_ = enable;
    }

    void GSViewer::loadPLYFile(const std::filesystem::path& path) {
        try {
            std::println("Loading PLY file: {}", path.string());

            // Clear any existing data
            clearCurrentData();

            // Load the PLY file
            auto splat_result = gs::load_ply(path);
            if (!splat_result) {
                gui_manager_->addConsoleLog("Error: Failed to load PLY: %s", splat_result.error().c_str());
                return;
            }

            standalone_model_ = std::make_unique<SplatData>(std::move(*splat_result));
            current_ply_path_ = path;
            current_mode_ = ViewerMode::PLYViewer;

            gui_manager_->addConsoleLog("Info: Loaded PLY with %lld Gaussians from %s",
                                        standalone_model_->size(),
                                        path.filename().string().c_str());

        } catch (const std::exception& e) {
            gui_manager_->addConsoleLog("Error: Exception loading PLY: %s", e.what());
        }
    }

    void GSViewer::loadDataset(const std::filesystem::path& path) {
        try {
            std::println("Loading dataset from: {}", path.string());

            // Clear any existing data
            clearCurrentData();

            // Use the parameters that were passed to the viewer
            param::TrainingParameters dataset_params = params_;
            dataset_params.dataset.data_path = path; // Override with the selected path

            // Setup training
            auto setup_result = gs::setupTraining(dataset_params);
            if (!setup_result) {
                gui_manager_->addConsoleLog("Error: Failed to setup training: %s", setup_result.error().c_str());
                return;
            }

            // Store the trainer (but don't take ownership yet)
            auto trainer_ptr = setup_result->trainer.get();

            // Link the trainer to this viewer
            trainer_ptr->setViewer(this);

            // Now take ownership
            trainer_ = setup_result->trainer.release();

            current_dataset_path_ = path;
            current_mode_ = ViewerMode::Training;

            // Get dataset info
            size_t num_images = setup_result->dataset->size().value();
            size_t num_gaussians = trainer_->get_strategy().get_model().size();

            gui_manager_->addConsoleLog("Info: Loaded dataset with %zu images and %zu initial Gaussians",
                                        num_images, num_gaussians);
            gui_manager_->addConsoleLog("Info: Ready to start training from %s",
                                        path.filename().string().c_str());
            gui_manager_->addConsoleLog("Info: Using parameters from command line/config");

        } catch (const std::exception& e) {
            gui_manager_->addConsoleLog("Error: Exception loading dataset: %s", e.what());
        }
    }

    void GSViewer::clearCurrentData() {
        // Stop any ongoing training thread
        if (training_thread_ && training_thread_->joinable()) {
            std::println("Stopping training thread...");
            training_thread_->request_stop();
            training_thread_->join();
            training_thread_.reset();
        }

        // Stop any ongoing training via trainer
        if (trainer_ && trainer_->is_running()) {
            trainer_->request_stop();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Clear data
        trainer_ = nullptr;
        standalone_model_.reset();

        // Reset state
        current_mode_ = ViewerMode::Empty;
        current_ply_path_.clear();
        current_dataset_path_.clear();

        // Clear training info
        if (info_) {
            std::lock_guard<std::mutex> lock(info_->mtx);
            info_->curr_iterations_ = 0;
            info_->total_iterations_ = 0;
            info_->num_splats_ = 0;
            info_->loss_buffer_.clear();
        }
    }

    void GSViewer::startTraining() {
        if (!trainer_ || training_thread_) {
            return;
        }

        // First notify the trainer that it's ready to start
        if (notifier_) {
            std::lock_guard<std::mutex> lock(notifier_->mtx);
            notifier_->ready = true;
            notifier_->cv.notify_one();
        }

        // Then start training in a separate thread
        training_thread_ = std::make_unique<std::jthread>(
            [trainer_ptr = trainer_](std::stop_token stop_token) {
                std::println("Training thread started");
                auto train_result = trainer_ptr->train(stop_token);
                if (!train_result) {
                    std::println(stderr, "Training error: {}", train_result.error());
                }
                std::println("Training thread finished");
            });

        std::println("Training thread launched");
    }

    bool GSViewer::isGuiActive() const {
        return gui_manager_ && gui_manager_->isAnyWindowActive();
    }

    void GSViewer::drawFrame() {
        // Only render if we have a model to render
        if (!trainer_ && !standalone_model_) {
            return;
        }

        glm::mat3 R = viewport_.getRotationMatrix();
        glm::vec3 t = viewport_.getTranslation();

        glm::ivec2& reso = viewport_.windowSize;
        // Comprehensive dimension validation, prevents crash when minimizing window (see issue 190)
        if (reso.x <= 0 || reso.y <= 0 || reso.x > 16384 || reso.y > 16384) {
            return; // Skip rendering for invalid dimensions
        }

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

        glm::vec2 fov = config_->getFov(reso.x, reso.y);

        Camera cam = Camera(
            R_tensor,
            t_tensor,
            fov2focal(fov.x, reso.x),
            fov2focal(fov.y, reso.y),
            reso.x / 2.0f,
            reso.y / 2.0f,
            torch::empty({0}, torch::kFloat32),
            torch::empty({0}, torch::kFloat32),
            gsplat::CameraModelType::PINHOLE,
            "online",
            "none image",
            reso.x,
            reso.y,
            -1);

        torch::Tensor background = torch::zeros({3});

        RenderOutput output;
        {
            std::lock_guard<std::mutex> lock(splat_mtx_);

            // Get model from trainer or standalone
            SplatData* model = nullptr;
            if (trainer_) {
                model = const_cast<SplatData*>(&trainer_->get_strategy().get_model());
            } else if (standalone_model_) {
                model = standalone_model_.get();
            }

            if (!model) {
                return;
            }

            output = gs::rasterize(
                cam,
                *model,
                background,
                config_->scaling_modifier,
                false,
                anti_aliasing_,
                RenderMode::RGB);
        }

        // Before the uploadData call, add another safety check
#ifdef CUDA_GL_INTEROP_ENABLED
        // Use interop for direct GPU transfer
        auto interop_renderer = std::dynamic_pointer_cast<ScreenQuadRendererInterop>(screen_renderer_);

        if (interop_renderer && interop_renderer->isInteropEnabled()) {
            // Keep data on GPU - convert [C, H, W] to [H, W, C] format
            auto image_hwc = output.image.permute({1, 2, 0}).contiguous();
            // Direct CUDA->OpenGL update (no CPU copy!)
            // Verify tensor dimensions match expected size
            if (image_hwc.size(0) == reso.y && image_hwc.size(1) == reso.x) {
                interop_renderer->uploadFromCUDA(image_hwc, reso.x, reso.y);
            }
        } else {
            // Fallback to CPU copy
            auto image = (output.image * 255).to(torch::kCPU).to(torch::kU8).permute({1, 2, 0}).contiguous();

            // Verify tensor dimensions before upload
            if (image.size(0) == reso.y && image.size(1) == reso.x && image.data_ptr<uchar>()) {
                screen_renderer_->uploadData(image.data_ptr<uchar>(), reso.x, reso.y);
            }
        }
#else
        // Original CPU copy path
        auto image = (output.image * 255).to(torch::kCPU).to(torch::kU8).permute({1, 2, 0}).contiguous();
        if (image.size(0) == reso.y && image.size(1) == reso.x && image.data_ptr<uchar>()) {
            screen_renderer_->uploadData(image.data_ptr<uchar>(), reso.x, reso.y);
        }
#endif

        screen_renderer_->render(quadShader_, viewport_);
    }

    void GSViewer::draw() {
        // Initialize GUI on first draw
        static bool gui_initialized = false;
        if (!gui_initialized) {
            gui_manager_->init();
            gui_initialized = true;

            // Set up input handlers after GUI is initialized

            // GUI gets first priority
            input_handler_->addMouseButtonHandler(
                [this](const InputHandler::MouseButtonEvent& event) {
                    return isGuiActive(); // Consume if GUI is active
                });

            input_handler_->addMouseMoveHandler(
                [this](const InputHandler::MouseMoveEvent& event) {
                    return isGuiActive(); // Consume if GUI is active
                });

            input_handler_->addMouseScrollHandler(
                [this](const InputHandler::MouseScrollEvent& event) {
                    return isGuiActive(); // Consume if GUI is active
                });

            // File drop handler
            input_handler_->addFileDropHandler(
                [this](const InputHandler::FileDropEvent& event) {
                    return handleFileDrop(event);
                });
        }

        // Draw the 3D frame first
        drawFrame();

        // Then render GUI on top
        gui_manager_->render();
    }

} // namespace gs
