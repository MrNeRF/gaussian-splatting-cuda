#include "config.h" // Include generated config
#include "core/ply_loader.hpp"
#include "core/splat_data.hpp"
#include "core/training_setup.hpp"
#include "visualizer/detail.hpp"
#include "visualizer/gui_manager.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <print>
#include <sstream>
#include <thread>

#include <cuda_runtime.h>

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
                    event_bus_->publish(LogMessageEvent{
                        LogMessageEvent::Level::Info,
                        std::format("Loaded PLY file via drag-and-drop: {}", filepath.filename().string()),
                        "GSViewer"});
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
                        event_bus_->publish(LogMessageEvent{
                            LogMessageEvent::Level::Info,
                            std::format("Loaded {} dataset via drag-and-drop: {}",
                                        is_colmap_dataset ? "COLMAP" : "Transforms",
                                        filepath.filename().string()),
                            "GSViewer"});
                    }

                    // Only process the first valid dataset if multiple were dropped
                    return true;
                }
            }
        }
        return false;
    }

    GSViewer::GSViewer(std::string title, int width, int height)
        : ViewerDetail(title, width, height) {

        // Initialize event bus first
        event_bus_ = std::make_shared<EventBus>();

        config_ = std::make_shared<RenderingConfig>();
        info_ = std::make_shared<TrainingInfo>();
        notifier_ = std::make_shared<ViewerNotifier>();
        crop_box_ = std::make_shared<RenderBoundingBox>();

        scene_ = std::make_unique<Scene>();
        trainer_manager_ = std::make_unique<TrainerManager>();
        trainer_manager_->setViewer(this);
        trainer_manager_->setEventBus(event_bus_);

        setFrameRate(30);

        // Create GUI manager with event bus
        gui_manager_ = std::make_unique<gui::GuiManager>(this, event_bus_);

        // Create error handler
        error_handler_ = std::make_unique<ErrorHandler>(event_bus_);

        // Create memory monitor and start it
        memory_monitor_ = std::make_unique<MemoryMonitor>(event_bus_);
        memory_monitor_->start();

        // Setup event handlers
        setupEventHandlers();

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
                if (!trainer_manager_->hasTrainer()) {
                    return "No trainer available (viewer mode)";
                }
                auto trainer = trainer_manager_->getTrainer();
                result << "Training Status:\n";
                result << "  State: " << [this]() {
                    switch (trainer_manager_->getState()) {
                    case TrainerManager::State::Idle: return "Idle";
                    case TrainerManager::State::Ready: return "Ready";
                    case TrainerManager::State::Running: return "Running";
                    case TrainerManager::State::Paused: return "Paused";
                    case TrainerManager::State::Stopping: return "Stopping";
                    case TrainerManager::State::Completed: return "Completed";
                    case TrainerManager::State::Error: return "Error";
                    default: return "Unknown";
                    }
                }() << "\n";
                result << "  Current Iteration: " << trainer->get_current_iteration() << "\n";
                result << "  Current Loss: " << std::fixed << std::setprecision(6) << trainer->get_current_loss();
                return result.str();
            }

            if (command == "model_info") {
                if (!scene_->hasModel()) {
                    return "No model available";
                }

                result << "Model Information:\n";

                const SplatData* model = scene_->getModel();
                if (model) {
                    result << "  Number of Gaussians: " << model->size() << "\n";
                    result << "  Positions shape: [" << model->get_means().size(0) << ", " << model->get_means().size(1) << "]\n";
                    result << "  Device: " << model->get_means().device() << "\n";
                    result << "  Dtype: " << model->get_means().dtype() << "\n";
                    result << "  Active SH degree: " << model->get_active_sh_degree() << "\n";
                    result << "  Scene scale: " << model->get_scene_scale();

                    if (scene_->getMode() == Scene::Mode::Viewing) {
                        result << "\n  Mode: Viewer (no training)";
                    }
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
                if (!scene_->hasModel()) {
                    return "No model available";
                }

                std::string tensor_name = "";
                if (command.length() > 12) {
                    tensor_name = command.substr(12); // Get parameter after "tensor_info "
                }

                if (tensor_name.empty()) {
                    return "Usage: tensor_info <tensor_name>\nAvailable: means, scaling, rotation, shs, opacity";
                }

                std::string tensor_result;
                SplatData* model = scene_->getMutableModel();
                if (!model) {
                    tensor_result = "Model not available";
                    return tensor_result;
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
                    tensor_result = "Unknown tensor: " + tensor_name + "\nAvailable: means, scaling, rotation, shs, opacity";
                    return tensor_result;
                }

                std::ostringstream oss;
                oss << "Tensor '" << tensor_name << "' info:\n";
                oss << "  Shape: [";
                for (int i = 0; i < tensor.dim(); i++) {
                    if (i > 0)
                        oss << ", ";
                    oss << tensor.size(i);
                }
                oss << "]\n";
                oss << "  Device: " << tensor.device() << "\n";
                oss << "  Dtype: " << tensor.dtype() << "\n";
                oss << "  Requires grad: " << (tensor.requires_grad() ? "Yes" : "No") << "\n";

                // Show some statistics if tensor is on CPU or we can move it
                try {
                    auto cpu_tensor = tensor.cpu();
                    auto flat = cpu_tensor.flatten();
                    if (flat.numel() > 0) {
                        oss << "  Min: " << torch::min(flat).item<float>() << "\n";
                        oss << "  Max: " << torch::max(flat).item<float>() << "\n";
                        oss << "  Mean: " << torch::mean(flat).item<float>() << "\n";
                        oss << "  Std: " << torch::std(flat).item<float>();
                    }
                } catch (...) {
                    oss << "  (Statistics unavailable)";
                }

                tensor_result = oss.str();
                return tensor_result;
            }

            return "Unknown command: '" + command + "'. Type 'help' for available commands.";
        });

        // Set up file selection callback
        gui_manager_->setFileSelectedCallback([this](const std::filesystem::path& path, bool is_dataset) {
            event_bus_->publish(LoadFileCommand{path, is_dataset});
        });
    }

    GSViewer::~GSViewer() {
        // Unsubscribe from all events
        for (auto id : event_handler_ids_) {
            // Note: Channels clean up automatically when EventBus is destroyed
        }

        // TrainerManager handles its own cleanup now
        trainer_manager_.reset();

        // Cleanup GUI
        if (gui_manager_) {
            gui_manager_->shutdown();
        }

        std::cout << "GSViewer destroyed." << std::endl;
    }

    void GSViewer::setupEventHandlers() {
        // Subscribe to command events
        event_handler_ids_.push_back(
            event_bus_->subscribe<StartTrainingCommand>(
                [this](const StartTrainingCommand& cmd) { handleStartTrainingCommand(cmd); }));

        event_handler_ids_.push_back(
            event_bus_->subscribe<PauseTrainingCommand>(
                [this](const PauseTrainingCommand& cmd) { handlePauseTrainingCommand(cmd); }));

        event_handler_ids_.push_back(
            event_bus_->subscribe<ResumeTrainingCommand>(
                [this](const ResumeTrainingCommand& cmd) { handleResumeTrainingCommand(cmd); }));

        event_handler_ids_.push_back(
            event_bus_->subscribe<StopTrainingCommand>(
                [this](const StopTrainingCommand& cmd) { handleStopTrainingCommand(cmd); }));

        event_handler_ids_.push_back(
            event_bus_->subscribe<SaveCheckpointCommand>(
                [this](const SaveCheckpointCommand& cmd) { handleSaveCheckpointCommand(cmd); }));

        event_handler_ids_.push_back(
            event_bus_->subscribe<LoadFileCommand>(
                [this](const LoadFileCommand& cmd) { handleLoadFileCommand(cmd); }));

        event_handler_ids_.push_back(
            event_bus_->subscribe<RenderingSettingsChangedEvent>(
                [this](const RenderingSettingsChangedEvent& event) { handleRenderingSettingsChanged(event); }));

        // Subscribe to new events
        event_handler_ids_.push_back(
            event_bus_->subscribe<CameraMovedEvent>(
                [this](const CameraMovedEvent& event) {
                    // Could be used for auto-save camera positions, etc.
                }));

        event_handler_ids_.push_back(
            event_bus_->subscribe<EvaluationCompletedEvent>(
                [this](const EvaluationCompletedEvent& event) {
                    // Update UI with evaluation results
                    if (gui_manager_) {
                        gui_manager_->addConsoleLog(
                            "Evaluation completed - PSNR: %.2f, SSIM: %.3f, LPIPS: %.3f",
                            event.psnr, event.ssim, event.lpips);
                    }
                }));

        event_handler_ids_.push_back(
            event_bus_->subscribe<MemoryUsageEvent>(
                [this](const MemoryUsageEvent& event) {
                    // Could update a memory usage display
                    last_memory_usage_ = event;
                }));

        event_handler_ids_.push_back(
            event_bus_->subscribe<MemoryWarningEvent>(
                [this](const MemoryWarningEvent& event) {
                    if (gui_manager_) {
                        gui_manager_->addConsoleLog(
                            "WARNING: %s", event.message.c_str());
                    }
                }));

        event_handler_ids_.push_back(
            event_bus_->subscribe<ErrorOccurredEvent>(
                [this](const ErrorOccurredEvent& event) {
                    if (gui_manager_) {
                        const char* level = event.severity == ErrorOccurredEvent::Severity::Critical ? "CRITICAL" : event.severity == ErrorOccurredEvent::Severity::Error ? "ERROR"
                                                                                                                                                                          : "WARNING";

                        gui_manager_->addConsoleLog(
                            "%s: %s", level, event.message.c_str());

                        if (event.recovery_suggestion) {
                            gui_manager_->addConsoleLog(
                                "Suggestion: %s", event.recovery_suggestion->c_str());
                        }
                    }
                }));
    }

    void GSViewer::handleStartTrainingCommand(const StartTrainingCommand& cmd) {
        if (trainer_manager_) {
            trainer_manager_->startTraining();
        }
    }

    void GSViewer::handlePauseTrainingCommand(const PauseTrainingCommand& cmd) {
        if (trainer_manager_) {
            trainer_manager_->pauseTraining();
        }
    }

    void GSViewer::handleResumeTrainingCommand(const ResumeTrainingCommand& cmd) {
        if (trainer_manager_) {
            trainer_manager_->resumeTraining();
        }
    }

    void GSViewer::handleStopTrainingCommand(const StopTrainingCommand& cmd) {
        if (trainer_manager_) {
            trainer_manager_->stopTraining();
        }
    }

    void GSViewer::handleSaveCheckpointCommand(const SaveCheckpointCommand& cmd) {
        if (trainer_manager_) {
            trainer_manager_->requestSaveCheckpoint();
        }
    }

    void GSViewer::handleLoadFileCommand(const LoadFileCommand& cmd) {
        if (cmd.is_dataset) {
            loadDataset(cmd.path);
        } else {
            loadPLYFile(cmd.path);
        }
    }

    void GSViewer::handleRenderingSettingsChanged(const RenderingSettingsChangedEvent& event) {
        if (event.fov) {
            config_->fov = *event.fov;
        }
        if (event.scaling_modifier) {
            config_->scaling_modifier = *event.scaling_modifier;
        }
        if (event.antialiasing) {
            anti_aliasing_ = *event.antialiasing;
        }
    }

    void GSViewer::setTrainer(Trainer* trainer) {
        // This method is now deprecated - should not be used
        std::cerr << "Warning: GSViewer::setTrainer is deprecated. Use loadDataset instead." << std::endl;
    }

    void GSViewer::setStandaloneModel(std::unique_ptr<SplatData> model) {
        if (scene_) {
            scene_->setModel(std::move(model));
        }
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
                event_bus_->publish(LogMessageEvent{
                    LogMessageEvent::Level::Error,
                    std::format("Failed to load PLY: {}", splat_result.error()),
                    "GSViewer"});
                return;
            }

            scene_->setModel(std::make_unique<SplatData>(std::move(*splat_result)));
            current_ply_path_ = path;
            current_mode_ = ViewerMode::PLYViewer;

            // Publish scene loaded event
            event_bus_->publish(SceneLoadedEvent{
                scene_.get(),
                path,
                SceneLoadedEvent::SourceType::PLY,
                static_cast<size_t>(scene_->getStandaloneModel()->size())});

            // Publish log message
            event_bus_->publish(LogMessageEvent{
                LogMessageEvent::Level::Info,
                std::format("Loaded PLY with {} Gaussians from {}",
                            static_cast<size_t>(scene_->getStandaloneModel()->size()),
                            path.filename().string()),
                "GSViewer"});

        } catch (const std::exception& e) {
            event_bus_->publish(LogMessageEvent{
                LogMessageEvent::Level::Error,
                std::format("Exception loading PLY: {}", e.what()),
                "GSViewer"});
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
                event_bus_->publish(LogMessageEvent{
                    LogMessageEvent::Level::Error,
                    std::format("Failed to setup training: {}", setup_result.error()),
                    "GSViewer"});
                return;
            }

            // Pass trainer to TrainerManager
            trainer_manager_->setTrainer(std::move(setup_result->trainer));

            // Pass event bus to trainer manager
            trainer_manager_->setEventBus(event_bus_);

            // Link scene to trainer
            scene_->linkToTrainer(trainer_manager_->getTrainer());

            current_dataset_path_ = path;
            current_mode_ = ViewerMode::Training;

            // Get dataset info
            size_t num_images = setup_result->dataset->size().value();
            size_t num_gaussians = trainer_manager_->getTrainer()->get_strategy().get_model().size();

            // Publish scene loaded event
            event_bus_->publish(SceneLoadedEvent{
                scene_.get(),
                path,
                SceneLoadedEvent::SourceType::Dataset,
                num_gaussians});

            // Publish log messages
            event_bus_->publish(LogMessageEvent{
                LogMessageEvent::Level::Info,
                std::format("Loaded dataset with {} images and {} initial Gaussians",
                            num_images, num_gaussians),
                "GSViewer"});

            event_bus_->publish(LogMessageEvent{
                LogMessageEvent::Level::Info,
                std::format("Ready to start training from {}", path.filename().string()),
                "GSViewer"});

            event_bus_->publish(LogMessageEvent{
                LogMessageEvent::Level::Info,
                "Using parameters from command line/config",
                "GSViewer"});

        } catch (const std::exception& e) {
            event_bus_->publish(LogMessageEvent{
                LogMessageEvent::Level::Error,
                std::format("Exception loading dataset: {}", e.what()),
                "GSViewer"});
        }
    }

    void GSViewer::clearCurrentData() {
        // Clear trainer manager (handles stopping training)
        trainer_manager_->clearTrainer();

        // Clear scene
        scene_->clearModel();

        // Reset state
        current_mode_ = ViewerMode::Empty;
        current_ply_path_.clear();
        current_dataset_path_.clear();

        // Clear training info
        if (info_) {
            info_->curr_iterations_ = 0;
            info_->total_iterations_ = 0;
            info_->num_splats_ = 0;
            info_->loss_buffer_.clear();
        }

        // Publish scene cleared event
        event_bus_->publish(SceneClearedEvent{});
    }

    void GSViewer::startTraining() {
        // This method is now deprecated - use event bus
        event_bus_->publish(StartTrainingCommand{});
    }

    bool GSViewer::isGuiActive() const {
        return gui_manager_ && gui_manager_->isAnyWindowActive();
    }

    GSViewer::ViewerMode GSViewer::getCurrentMode() const {
        switch (scene_->getMode()) {
        case Scene::Mode::Empty:
            return ViewerMode::Empty;
        case Scene::Mode::Viewing:
            return ViewerMode::PLYViewer;
        case Scene::Mode::Training:
            return ViewerMode::Training;
        default:
            return ViewerMode::Empty;
        }
    }

    void GSViewer::drawFrame() {
        // Only render if we have a model to render
        if (!scene_->hasModel()) {
            return;
        }

        RenderBoundingBox* render_crop_box = nullptr;
        if (gui_manager_->useCropBox()) {
            render_crop_box = crop_box_.get();
        }
        // Build render request
        RenderingPipeline::RenderRequest request{
            .view_rotation = viewport_.getRotationMatrix(),
            .view_translation = viewport_.getTranslation(),
            .viewport_size = viewport_.windowSize,
            .fov = config_->fov,
            .scaling_modifier = config_->scaling_modifier,
            .antialiasing = anti_aliasing_,
            .render_mode = RenderMode::RGB,
            .crop_box = render_crop_box};

        RenderingPipeline::RenderResult result;

        auto trainer = trainer_manager_->getTrainer();
        if (trainer && trainer->is_running()) {
            std::shared_lock<std::shared_mutex> lock(trainer->getRenderMutex());
            result = scene_->render(request);
        } else {
            result = scene_->render(request);
        }

        if (result.valid) {
            RenderingPipeline::uploadToScreen(result, *screen_renderer_, viewport_.windowSize);
            screen_renderer_->render(quadShader_, viewport_);
        }
        // Render bounding box if enabled
        if (gui_manager_->showCropBox()) {

            glm::ivec2& reso = viewport_.windowSize;
            auto fov_rad = glm::radians(config_->fov);
            auto projection = glm::perspective((float)fov_rad, (float)reso.x / reso.y, .1f, 1000.0f);

            if (!crop_box_->isInitilized()) {
                crop_box_->init();
            }
            // because init can fail
            if (crop_box_->isInitialized()) {
                glm::mat4 view = viewport_.getViewMatrix(); // Replace with actual view matrix

                // Render the bounding box
                crop_box_->render(view, projection);
            }
        }
    }

    void GSViewer::draw() {
        // Initialize GUI on first draw
        static bool gui_initialized = false;
        if (!gui_initialized) {
            gui_manager_->init();
            gui_initialized = true;

            // Set event bus for camera controller
            camera_controller_->setEventBus(event_bus_);

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