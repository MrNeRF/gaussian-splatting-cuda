#include "visualizer_impl.hpp"
#include "config.h" // Include generated config
#include "core/model_providers.hpp"
#include "core/splat_data.hpp"
#include "core/training_setup.hpp"
#include "internal/resource_paths.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <print>
#include <sstream>
#include <thread>

#include <cuda_runtime.h>

#ifdef CUDA_GL_INTEROP_ENABLED
#include "rendering/cuda_gl_interop.hpp"
#endif

namespace gs::visualizer {

    VisualizerImpl::VisualizerImpl(const ViewerOptions& options)
        : options_(options),
          title_(options.title),
          viewport_(options.width, options.height),
          window_manager_(std::make_unique<WindowManager>(options.title, options.width, options.height)) {

        config_ = std::make_shared<RenderingConfig>();
        info_ = std::make_shared<TrainingInfo>();
        notifier_ = std::make_shared<ViewerNotifier>();
        crop_box_ = std::make_shared<RenderBoundingBox>();

        // Create scene manager
        scene_manager_ = std::make_unique<SceneManager>();

        // Create scene and give it to scene manager
        auto scene = std::make_unique<Scene>();
        scene_manager_->setScene(std::move(scene));

        // Create trainer manager
        trainer_manager_ = std::make_unique<TrainerManager>();
        trainer_manager_->setViewer(this);

        // Link trainer manager to scene manager
        scene_manager_->setTrainerManager(trainer_manager_.get());

        setFrameRate(options.target_fps);
        anti_aliasing_ = options.antialiasing;

        // Create GUI manager
        gui_manager_ = std::make_unique<gui::GuiManager>(this);

        // Create error handler
        error_handler_ = std::make_unique<ErrorHandler>();

        // Create memory monitor and start it
        memory_monitor_ = std::make_unique<MemoryMonitor>();
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
                // Use event query instead of direct access
                try {
                    auto response = Query<events::query::GetTrainerState, events::query::TrainerState>()
                        .send(events::query::GetTrainerState{});

                    result << "Training Status:\n";
                    result << "  State: " << [](const events::query::TrainerState::State& state) -> const char* {
                        switch (state) {
                        case events::query::TrainerState::State::Idle: return "Idle";
                        case events::query::TrainerState::State::Ready: return "Ready";
                        case events::query::TrainerState::State::Running: return "Running";
                        case events::query::TrainerState::State::Paused: return "Paused";
                        case events::query::TrainerState::State::Completed: return "Completed";
                        case events::query::TrainerState::State::Error: return "Error";
                        default: return "Unknown";
                        }
                    }(response.state) << "\n";
                    result << "  Current Iteration: " << response.current_iteration << "\n";
                    result << "  Current Loss: " << std::fixed << std::setprecision(6) << response.current_loss;
                    if (response.error_message) {
                        result << "\n  Error: " << *response.error_message;
                    }
                } catch (...) {
                    result << "No trainer available (viewer mode)";
                }
                return result.str();
            }

            if (command == "model_info") {
                // First query scene state
                try {
                    auto stateResponse = Query<events::query::GetSceneInfo, events::query::SceneInfo>()
                        .send(events::query::GetSceneInfo{});

                    if (stateResponse.has_model) {
                        result << "Scene Information:\n";
                        result << "  Type: " << [&]() {
                            switch (stateResponse.type) {
                            case events::query::SceneInfo::Type::None: return "None";
                            case events::query::SceneInfo::Type::PLY: return "PLY";
                            case events::query::SceneInfo::Type::Dataset: return "Dataset";
                            default: return "Unknown";
                            }
                        }() << "\n";
                        result << "  Source: " << stateResponse.source_path.filename().string() << "\n";
                        result << "  Number of Gaussians: " << stateResponse.num_gaussians << "\n";

                        if (stateResponse.is_training) {
                            result << "  Training Mode: Active\n";
                        }

                        // Then query model details if needed
                        auto modelResponse = Query<events::query::GetModelInfo, events::query::ModelInfo>()
                            .send(events::query::GetModelInfo{});

                        if (modelResponse.has_model) {
                            result << "  SH Degree: " << modelResponse.sh_degree << "\n";
                            result << "  Scene Scale: " << modelResponse.scene_scale << "\n";
                        }
                    } else {
                        result << "No scene loaded";
                    }
                } catch (...) {
                    result << "Failed to query scene information";
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
                if (!scene_manager_->hasScene()) {
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
                SplatData* model = scene_manager_->getScene()->getMutableModel();
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
            events::cmd::LoadFile{.path = path, .is_dataset = is_dataset}.emit();
        });
    }

    VisualizerImpl::~VisualizerImpl() {
        // TrainerManager handles its own cleanup now
        trainer_manager_.reset();

        // Cleanup GUI
        if (gui_manager_) {
            gui_manager_->shutdown();
        }

        std::cout << "Visualizer destroyed." << std::endl;
    }

    void VisualizerImpl::setupEventHandlers() {
        using namespace events;

        // Commands
        cmd::StartTraining::when([this](const auto&) {
            if (trainer_manager_) {
                trainer_manager_->startTraining();
            }
        });

        cmd::PauseTraining::when([this](const auto&) {
            if (trainer_manager_) {
                trainer_manager_->pauseTraining();
            }
        });

        cmd::ResumeTraining::when([this](const auto&) {
            if (trainer_manager_) {
                trainer_manager_->resumeTraining();
            }
        });

        cmd::StopTraining::when([this](const auto&) {
            if (trainer_manager_) {
                trainer_manager_->stopTraining();
            }
        });

        cmd::SaveCheckpoint::when([this](const auto&) {
            if (trainer_manager_) {
                trainer_manager_->requestSaveCheckpoint();
            }
        });

        cmd::LoadFile::when([this](const auto& cmd) {
            if (cmd.is_dataset) {
                loadDatasetInternal(cmd.path);
            } else {
                loadPLYFile(cmd.path);
            }
        });

        ui::RenderSettingsChanged::when([this](const auto& event) {
            if (event.fov) {
                config_->fov = *event.fov;
            }
            if (event.scaling_modifier) {
                config_->scaling_modifier = *event.scaling_modifier;
            }
            if (event.antialiasing) {
                anti_aliasing_ = *event.antialiasing;
            }
        });

        // Subscribe to new events
        ui::CameraMove::when([this](const auto&) {
            // Could be used for auto-save camera positions, etc.
        });

        state::EvaluationCompleted::when([this](const auto& event) {
            // Update UI with evaluation results
            if (gui_manager_) {
                gui_manager_->addConsoleLog(
                    "Evaluation completed - PSNR: %.2f, SSIM: %.3f, LPIPS: %.3f",
                    event.psnr, event.ssim, event.lpips);
            }
        });

        state::MemoryUsage::when([this](const auto&) {
            // Could update a memory usage display
        });

        notify::MemoryWarning::when([this](const auto& event) {
            if (gui_manager_) {
                gui_manager_->addConsoleLog(
                    "WARNING: %s", event.message.c_str());
            }
        });

        notify::Error::when([this](const auto& event) {
            if (gui_manager_) {
                gui_manager_->addConsoleLog(
                    "ERROR: %s", event.message.c_str());

                if (!event.details.empty()) {
                    gui_manager_->addConsoleLog(
                        "Details: %s", event.details.c_str());
                }
            }
        });

        // Subscribe to training progress events
        state::TrainingProgress::when([this](const auto& event) {
            // Update training info from event
            if (info_) {
                info_->updateProgress(event.iteration, event.num_gaussians);
                info_->updateNumSplats(event.num_gaussians);
                info_->updateLoss(event.loss);
            }
        });

        // Subscribe to trainer ready event
        internal::TrainerReady::when([this](const auto&) {
            // Trainer is ready, signal it can start
            internal::TrainingReadyToStart{}.emit();
        });
    }

    void VisualizerImpl::loadPLYFile(const std::filesystem::path& path) {
        try {
            std::println("Loading PLY file: {}", path.string());

            scene_manager_->loadPLY(path);
            current_ply_path_ = path;
            current_mode_ = ViewerMode::PLYViewer;

        } catch (const std::exception& e) {
            events::notify::Error{
                .message = std::format("Failed to load PLY: {}", e.what()),
                .details = std::format("Path: {}", path.string())
            }.emit();
        }
    }

    void VisualizerImpl::loadDatasetInternal(const std::filesystem::path& path) {
        try {
            std::println("Loading dataset from: {}", path.string());

            scene_manager_->loadDataset(path, params_);
            current_dataset_path_ = path;
            current_mode_ = ViewerMode::Training;

        } catch (const std::exception& e) {
            events::notify::Error{
                .message = std::format("Failed to load dataset: {}", e.what()),
                .details = std::format("Path: {}", path.string())
            }.emit();
        }
    }

    void VisualizerImpl::clearCurrentData() {
        scene_manager_->clearScene();
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
    }

    void VisualizerImpl::startTraining() {
        // This method is now deprecated - use event bus
        events::cmd::StartTraining{}.emit();
    }

    bool VisualizerImpl::isGuiActive() const {
        return gui_manager_ && gui_manager_->isAnyWindowActive();
    }

    VisualizerImpl::ViewerMode VisualizerImpl::getCurrentMode() const {
        if (!scene_manager_)
            return ViewerMode::Empty;

        auto state = scene_manager_->getCurrentState();

        switch (state.type) {
        case SceneManager::SceneType::None:
            return ViewerMode::Empty;
        case SceneManager::SceneType::PLY:
            return ViewerMode::PLYViewer;
        case SceneManager::SceneType::Dataset:
            return ViewerMode::Training;
        default:
            return ViewerMode::Empty;
        }
    }

    bool VisualizerImpl::handleFileDrop(const InputHandler::FileDropEvent& event) {
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
                    events::notify::Log{
                        .level = events::notify::Log::Level::Info,
                        .message = std::format("Loaded PLY file via drag-and-drop: {}", filepath.filename().string()),
                        .source = "VisualizerImpl"
                    }.emit();
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
                    loadDatasetInternal(filepath);

                    // Log the action
                    if (gui_manager_) {
                        gui_manager_->showScriptingConsole(true);
                        events::notify::Log{
                            .level = events::notify::Log::Level::Info,
                            .message = std::format("Loaded {} dataset via drag-and-drop: {}",
                                        is_colmap_dataset ? "COLMAP" : "Transforms",
                                        filepath.filename().string()),
                            .source = "VisualizerImpl"
                        }.emit();
                    }

                    // Only process the first valid dataset if multiple were dropped
                    return true;
                }
            }
        }
        return false;
    }

    bool VisualizerImpl::init() {
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

    void VisualizerImpl::updateWindowSize() {
        window_manager_->updateWindowSize();
        viewport_.windowSize = window_manager_->getWindowSize();
        viewport_.frameBufferSize = window_manager_->getFramebufferSize();
    }

    float VisualizerImpl::getGPUUsage() {
        size_t free_byte, total_byte;
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free_byte, &total_byte);
        size_t used_byte = total_byte - free_byte;
        float gpuUsage = used_byte / (float)total_byte * 100;

        return gpuUsage;
    }

    void VisualizerImpl::setFrameRate(const int fps) {
        target_fps_ = fps;
        frame_time_ = 1000 / target_fps_;
    }

    void VisualizerImpl::controlFrameRate() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time_).count();
        if (duration < frame_time_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(frame_time_ - duration));
        }
        last_time_ = std::chrono::high_resolution_clock::now();
    }

    void VisualizerImpl::drawFrame() {
        // Only render if we have a scene
        if (!scene_manager_->hasScene()) {
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
            result = scene_manager_->render(request);
        } else {
            result = scene_manager_->render(request);
        }

        if (result.valid) {
            RenderingPipeline::uploadToScreen(result, *screen_renderer_, viewport_.windowSize);
            screen_renderer_->render(quad_shader_, viewport_);
        }
        // Render bounding box if enabled
        if (gui_manager_->showCropBox()) {

            glm::ivec2& reso = viewport_.windowSize;

            // this happened to me on debug - so I wanted to add protection
            if (reso.x <= 0 || reso.y <= 0) {
                return;
            }

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

    void VisualizerImpl::draw() {
        // Initialize GUI on first draw
        static bool gui_initialized = false;
        if (!gui_initialized) {
            gui_manager_->init();
            gui_initialized = true;

            // GUI gets first priority
            input_handler_->addMouseButtonHandler(
                [this](const InputHandler::MouseButtonEvent&) {
                    return isGuiActive(); // Consume if GUI is active
                });

            input_handler_->addMouseMoveHandler(
                [this](const InputHandler::MouseMoveEvent&) {
                    return isGuiActive(); // Consume if GUI is active
                });

            input_handler_->addMouseScrollHandler(
                [this](const InputHandler::MouseScrollEvent&) {
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

    void VisualizerImpl::run() {
        if (!init()) {
            std::cerr << "Viewer initialization failed!" << std::endl;
            return;
        }

        std::string shader_path = std::string(PROJECT_ROOT_PATH) + "/src/visualizer/rendering/shaders/";
        quad_shader_ = std::make_shared<Shader>(
            (gs::visualizer::getShaderPath("screen_quad.vert")).c_str(),
            (gs::visualizer::getShaderPath("screen_quad.frag")).c_str(),
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

    void VisualizerImpl::setParameters(const param::TrainingParameters& params) {
        params_ = params;
    }

    std::expected<void, std::string> VisualizerImpl::loadPLY(const std::filesystem::path& path) {
        try {
            loadPLYFile(path);
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to load PLY: {}", e.what()));
        }
    }

    std::expected<void, std::string> VisualizerImpl::loadDataset(const std::filesystem::path& path) {
        try {
            loadDatasetInternal(path);
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to load dataset: {}", e.what()));
        }
    }

    void VisualizerImpl::clearScene() {
        clearCurrentData();
    }

} // namespace gs::visualizer