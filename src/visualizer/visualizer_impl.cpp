#include "visualizer_impl.hpp"
#include "core/command_processor.hpp"
#include "core/data_loading_service.hpp"
#include "core/logger.hpp"
#include "scene/scene_manager.hpp"

namespace gs::visualizer {

    VisualizerImpl::VisualizerImpl(const ViewerOptions& options)
        : options_(options),
          viewport_(options.width, options.height),
          window_manager_(std::make_unique<WindowManager>(options.title, options.width, options.height)) {

        LOG_DEBUG("Creating visualizer");
        // Create scene manager - it creates its own Scene internally
        scene_manager_ = std::make_unique<SceneManager>();

        // Create trainer manager
        trainer_manager_ = std::make_shared<TrainerManager>();
        trainer_manager_->setViewer(this);
        scene_manager_->setTrainerManager(trainer_manager_.get());

        // Create support components
        gui_manager_ = std::make_unique<gui::GuiManager>(this);
        error_handler_ = std::make_unique<ErrorHandler>();
        memory_monitor_ = std::make_unique<MemoryMonitor>();
        memory_monitor_->start();

        // Create rendering manager with initial antialiasing setting
        rendering_manager_ = std::make_unique<RenderingManager>();

        // Set initial antialiasing
        RenderSettings initial_settings;
        initial_settings.antialiasing = options.antialiasing;
        rendering_manager_->updateSettings(initial_settings);

        // Create command processor
        command_processor_ = std::make_unique<CommandProcessor>(scene_manager_.get());

        // Create data loading service - no longer needs state_manager
        data_loader_ = std::make_unique<DataLoadingService>(scene_manager_.get());

        // Create main loop
        main_loop_ = std::make_unique<MainLoop>();

        // Setup connections
        setupEventHandlers();
        setupComponentConnections();
    }

    VisualizerImpl::~VisualizerImpl() {
        trainer_manager_.reset();
        if (gui_manager_) {
            gui_manager_->shutdown();
        }
        std::cout << "Visualizer destroyed." << std::endl;
    }

    void VisualizerImpl::setupComponentConnections() {
        // Set up main loop callbacks
        main_loop_->setInitCallback([this]() { return initialize(); });
        main_loop_->setUpdateCallback([this]() { update(); });
        main_loop_->setRenderCallback([this]() { render(); });
        main_loop_->setShutdownCallback([this]() { shutdown(); });
        main_loop_->setShouldCloseCallback([this]() { return window_manager_->shouldClose(); });

        // Set up GUI connections
        gui_manager_->setScriptExecutor([this](const std::string& cmd) {
            return command_processor_->processCommand(cmd);
        });

        gui_manager_->setFileSelectedCallback([this](const std::filesystem::path& path, bool is_dataset) {
            events::cmd::LoadFile{.path = path, .is_dataset = is_dataset}.emit();
        });
    }

    void VisualizerImpl::setupEventHandlers() {
        using namespace events;

        // Training commands
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

        // Render settings changes
        ui::RenderSettingsChanged::when([this](const auto& event) {
            if (rendering_manager_) {
                // The rendering manager handles this internally now
                // Just need to mark dirty which happens in its event handler
            }
        });

        // Camera moves - mark dirty
        ui::CameraMove::when([this](const auto&) {
            if (rendering_manager_) {
                rendering_manager_->markDirty();
            }
        });

        // Scene changes - mark dirty
        state::SceneChanged::when([this](const auto&) {
            if (window_manager_) {
                window_manager_->requestRedraw();
            }
            if (rendering_manager_) {
                rendering_manager_->markDirty();
            }
        });

        // Evaluation completed
        state::EvaluationCompleted::when([this](const auto& event) {
            if (gui_manager_) {
                gui_manager_->addConsoleLog(
                    "Evaluation completed - PSNR: %.2f, SSIM: %.3f, LPIPS: %.3f",
                    event.psnr, event.ssim, event.lpips);
            }
        });

        // Notifications
        notify::MemoryWarning::when([this](const auto& event) {
            if (gui_manager_) {
                gui_manager_->addConsoleLog("WARNING: %s", event.message.c_str());
            }
        });

        notify::Error::when([this](const auto& event) {
            if (gui_manager_) {
                gui_manager_->addConsoleLog("ERROR: %s", event.message.c_str());
                if (!event.details.empty()) {
                    gui_manager_->addConsoleLog("Details: %s", event.details.c_str());
                }
            }
        });

        internal::TrainerReady::when([this](const auto&) {
            internal::TrainingReadyToStart{}.emit();
        });

        // Training progress - don't mark dirty, let throttling handle it
        state::TrainingProgress::when([this](const auto& event) {
            // Just update loss buffer, don't force render
            // The 1 FPS throttle will handle rendering
        });

        // Listen for file load commands
        cmd::LoadProject::when([this](const auto& cmd) {
            handleLoadProjectCommand(cmd);
        });
    }

    bool VisualizerImpl::initialize() {
        // Initialize window first
        if (!window_initialized_) {
            if (!window_manager_->init()) {
                return false;
            }
            window_initialized_ = true;
        }

        // Initialize GUI (sets up ImGui callbacks)
        if (!gui_initialized_) {
            gui_manager_->init();
            gui_initialized_ = true;
        }

        // Create simplified input controller AFTER ImGui is initialized
        if (!input_controller_) {
            input_controller_ = std::make_unique<InputController>(
                window_manager_->getWindow(), viewport_);
            input_controller_->initialize();
            input_controller_->setTrainingManager(trainer_manager_);
        }

        // Initialize rendering
        if (!rendering_manager_->isInitialized()) {
            rendering_manager_->initialize();
        }

        return true;
    }

    void VisualizerImpl::update() {
        window_manager_->updateWindowSize();

        // Update the main viewport with window size
        viewport_.windowSize = window_manager_->getWindowSize();
        viewport_.frameBufferSize = window_manager_->getFramebufferSize();
    }

    void VisualizerImpl::render() {
        // Update input controller with viewport bounds
        if (gui_manager_) {
            auto pos = gui_manager_->getViewportPos();
            auto size = gui_manager_->getViewportSize();
            input_controller_->updateViewportBounds(pos.x, pos.y, size.x, size.y);
        }

        // Update point cloud mode in input controller
        auto* rendering_manager = getRenderingManager();
        if (rendering_manager) {
            const auto& settings = rendering_manager->getSettings();
            input_controller_->setPointCloudMode(settings.point_cloud_mode);
        }

        // Get viewport region from GUI
        ViewportRegion viewport_region;
        bool has_viewport_region = false;
        if (gui_manager_) {
            ImVec2 pos = gui_manager_->getViewportPos();
            ImVec2 size = gui_manager_->getViewportSize();

            viewport_region.x = pos.x;
            viewport_region.y = pos.y;
            viewport_region.width = size.x;
            viewport_region.height = size.y;

            has_viewport_region = true;
        }

        // Create render context
        RenderingManager::RenderContext context{
            .viewport = viewport_,
            .settings = rendering_manager_->getSettings(),
            .viewport_region = has_viewport_region ? &viewport_region : nullptr,
            .has_focus = gui_manager_ && gui_manager_->isViewportFocused()};

        rendering_manager_->renderFrame(context, scene_manager_.get());

        gui_manager_->render();

        window_manager_->swapBuffers();
        window_manager_->pollEvents();
    }

    void VisualizerImpl::shutdown() {
        // Nothing to shutdown now that tools are gone
    }

    bool VisualizerImpl::LoadProject() {
        if (project_) {
            try {
                // slicing intended
                auto dataset = static_cast<const param::DatasetConfig&>(project_->getProjectData().data_set_info);
                if (!dataset.data_path.empty()) {
                    auto result = loadDataset(dataset.data_path);
                    if (!result) {
                        std::println(stderr, "Error: {}", result.error());
                        return false;
                    }
                }
                // load plys
                auto plys = project_->getPlys();
                // sort according to iter numbers
                std::sort(plys.begin(), plys.end(),
                          [](const gs::management::PlyData& a, const gs::management::PlyData& b) {
                              return a.ply_training_iter_number < b.ply_training_iter_number;
                          });

                if (!plys.empty()) {
                    scene_manager_->changeContentType(SceneManager::ContentType::PLYFiles);
                }
                // set all of the nodes to invisible except the last one
                for (auto it = plys.begin(); it != plys.end(); ++it) {
                    std::string ply_name = it->ply_name;

                    bool is_last = (std::next(it) == plys.end());
                    scene_manager_->addPLY(it->ply_path, ply_name, is_last);
                    scene_manager_->setPLYVisibility(ply_name, is_last);
                }
            } catch (const std::exception& e) {
                std::println(stderr, "Failed to load project: {}", e.what());
                return false;
            }

            return true;
        }
        return false;
    }

    void VisualizerImpl::run() {
        // Ensure basic initialization before running
        if (!window_initialized_) {
            if (!window_manager_->init()) {
                throw std::runtime_error("Failed to initialize window manager");
            }
            window_initialized_ = true;
        }

        // Initialize rendering
        if (!rendering_manager_->isInitialized()) {
            rendering_manager_->initialize();
        }

        LoadProject(); // load a project if exists
        main_loop_->run();
    }

    void VisualizerImpl::setParameters(const param::TrainingParameters& params) {
        data_loader_->setParameters(params);
    }

    std::expected<void, std::string> VisualizerImpl::loadPLY(const std::filesystem::path& path) {
        // Ensure proper initialization order before loading PLY
        // This handles command line PLY loading
        if (!window_initialized_) {
            if (!window_manager_->init()) {
                return std::unexpected("Failed to initialize window manager");
            }
            window_initialized_ = true;
        }

        // Initialize rendering
        if (!rendering_manager_->isInitialized()) {
            rendering_manager_->initialize();
        }

        return data_loader_->loadPLY(path);
    }

    std::expected<void, std::string> VisualizerImpl::loadDataset(const std::filesystem::path& path) {
        // Ensure proper initialization order before loading dataset
        if (!window_initialized_) {
            if (!window_manager_->init()) {
                return std::unexpected("Failed to initialize window manager");
            }
            window_initialized_ = true;
        }

        // Initialize rendering
        if (!rendering_manager_->isInitialized()) {
            rendering_manager_->initialize();
        }

        return data_loader_->loadDataset(path);
    }

    void VisualizerImpl::clearScene() {
        data_loader_->clearScene();
    }


    bool VisualizerImpl::openProject(const std::filesystem::path& path) {

        auto project = std::make_shared<gs::management::Project>();

        if (!project) {
            std::cerr << "openProject: error creating project " << std::endl;
            return false;
        }
        if (!project->readFromFile(path)) {
            std::cerr << "reading  project file failed " << path.string() << std::endl;
            return false;
        }
        if (!project->validateProjectData()) {
            std::cerr << "failed to validate project" << std::endl;
            return false;
        }

        project_ = project;

        return true;
    }

    bool VisualizerImpl::closeProject(const std::filesystem::path& path) {

        if (!project_) {
            return false;
        }
        if (!path.empty()) {
            project_->setProjectFileName(path);
        }

        return project_->writeToFile();
    }

    std::shared_ptr<gs::management::Project> VisualizerImpl::getProject() {
        return project_;
    }

    void VisualizerImpl::handleLoadProjectCommand(const events::cmd::LoadProject& cmd) {

        bool success = openProject(cmd.path);
        if (!success) {
            std::string error_msg = std::format("Failed opening project: {}", cmd.path.string());
            events::notify::Error{
                .message = error_msg,
                .details = std::format("Path: {}", cmd.path.string())}
            .emit();
        }

        success = LoadProject();
        if (!success) {
            std::string error_msg = std::format("Failed to load project: {}", cmd.path.string());
            events::notify::Error{
                .message = error_msg,
                .details = std::format("Path: {}", cmd.path.string())}
            .emit();
        }
    }
} // namespace gs::visualizer