#include "visualizer_impl.hpp"
#include "core/command_processor.hpp"
#include "core/data_loading_service.hpp"
#include "scene/scene_manager.hpp"
#include "tools/background_tool.hpp"
#include "tools/crop_box_tool.hpp"
#include <tools/world_transform_tool.hpp>

namespace gs::visualizer {

    VisualizerImpl::VisualizerImpl(const ViewerOptions& options)
        : options_(options),
          viewport_(options.width, options.height),
          window_manager_(std::make_unique<WindowManager>(options.title, options.width, options.height)) {

        // Initialize window manager first
        if (!window_manager_->init()) {
            throw std::runtime_error("Failed to initialize window manager");
        }

        // Create scene manager - it creates its own Scene internally
        scene_manager_ = std::make_unique<SceneManager>();

        // Create trainer manager
        trainer_manager_ = std::make_shared<TrainerManager>();
        trainer_manager_->setViewer(this);
        scene_manager_->setTrainerManager(trainer_manager_.get());

        // Create tool manager
        tool_manager_ = std::make_unique<ToolManager>(this);
        tool_manager_->registerBuiltinTools();
        tool_manager_->addTool("Crop Box");
        tool_manager_->addTool("World Transform");
        tool_manager_->addTool("Background");

        // Create support components
        gui_manager_ = std::make_unique<gui::GuiManager>(this);
        error_handler_ = std::make_unique<ErrorHandler>();
        memory_monitor_ = std::make_unique<MemoryMonitor>();
        memory_monitor_->start();

        // Create and initialize rendering manager early to avoid crashes
        rendering_manager_ = std::make_unique<RenderingManager>();
        rendering_manager_->initialize();

        // Set initial antialiasing
        RenderSettings initial_settings = rendering_manager_->getSettings();
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

        // Tool settings changes - mark dirty
        tools::CropBoxSettingsChanged::when([this](const auto& event) {
            if (rendering_manager_) {
                RenderSettings settings = rendering_manager_->getSettings();
                settings.show_crop_box = event.show_box;
                settings.use_crop_box = event.use_box;
                rendering_manager_->updateSettings(settings);
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
    }

    bool VisualizerImpl::initialize() {
        // Window manager already initialized in constructor

        // Initialize GUI first (sets up ImGui callbacks)
        gui_manager_->init();
        gui_initialized_ = true;

        // Create simplified input controller AFTER ImGui is initialized
        input_controller_ = std::make_unique<InputController>(
            window_manager_->getWindow(), viewport_);
        input_controller_->initialize();
        input_controller_->setTrainingManager(trainer_manager_);

        // Rendering manager already initialized in constructor
        // Just check if it's ready
        if (!rendering_manager_->isInitialized()) {
            rendering_manager_->initialize();
        }

        // Initialize tools AFTER rendering is ready
        tool_manager_->initialize();

        return true;
    }

    void VisualizerImpl::update() {
        window_manager_->updateWindowSize();

        // Update the main viewport with window size
        viewport_.windowSize = window_manager_->getWindowSize();
        viewport_.frameBufferSize = window_manager_->getFramebufferSize();

        // Update tools
        tool_manager_->update();
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

        // Update rendering settings
        RenderSettings settings = rendering_manager_->getSettings();

        // Get crop box state from tool
        if (auto* crop_tool = dynamic_cast<CropBoxTool*>(tool_manager_->getTool("Crop Box"))) {
            settings.show_crop_box = crop_tool->shouldShowBox();
            settings.use_crop_box = crop_tool->shouldUseBox();
        } else {
            settings.show_crop_box = false;
            settings.use_crop_box = false;
        }

        if (auto* world_trans = dynamic_cast<WorldTransformTool*>(tool_manager_->getTool("World Transform"))) {
            settings.show_coord_axes = world_trans->ShouldShowAxes();
        } else {
            settings.show_coord_axes = false;
        }

        rendering_manager_->updateSettings(settings);

        // Get crop box for rendering
        const gs::rendering::IBoundingBox* crop_box_ptr = nullptr;
        if (auto crop_box = getCropBox()) {
            crop_box_ptr = crop_box.get();
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

        // Get coord axes and world 2 user for rendering
        const gs::rendering::ICoordinateAxes* coord_axes_ptr = nullptr;
        if (auto coord_axes = getAxes()) {
            coord_axes_ptr = coord_axes.get();
        }
        const geometry::EuclideanTransform* world_to_user = nullptr;
        if (auto coord_axes = getWorldToUser()) {
            world_to_user = coord_axes.get();
        }

        const BackgroundTool* background_tool = nullptr;
        if (auto* bg_tool = dynamic_cast<BackgroundTool*>(tool_manager_->getTool("Background"))) {
            background_tool = bg_tool;
        }

        // Render
        RenderingManager::RenderContext context{
            .viewport = viewport_,
            .settings = rendering_manager_->getSettings(),
            .crop_box = crop_box_ptr,
            .coord_axes = coord_axes_ptr,
            .world_to_user = world_to_user,
            .viewport_region = has_viewport_region ? &viewport_region : nullptr,
            .has_focus = gui_manager_ && gui_manager_->isViewportFocused(),
            .background_tool = background_tool};

        rendering_manager_->renderFrame(context, scene_manager_.get());

        // Render tools
        tool_manager_->render();

        gui_manager_->render();

        window_manager_->swapBuffers();
        window_manager_->pollEvents();
    }

    void VisualizerImpl::shutdown() {
        tool_manager_->shutdown();
    }

    void VisualizerImpl::LoadProject() {
        if (project_) {
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
        }
    }

    void VisualizerImpl::run() {
        LoadProject(); // load a project if exists
        main_loop_->run();
    }

    void VisualizerImpl::setParameters(const param::TrainingParameters& params) {
        data_loader_->setParameters(params);
    }

    std::expected<void, std::string> VisualizerImpl::loadPLY(const std::filesystem::path& path) {
        return data_loader_->loadPLY(path);
    }

    std::expected<void, std::string> VisualizerImpl::loadDataset(const std::filesystem::path& path) {
        return data_loader_->loadDataset(path);
    }

    void VisualizerImpl::clearScene() {
        data_loader_->clearScene();
    }

    std::shared_ptr<gs::rendering::IBoundingBox> VisualizerImpl::getCropBox() const {
        if (auto* crop_tool = dynamic_cast<CropBoxTool*>(tool_manager_->getTool("Crop Box"))) {
            return crop_tool->getBoundingBox();
        }
        return nullptr;
    }

    std::shared_ptr<const gs::rendering::ICoordinateAxes> VisualizerImpl::getAxes() const {
        if (auto* world_transform = dynamic_cast<WorldTransformTool*>(tool_manager_->getTool("World Transform"))) {
            return world_transform->getAxes();
        }
        return nullptr;
    }

    std::shared_ptr<const geometry::EuclideanTransform> VisualizerImpl::getWorldToUser() const {
        if (auto* world_transform = dynamic_cast<WorldTransformTool*>(tool_manager_->getTool("World Transform"))) {
            if (world_transform->IsTrivialTrans()) {
                return nullptr;
            }
            return world_transform->GetTransform();
        }
        return nullptr;
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
        return project_->writeToFile();
    }

    std::shared_ptr<gs::management::Project> VisualizerImpl::getProject() {
        return project_;
    }

} // namespace gs::visualizer