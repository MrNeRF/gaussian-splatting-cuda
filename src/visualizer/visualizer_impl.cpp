#include "visualizer_impl.hpp"
#include "core/command_processor.hpp"
#include "core/data_loading_service.hpp"
#include "core/model_providers.hpp"
#include "tools/crop_box_tool.hpp"

#include <tools/world_transform_tool.hpp>

namespace gs::visualizer {

    VisualizerImpl::VisualizerImpl(const ViewerOptions& options)
        : options_(options),
          viewport_(options.width, options.height),
          window_manager_(std::make_unique<WindowManager>(options.title, options.width, options.height)) {

        // Create state manager first
        state_manager_ = std::make_unique<ViewerStateManager>();

        // Set up compatibility pointers
        info_ = state_manager_->getTrainingInfo();
        config_ = state_manager_->getRenderingConfig();
        anti_aliasing_ = options.antialiasing;
        state_manager_->setAntiAliasing(options.antialiasing);

        // Create scene manager
        scene_manager_ = std::make_unique<SceneManager>();
        auto scene = std::make_unique<Scene>();
        scene_manager_->setScene(std::move(scene));

        // Create trainer manager
        trainer_manager_ = std::make_unique<TrainerManager>();
        trainer_manager_->setViewer(this);
        scene_manager_->setTrainerManager(trainer_manager_.get());

        // Create tool manager
        tool_manager_ = std::make_unique<ToolManager>(this);
        tool_manager_->registerBuiltinTools();
        tool_manager_->addTool("Crop Box"); // Add crop box by default
        tool_manager_->addTool("World Transform");

        // Create support components
        gui_manager_ = std::make_unique<gui::GuiManager>(this);
        error_handler_ = std::make_unique<ErrorHandler>();
        memory_monitor_ = std::make_unique<MemoryMonitor>();
        memory_monitor_->start();

        // Create rendering manager
        rendering_manager_ = std::make_unique<RenderingManager>();

        // Create command processor
        command_processor_ = std::make_unique<CommandProcessor>(scene_manager_.get());

        // Create data loading service
        data_loader_ = std::make_unique<DataLoadingService>(scene_manager_.get(), state_manager_.get());

        // Create main loop
        main_loop_ = std::make_unique<MainLoop>();
        main_loop_->setTargetFPS(options.target_fps);

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
        gui_manager_->setScriptExecutor([this](const std::string& command) -> std::string {
            return command_processor_->processCommand(command);
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

        // Render settings changes are handled by ViewerStateManager
        // but we need to sync with rendering manager
        ui::RenderSettingsChanged::when([this](const auto&) {
            auto settings = rendering_manager_->getSettings();

            // Get current values from state manager
            settings.fov = state_manager_->getRenderingConfig()->getFovDegrees();
            settings.scaling_modifier = state_manager_->getRenderingConfig()->getScalingModifier();
            settings.antialiasing = state_manager_->isAntiAliasingEnabled();

            // Update compatibility flag
            anti_aliasing_ = settings.antialiasing;

            rendering_manager_->updateSettings(settings);
        });

        // UI events
        ui::CameraMove::when([this](const auto&) {
            // Could be used for auto-save camera positions
        });

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

        // Trainer ready
        internal::TrainerReady::when([this](const auto&) {
            internal::TrainingReadyToStart{}.emit();
        });
    }

    bool VisualizerImpl::initialize() {
        if (!window_manager_->init()) {
            return false;
        }

        // Create input manager
        input_manager_ = std::make_unique<InputManager>(window_manager_->getWindow(), viewport_);
        input_manager_->initialize();

        // NOW set up input callbacks after input_manager_ is created
        input_manager_->setupCallbacks(
            [this]() { return gui_manager_ && gui_manager_->isAnyWindowActive(); },
            [this](const std::filesystem::path& path, bool is_dataset) {
                // The actual loading is now handled by DataLoadingService via events
                events::cmd::LoadFile{.path = path, .is_dataset = is_dataset}.emit();

                if (gui_manager_) {
                    gui_manager_->showScriptingConsole(true);
                }
                return true;
            });

        rendering_manager_->initialize();

        // Initialize tools
        tool_manager_->initialize();

        // Initialize GUI
        gui_manager_->init();
        gui_initialized_ = true;

        return true;
    }

    void VisualizerImpl::update() {
        window_manager_->updateWindowSize();
        viewport_.windowSize = window_manager_->getWindowSize();
        viewport_.frameBufferSize = window_manager_->getFramebufferSize();

        // Update tools
        tool_manager_->update();
    }

    void VisualizerImpl::render() {
        // Update rendering settings from state manager
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

        settings.antialiasing = state_manager_->isAntiAliasingEnabled();
        settings.fov = state_manager_->getRenderingConfig()->getFovDegrees();
        settings.scaling_modifier = state_manager_->getRenderingConfig()->getScalingModifier();
        rendering_manager_->updateSettings(settings);

        // Get crop box for rendering
        RenderBoundingBox* crop_box_ptr = nullptr;
        if (auto crop_box = getCropBox()) {
            crop_box_ptr = crop_box.get();
        }

        // Get crop box for rendering
        const RenderCoordinateAxes* coord_axes_ptr = nullptr;
        if (auto coord_axes = getAxes()) {
            coord_axes_ptr = coord_axes.get();
        }
        const geometry::EuclideanTransform* world_to_user = nullptr;
        if (auto coord_axes = getWorldToUser()) {
            world_to_user = coord_axes.get();
        }

        // Render
        RenderingManager::RenderContext context{
            .viewport = viewport_,
            .settings = rendering_manager_->getSettings(),
            .crop_box = crop_box_ptr,
            .coord_axes = coord_axes_ptr,
            .world_to_user = world_to_user};

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

    void VisualizerImpl::run() {
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

    std::shared_ptr<RenderBoundingBox> VisualizerImpl::getCropBox() const {
        if (auto* crop_tool = dynamic_cast<CropBoxTool*>(tool_manager_->getTool("Crop Box"))) {
            return crop_tool->getBoundingBox();
        }
        return nullptr;
    }

    std::shared_ptr<const RenderCoordinateAxes> VisualizerImpl::getAxes() const {
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

} // namespace gs::visualizer