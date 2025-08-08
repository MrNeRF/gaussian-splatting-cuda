#include "input/input_manager.hpp"
#include <imgui.h>
#include <print>

namespace gs::visualizer {

    InputManager::InputManager(GLFWwindow* window, Viewport& viewport)
        : window_(window),
          viewport_(viewport) {

        setupEventHandlers();
    }

    void InputManager::setupEventHandlers() {
        // Subscribe to GoToCamView events
        events::cmd::GoToCamView::when([this](const auto& event) {
            handleGoToCamView(event);
        });
    }

    void InputManager::handleGoToCamView(const events::cmd::GoToCamView& event) {

        if (!trainer_manager_) {
            std::cerr << "handleGoToCamView: trainer_manager_ was not initilized" << std::endl;
            return;
        }
        const auto cam_data = trainer_manager_->getCamById(event.cam_id);

        if (!cam_data) {
            std::cerr << "cam id " << event.cam_id << " was not found" << std::endl;
            return;
        }
        // Convert torch tensors to glm matrices/vectors
        // cam_data contains WorldToCam transform, but viewport uses CamToWorld
        glm::mat3 world_to_cam_R;
        glm::vec3 world_to_cam_T;

        // Extract rotation matrix from torch tensor (WorldToCam)
        auto R_accessor = cam_data->R().accessor<float, 2>();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                world_to_cam_R[j][i] = R_accessor[i][j]; // Note: glm is column-major
            }
        }

        // Extract translation vector from torch tensor (WorldToCam)
        auto T_accessor = cam_data->T().accessor<float, 1>();
        world_to_cam_T = glm::vec3(T_accessor[0], T_accessor[1], T_accessor[2]);

        // Convert from WorldToCam to CamToWorld convention
        // If [R|T] is WorldToCam, then CamToWorld is [R^T | -R^T * T]
        glm::mat3 cam_to_world_R = glm::transpose(world_to_cam_R);   // R^T
        glm::vec3 cam_to_world_T = -cam_to_world_R * world_to_cam_T; // -R^T * T

        // Set the camera transform (CamToWorld)
        viewport_.camera.R = cam_to_world_R;
        viewport_.camera.t = cam_to_world_T;

        float focal_x = cam_data->focal_x();
        float width = cam_data->image_width();

        // Calculate and set FOV based on focal length and image dimensions
        if (focal_x > 0.0f && width > 0) {
            // Calculate horizontal FOV from focal length
            float fov_horizontal_rad = 2.0f * std::atan(width / (2.0f * focal_x));
            float fov_horizontal_deg = glm::degrees(fov_horizontal_rad);

            // Emit render settings change event with new FOV
            events::ui::RenderSettingsChanged{
                .fov = fov_horizontal_deg,
                .scaling_modifier = std::nullopt,
                .antialiasing = std::nullopt}
                .emit();

            // Log FOV change
            events::notify::Log{
                .level = events::notify::Log::Level::Debug,
                .message = std::format("FOV set to {:.2f} degrees (focal_x: {:.2f}, width: {})",
                                       fov_horizontal_deg, focal_x, width),
                .source = "CameraController"}
                .emit();
        }

        // Force publish the camera change immediately
        events::ui::CameraMove{
            .rotation = viewport_.getRotationMatrix(),
            .translation = viewport_.getTranslation()}
            .emit();

        // Log the action
        events::notify::Log{
            .level = events::notify::Log::Level::Info,
            .message = std::format("Camera moved to view of image: {} (Camera ID: {})",
                                   cam_data->image_name(),
                                   cam_data->uid()),
            .source = "CameraController"}
            .emit();
    }

    InputManager::~InputManager() {
        // Cleanup handled automatically by unique_ptr
    }

    void InputManager::initialize() {
        // Create input handler
        input_handler_ = std::make_unique<InputHandler>(window_);

        // Create camera controller with viewport focus check
        camera_controller_ = std::make_unique<CameraController>(viewport_, viewport_focus_check_);
        camera_controller_->connectToInputHandler(*input_handler_);

        // Pass position check to camera controller
        if (position_check_) {
            camera_controller_->setPositionCheckCallback(position_check_);
        }

        setupInputHandlers();
    }

    void InputManager::setupCallbacks(GuiActiveCheck gui_check, FileDropCallback file_drop) {
        gui_active_check_ = gui_check;
        file_drop_callback_ = file_drop;
    }

    void InputManager::setViewportFocusCheck(std::function<bool()> focus_check) {
        viewport_focus_check_ = focus_check;

        // Update camera controller if it exists
        if (camera_controller_) {
            // Recreate camera controller with new focus check
            camera_controller_ = std::make_unique<CameraController>(viewport_, viewport_focus_check_);
            camera_controller_->connectToInputHandler(*input_handler_);

            // Reapply position check if it exists
            if (position_check_) {
                camera_controller_->setPositionCheckCallback(position_check_);
            }
        }
    }

    void InputManager::setPositionCheck(std::function<bool(double, double)> check) {
        position_check_ = check;

        // Pass to input handler for viewport detection
        if (input_handler_) {
            input_handler_->setViewportCheckCallback(check);
        }

        // Also pass to camera controller
        if (camera_controller_) {
            camera_controller_->setPositionCheckCallback(check);
        }
    }

    void InputManager::updateInputRouting() {
        // Simple focus-based routing for keyboard and scroll
        bool viewport_has_focus = viewport_focus_check_ ? viewport_focus_check_() : false;

        if (viewport_has_focus && !ImGui::GetIO().WantCaptureKeyboard) {
            input_handler_->setInputConsumer(InputHandler::InputConsumer::Viewport);
        } else {
            input_handler_->setInputConsumer(InputHandler::InputConsumer::GUI);
        }
    }

    void InputManager::setupInputHandlers() {
        if (!input_handler_)
            return;

        // Set viewport callbacks
        input_handler_->setViewportCallbacks(
            [this](const InputHandler::MouseButtonEvent& event) {
                if (camera_controller_) {
                    camera_controller_->handleMouseButton(event);
                }
            },
            [this](const InputHandler::MouseMoveEvent& event) {
                if (camera_controller_) {
                    camera_controller_->handleMouseMove(event);
                }
            },
            [this](const InputHandler::MouseScrollEvent& event) {
                if (camera_controller_) {
                    camera_controller_->handleMouseScroll(event);
                }
            },
            [this](const InputHandler::KeyEvent& event) {
                if (camera_controller_) {
                    camera_controller_->handleKey(event);
                }
            });

        // GUI just uses ImGui's input handling, no special callbacks needed

        // File drop handler
        input_handler_->setFileDropCallback(
            [this](const InputHandler::FileDropEvent& event) {
                handleFileDrop(event);
            });
    }

    void InputManager::handleFileDrop(const InputHandler::FileDropEvent& event) {
        if (!file_drop_callback_)
            return;

        // Process each dropped file
        for (const auto& path_str : event.paths) {
            std::filesystem::path filepath(path_str);

            // Check if it's a PLY file
            if (filepath.extension() == ".ply" || filepath.extension() == ".PLY") {
                std::println("Dropped PLY file: {}", filepath.string());

                if (file_drop_callback_(filepath, false)) {
                    // Log the action
                    events::notify::Log{
                        .level = events::notify::Log::Level::Info,
                        .message = std::format("Loaded PLY file via drag-and-drop: {}",
                                               filepath.filename().string()),
                        .source = "InputManager"}
                        .emit();
                    return;
                }
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

                    if (file_drop_callback_(filepath, true)) {
                        // Log the action
                        events::notify::Log{
                            .level = events::notify::Log::Level::Info,
                            .message = std::format("Loaded {} dataset via drag-and-drop: {}",
                                                   is_colmap_dataset ? "COLMAP" : "Transforms",
                                                   filepath.filename().string()),
                            .source = "InputManager"}
                            .emit();
                        return;
                    }
                }
            }
        }
    }

} // namespace gs::visualizer