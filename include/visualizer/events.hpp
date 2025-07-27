#pragma once

#include <filesystem>
#include <glm/glm.hpp>
#include <optional>
#include <string>

namespace gs {

    // Forward declarations
    class Scene;
    class Camera;

    // Note: We don't need SplatData forward declaration here since events don't use it directly

    /**
     * @brief Events use composition - no base class inheritance
     * Each event is a simple struct with data
     */

    // ============================================================================
    // Scene Events
    // ============================================================================

    struct SceneLoadedEvent {
        Scene* scene;
        std::filesystem::path source_path;
        enum class SourceType { PLY,
                                Dataset } source_type;
        size_t num_gaussians;
    };

    struct SceneClearedEvent {
        // No data needed
    };

    // ============================================================================
    // Training Events
    // ============================================================================

    struct TrainingStartedEvent {
        int total_iterations;
    };

    struct TrainingProgressEvent {
        int iteration;
        float loss;
        int num_gaussians;
        bool is_refining;
    };

    struct TrainingPausedEvent {
        int iteration;
    };

    struct TrainingResumedEvent {
        int iteration;
    };

    struct TrainingCompletedEvent {
        int final_iteration;
        float final_loss;
        bool success;
        std::optional<std::string> error_message;
    };

    struct TrainingStoppedEvent {
        int iteration;
        bool user_requested;
    };

    // ============================================================================
    // Model Events
    // ============================================================================

    struct ModelUpdatedEvent {
        int iteration;
        size_t num_gaussians;
    };

    struct CheckpointSavedEvent {
        int iteration;
        std::filesystem::path path;
    };

    // ============================================================================
    // Rendering Events
    // ============================================================================

    struct RenderingSettingsChangedEvent {
        std::optional<float> fov;
        std::optional<float> scaling_modifier;
        std::optional<bool> antialiasing;
    };

    struct ViewportResizedEvent {
        int width;
        int height;
    };

    // ============================================================================
    // UI Events (Commands)
    // ============================================================================

    struct StartTrainingCommand {
        // No data needed - trainer manager will use current state
    };

    struct PauseTrainingCommand {
        // No data needed
    };

    struct ResumeTrainingCommand {
        // No data needed
    };

    struct StopTrainingCommand {
        // No data needed
    };

    struct SaveCheckpointCommand {
        std::optional<int> iteration; // If not specified, use current
    };

    struct LoadFileCommand {
        std::filesystem::path path;
        bool is_dataset; // true for dataset, false for PLY
    };

    // ============================================================================
    // Console/Log Events
    // ============================================================================

    struct LogMessageEvent {
        enum class Level { Info,
                           Warning,
                           Error,
                           Debug };
        Level level;
        std::string message;
        std::optional<std::string> source; // Which component logged this
    };

    struct ConsoleCommandEvent {
        std::string command;
        std::optional<std::string> result;
    };

    // ============================================================================
    // Notification Events
    // ============================================================================

    struct NotificationEvent {
        enum class Type { Info,
                          Success,
                          Warning,
                          Error };
        Type type;
        std::string message;
        int duration_ms = 3000; // How long to show
    };

    // ============================================================================
    // Camera Events
    // ============================================================================

    struct CameraMovedEvent {
        glm::mat3 rotation;
        glm::vec3 translation;
    };

    struct CameraResetEvent {
        // No data needed
    };

} // namespace gs