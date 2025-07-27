#pragma once

#include <filesystem>
#include <glm/glm.hpp>
#include <optional>
#include <string>

namespace gs {

    class Scene;
    class Camera;

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

    struct RenderModeChangedEvent {
        std::string old_mode;
        std::string new_mode;
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

    // ============================================================================
    // Data Loading Events
    // ============================================================================

    struct DatasetLoadStartedEvent {
        std::filesystem::path path;
        enum class Type { Colmap,
                          Blender,
                          PLY } type;
    };

    struct DatasetLoadProgressEvent {
        std::filesystem::path path;
        float progress;           // 0.0 to 1.0
        std::string current_step; // "Reading cameras", "Loading images", etc.
    };

    struct DatasetLoadCompletedEvent {
        std::filesystem::path path;
        bool success;
        std::optional<std::string> error_message;
        size_t num_images;
        size_t num_points;
    };

    // ============================================================================
    // Evaluation Events
    // ============================================================================

    struct EvaluationStartedEvent {
        int iteration;
        size_t num_images;
    };

    struct EvaluationProgressEvent {
        int iteration;
        size_t current_image;
        size_t total_images;
    };

    struct EvaluationCompletedEvent {
        int iteration;
        float psnr;
        float ssim;
        float lpips;
        float elapsed_time;
        int num_gaussians;
    };

    // ============================================================================
    // Memory Events
    // ============================================================================

    struct MemoryUsageEvent {
        size_t gpu_used_bytes;
        size_t gpu_total_bytes;
        float gpu_usage_percent;
        size_t ram_used_bytes;
        size_t ram_total_bytes;
        float ram_usage_percent;
    };

    struct MemoryWarningEvent {
        enum class Type { GPU,
                          RAM };
        Type type;
        float usage_percent;
        std::string message;
    };

    // ============================================================================
    // Performance Events
    // ============================================================================

    struct FrameRenderedEvent {
        float render_time_ms;
        float fps;
        int num_gaussians_rendered;
    };

    struct PerformanceReportEvent {
        float avg_fps;
        float avg_render_time_ms;
        float avg_training_step_time_ms;
        int time_window_seconds; // Stats over this time window
    };

    // ============================================================================
    // Error Events
    // ============================================================================

    struct ErrorOccurredEvent {
        enum class Severity { Warning,
                              Error,
                              Critical };
        enum class Category {
            Rendering,
            Training,
            IO,
            Memory,
            CUDA,
            System,
            Unknown
        };
        Severity severity;
        Category category;
        std::string message;
        std::optional<std::string> details;
        std::optional<std::string> recovery_suggestion;
    };

    struct ErrorRecoveredEvent {
        ErrorOccurredEvent::Category category;
        std::string recovery_action;
    };

    // ============================================================================
    // Crop Box Events
    // ============================================================================

    struct CropBoxChangedEvent {
        glm::vec3 min_bounds;
        glm::vec3 max_bounds;
        bool enabled;
    };

    struct CropBoxVisibilityChangedEvent {
        bool visible;
    };

    struct CropBoxResetEvent {
        // No data needed
    };

} // namespace gs